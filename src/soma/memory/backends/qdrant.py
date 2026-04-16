"""QdrantBackend — Qdrant-backed ``VectorBackend`` adapter.

Three modes, one class:

- ``"memory"`` — embedded in-process Qdrant core. Zero setup,
  volatile. Handy for tests and local-first agents that don't need
  durability past the process lifetime.
- ``"local"`` — on-disk single-node Qdrant in a directory you own.
  Good up to ~20K vectors on consumer hardware; a
  :class:`UserWarning` fires when ``ntotal`` crosses the cap so
  operators have time to migrate.
- ``"http"`` — remote Qdrant (cloud, docker, k8s). The scale path.
  Qdrant stays authoritative on disk; SOMA's WAL is the replay
  source if the collection ever has to be rebuilt.

Features:

- Cosine distance (the only metric MemoryLayer advertises today).
- Opaque SOMA ``node_id`` strings stored as the Qdrant payload
  ``node_id`` field; we generate deterministic int64 point ids from
  an internal monotonic counter so upserts are repeatable.
- ``supports_filter_pushdown=True``; :func:`to_qdrant_filter`
  translates Chroma-style ``where`` clauses to ``qm.Filter``.
- Snapshot writes ``qdrant.snapshot`` (local mode) or just a
  ``backend.json`` sidecar (HTTP mode).

qdrant-client is an optional dep — the import is guarded so SOMA
still ships without it. Constructing a ``QdrantBackend`` when the
dep is missing raises :class:`ImportError` with a pip install hint.
"""

from __future__ import annotations

import json
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from soma.memory.backends.qdrant_filter import to_qdrant_filter

# 20K is the documented cap for local-file Qdrant before the embedded
# core's latency starts dominating consumer-hardware retrieve budgets.
# Operators should migrate to HTTP mode past this; we emit a
# UserWarning so the hand-off is visible in logs.
LOCAL_MODE_ENTRY_WARNING_THRESHOLD = 20_000


def _ensure_qdrant_available() -> None:
    try:
        import qdrant_client  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "QdrantBackend requires qdrant-client. "
            "Install with: pip install 'soma[qdrant]'"
        ) from exc


class QdrantBackend:
    """Qdrant-backed :class:`VectorBackend`.

    Parameters
    ----------
    mode:
        ``"memory"``, ``"local"``, or ``"http"``.
    dim:
        Embedding dimension. Pinned at construction; mismatch on
        upsert raises.
    path:
        Directory for local mode. Required when ``mode="local"``.
    url:
        Qdrant server URL (``http://...``). Required for HTTP mode.
    api_key:
        Optional API key forwarded to :class:`QdrantClient`.
    collection:
        Collection name. Defaults to ``"soma_default"`` so a naïve
        constructor just works; MemoryLayer passes the bundle name.
    recreate:
        When True, drop any existing collection before upserting.
        Useful from tests and benchmarks.
    """

    supports_filter_pushdown: bool = True
    name: str = "qdrant"

    def __init__(
        self,
        *,
        mode: str,
        dim: int,
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        recreate: bool = False,
    ) -> None:
        _ensure_qdrant_available()
        if mode not in ("memory", "local", "http"):
            raise ValueError(
                f"QdrantBackend mode must be memory|local|http, got {mode!r}"
            )
        if mode == "local" and path is None:
            raise ValueError("QdrantBackend(mode='local') requires path=")
        if mode == "http" and url is None:
            raise ValueError("QdrantBackend(mode='http') requires url=")

        self._mode = mode
        self._dim = int(dim)
        self._path = Path(path) if path is not None else None
        self._url = url
        self._api_key = api_key
        self._collection = collection or "soma_default"
        self._client: Any = None
        self._point_counter = 0
        # node_id <-> int64 point_id mapping. Qdrant accepts string
        # ids but numeric ones are cheaper to index and keep the
        # bundle on-disk format tighter.
        self._id_to_point: dict[str, int] = {}
        self._point_to_id: dict[int, str] = {}
        self._opened = False
        self._recreate = recreate
        self._warned_local_cap = False
        self.open()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return
        from qdrant_client import QdrantClient

        if self._mode == "memory":
            self._client = QdrantClient(":memory:")
        elif self._mode == "local":
            assert self._path is not None
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._path))
        else:
            self._client = QdrantClient(url=self._url, api_key=self._api_key)

        self._ensure_collection()
        # Rehydrate the id ↔ point_id map from whatever the collection
        # already holds. Necessary when reopening a local-mode path
        # after a previous session or when restoring from a snapshot.
        self._rehydrate_id_map()
        self._opened = True

    def close(self) -> None:
        import contextlib

        if self._client is not None:
            # Best-effort teardown — don't fail close() because of a
            # client that's already been torn down.
            with contextlib.suppress(Exception):
                self._client.close()
        self._client = None
        self._opened = False

    def clear(self) -> None:
        from qdrant_client.http import models as qm

        assert self._client is not None
        # Recreate is the cheapest reliable "drop all rows" path.
        self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(
                size=self._dim, distance=qm.Distance.COSINE
            ),
        )
        self._id_to_point.clear()
        self._point_to_id.clear()
        self._point_counter = 0

    # ------------------------------------------------------------------
    # Vector ops
    # ------------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        return len(self._id_to_point)

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        from qdrant_client.http import models as qm

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(ids) or arr.shape[1] != self._dim:
            raise ValueError(
                f"vectors shape {arr.shape} mismatched; expected ({len(ids)}, {self._dim})"
            )
        points: list[Any] = []
        for nid, vec in zip(ids, arr, strict=True):
            if nid in self._id_to_point:
                pid = self._id_to_point[nid]
            else:
                pid = self._point_counter
                self._point_counter += 1
                self._id_to_point[nid] = pid
                self._point_to_id[pid] = nid
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload={"node_id": nid},
                )
            )
        assert self._client is not None
        self._client.upsert(
            collection_name=self._collection, points=points, wait=True
        )
        self._maybe_warn_local_cap()

    def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        from qdrant_client.http import models as qm

        pids: list[int] = []
        for nid in ids:
            pid = self._id_to_point.pop(nid, None)
            if pid is None:
                continue
            self._point_to_id.pop(pid, None)
            pids.append(pid)
        if not pids:
            return
        assert self._client is not None
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.PointIdsList(points=pids),
            wait=True,
        )

    def get_vectors(self, ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty((0, self._dim), dtype=np.float32)
        pids: list[int] = []
        for nid in ids:
            if nid not in self._id_to_point:
                raise KeyError(f"id {nid!r} not in backend")
            pids.append(self._id_to_point[nid])
        assert self._client is not None
        records = self._client.retrieve(
            collection_name=self._collection,
            ids=pids,
            with_vectors=True,
            with_payload=False,
        )
        by_pid: dict[int, Any] = {r.id: r.vector for r in records}
        rows = np.zeros((len(ids), self._dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            vec = by_pid.get(self._id_to_point[nid])
            if vec is None:
                raise KeyError(f"id {nid!r} missing from qdrant retrieve")
            rows[i] = np.asarray(vec, dtype=np.float32)
        return rows

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude_ids: set[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if self.ntotal == 0 or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1).tolist()
        query_filter = None
        if where is not None:
            # Translator raises FilterPushdownUnsupported for anything
            # we can't express; MemoryLayer catches + falls back.
            query_filter = to_qdrant_filter(where)
        # Qdrant handles exclude_ids via must_not filter.
        if exclude_ids:
            exclude_pids = [
                self._id_to_point[nid]
                for nid in exclude_ids
                if nid in self._id_to_point
            ]
            if exclude_pids:
                query_filter = _append_exclude(query_filter, exclude_pids)

        assert self._client is not None
        # Over-fetch a few when we know we'll drop excludes client-side
        # as a defense against Qdrant ranking quirks; otherwise 1-to-1.
        limit = k
        response = self._client.search(
            collection_name=self._collection,
            query_vector=q,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        out: list[tuple[str, float]] = []
        for hit in response:
            pid = hit.id
            if isinstance(pid, str) and pid not in self._point_to_id:
                # Qdrant may return UUID-style ids for point-string
                # ingest paths; fall back to payload lookup.
                nid = (hit.payload or {}).get("node_id")
                if nid is None:
                    continue
            else:
                nid = self._point_to_id.get(pid)
                if nid is None:
                    nid = (hit.payload or {}).get("node_id")
                if nid is None:
                    continue
            out.append((nid, float(hit.score)))
        return out[:k]

    def search_subset(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        if not candidate_ids or k <= 0:
            return []
        from qdrant_client.http import models as qm

        pids = [
            self._id_to_point[nid]
            for nid in candidate_ids
            if nid in self._id_to_point
        ]
        if not pids:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1).tolist()
        assert self._client is not None
        response = self._client.search(
            collection_name=self._collection,
            query_vector=q,
            limit=min(k, len(pids)),
            query_filter=qm.Filter(
                must=[
                    qm.HasIdCondition(has_id=pids),
                ]
            ),
            with_payload=True,
        )
        out: list[tuple[str, float]] = []
        for hit in response:
            nid = self._point_to_id.get(
                hit.id, (hit.payload or {}).get("node_id")
            )
            if nid is None:
                continue
            out.append((nid, float(hit.score)))
        return out

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self, bundle_dir: Path) -> None:
        """Persist backend-specific state so :meth:`restore` can come
        back up pointing at the right data.

        - HTTP mode: writes ``backend.json`` with the URL + collection
          so restore reconnects without re-ingesting.
        - Local/memory mode: not implemented as a full client
          snapshot yet; writes a sidecar describing the bundle so a
          follow-up restore can reuse the ``path`` directory.
        """
        bundle_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": "qdrant",
            "mode": self._mode,
            "url": self._url,
            "collection": self._collection,
            "dim": self._dim,
            "path": str(self._path) if self._path is not None else None,
            "qdrant_version": _qdrant_version(),
        }
        (bundle_dir / "backend.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        if self._mode == "local" and self._path is not None:
            # Copy the collection's on-disk dir into the bundle so an
            # operator can restore on another machine just by unzipping
            # the bundle. close() is required because Qdrant holds file
            # locks on the directory.
            dest = bundle_dir / "qdrant.local"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(self._path, dest)

    def restore(self, bundle_dir: Path) -> None:
        meta_path = bundle_dir / "backend.json"
        if not meta_path.exists():
            # No backend sidecar; nothing to do — MemoryLayer will
            # rebuild via WAL replay.
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("backend") != "qdrant":
            return
        self._mode = meta.get("mode", self._mode)
        self._url = meta.get("url", self._url)
        self._collection = meta.get("collection", self._collection)
        local_copy = bundle_dir / "qdrant.local"
        if self._mode == "local" and local_copy.exists() and self._path is not None:
            if self._path.exists():
                shutil.rmtree(self._path)
            shutil.copytree(local_copy, self._path)
        self.close()
        self.open()
        # open() rehydrates the id map; nothing further needed.

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        from qdrant_client.http import models as qm

        assert self._client is not None
        existing = [c.name for c in self._client.get_collections().collections]
        if self._recreate and self._collection in existing:
            self._client.delete_collection(self._collection)
            existing.remove(self._collection)
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._dim, distance=qm.Distance.COSINE
                ),
            )

    def _rehydrate_id_map(self) -> None:
        """Rebuild ``_id_to_point`` / ``_point_to_id`` from whatever
        the Qdrant collection currently holds.

        Called on :meth:`open` so reopening a local path or HTTP
        collection automatically knows which ids exist. No-op when the
        collection is empty.
        """
        assert self._client is not None
        self._id_to_point.clear()
        self._point_to_id.clear()
        self._point_counter = 0
        offset: Any = None
        while True:
            page, offset = self._client.scroll(
                collection_name=self._collection,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                limit=1024,
            )
            for rec in page:
                pid = rec.id
                nid = (rec.payload or {}).get("node_id")
                if nid is None:
                    continue
                if not isinstance(pid, int):
                    continue
                self._id_to_point[nid] = pid
                self._point_to_id[pid] = nid
                if pid + 1 > self._point_counter:
                    self._point_counter = pid + 1
            if offset is None:
                break

    def _maybe_warn_local_cap(self) -> None:
        if self._mode != "local" or self._warned_local_cap:
            return
        if self.ntotal > LOCAL_MODE_ENTRY_WARNING_THRESHOLD:
            warnings.warn(
                f"QdrantBackend(mode='local') now holds {self.ntotal} "
                "vectors — more than 20 000. Local-file mode is "
                "documented as a <= 20K path; migrate to HTTP mode "
                "before retrieve latency dominates the budget.",
                UserWarning,
                stacklevel=3,
            )
            self._warned_local_cap = True


def _qdrant_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("qdrant-client"))
    except Exception:
        return "unknown"


def _append_exclude(existing: Any, exclude_pids: list[int]) -> Any:
    """Fold an ``exclude_ids`` set into an existing ``qm.Filter``
    (or create one) by adding a ``HasIdCondition`` into ``must_not``.
    """
    from qdrant_client.http import models as qm

    if existing is None:
        return qm.Filter(
            must_not=[qm.HasIdCondition(has_id=exclude_pids)],
        )
    must = list(getattr(existing, "must", None) or [])
    must_not = list(getattr(existing, "must_not", None) or [])
    must_not.append(qm.HasIdCondition(has_id=exclude_pids))
    return qm.Filter(must=must or None, must_not=must_not)


def _uuid_from_nid(nid: str) -> str:
    """Deterministic UUID from an opaque node_id string.

    Unused in v1 (we use an integer counter), but kept here for the
    HTTP-mode multi-process case where the counter isn't coherent
    across workers. Phase 7 can swap in.
    """
    return str(uuid.UUID(nid)) if len(nid) == 32 else str(uuid.uuid5(uuid.NAMESPACE_OID, nid))


__all__ = ["QdrantBackend", "LOCAL_MODE_ENTRY_WARNING_THRESHOLD"]
