"""ChromaBackend — Chroma-backed :class:`VectorBackend` adapter.

The strategic slot for Chroma in SOMA's backend lineup is **migration
ergonomics**: a large fraction of existing agent deployments already
keep their vectors in a Chroma collection, and the switching cost
from "export + reimport" to "point ``MemoryLayer`` at the same Chroma
store" is just this adapter. Chroma joins InProcFlat, Qdrant (local +
HTTP), and LanceDB as the fourth pluggable vector backend.

Features:

- Cosine distance (the only metric MemoryLayer advertises today).
  Collection is created with ``metadata={"hnsw:space": "cosine"}``.
- Opaque SOMA ``node_id`` strings passed straight through as Chroma
  document ids — no internal remapping.
- ``supports_filter_pushdown=True``. The translator
  (:mod:`soma.memory.backends.chroma_filter`) turns SOMA's flat
  ``where`` dict into Chroma's native ``where`` dialect. Unsupported
  ops raise :class:`FilterPushdownUnsupported`; MemoryLayer falls
  back to its Python pre-filter + ``search_subset`` path.
- Snapshot = copy Chroma's persist directory into the bundle; restore
  = inverse. Chroma 0.5+ auto-persists (``client.persist()`` was
  removed in 0.5 in favour of continuous flushes), so the directory
  copy is the canonical snapshot.

Pinned version floor: ``chromadb>=0.5``. Earlier versions had a
different ``delete_collection`` + recreate story (we rely on the
post-0.5 idempotent ``get_or_create_collection`` path) and still
exposed a now-removed ``persist()`` method that older adapter code
sometimes called. Requiring 0.5 lets us skip that branch.

chromadb is an optional dep — the import is guarded so SOMA still
ships without it. Constructing a ``ChromaBackend`` when the dep is
missing raises :class:`ImportError` with a pip install hint.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from soma.memory.backend import FilterPushdownUnsupported, _default_search_near_id
from soma.memory.backends.chroma_filter import to_chroma_where

try:
    import chromadb

    _HAS_CHROMA = True
except ImportError:  # pragma: no cover - exercised via monkeypatch
    chromadb = None  # type: ignore[assignment]
    _HAS_CHROMA = False


class ChromaBackend:
    """Chroma-backed :class:`VectorBackend`.

    Parameters
    ----------
    path:
        Directory root for :class:`chromadb.PersistentClient`. Required
        unless ``client`` is provided. Chroma creates the directory on
        first open and auto-persists from that point on (0.5+ dropped
        the explicit ``persist()`` method in favour of continuous
        flushes, so no "save" step is needed).
    client:
        Optional pre-built chromadb client (``PersistentClient`` /
        ``EphemeralClient`` / ``HttpClient``). Escape hatch for
        non-default configs — custom auth, tenancy/database, or
        remote HTTP mode. When supplied, ``path`` is ignored.
    collection_name:
        Which Chroma collection to open or create. Defaults to
        ``"soma"``; MemoryLayer passes the bundle name for multi-
        tenant layouts. Chroma enforces 3-63 chars, alphanumeric
        plus ``_`` and ``-``.
    dim:
        Embedding dimension. Optional — Chroma doesn't enforce a
        schema-level dim, it just stores whatever you add. Supplied
        here so :attr:`dim` can answer without probing the
        collection. When omitted, we fall back to introspecting the
        first stored row.
    """

    supports_filter_pushdown: bool = True
    name: str = "chroma"

    def __init__(
        self,
        *,
        path: str | None = None,
        client: Any = None,
        collection_name: str = "soma",
        dim: int | None = None,
    ) -> None:
        if not _HAS_CHROMA:
            raise ImportError(
                "ChromaBackend requires chromadb. Install with: pip install 'soma[chroma]'"
            )
        if client is None and path is None:
            raise ValueError("ChromaBackend requires either path= or client=")
        self._external_client = client is not None
        self._path = path
        self._collection_name = collection_name
        self._dim: int | None = int(dim) if dim is not None else None
        self._client: Any = client
        self._coll: Any = None
        # Whether any ``add`` call has supplied metadata. Drives
        # whether we attempt filter pushdown on ``where``: if no
        # metadata was ever stored, a pushdown against a metadata
        # field is guaranteed to match nothing (which would silently
        # return zero rows), so we raise ``FilterPushdownUnsupported``
        # instead and let MemoryLayer's Python pre-filter path
        # inspect its own metadata index.
        self._has_metadata: bool = False
        self.open()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Idempotent — attach client + collection."""
        if self._coll is not None:
            return
        if self._client is None:
            assert self._path is not None
            Path(self._path).mkdir(parents=True, exist_ok=True)
            assert chromadb is not None
            self._client = chromadb.PersistentClient(path=self._path)
        # Cosine is the only metric MemoryLayer advertises; encode it
        # in the collection metadata so queries use cosine distance
        # without caller opt-in.
        self._coll = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Refresh the metadata-presence flag from any already-stored
        # rows so reopens of a pre-populated collection know whether
        # to attempt pushdown.
        try:
            probe = self._coll.get(limit=1, include=["metadatas"])
            metas = probe.get("metadatas") or []
            if any(m for m in metas):
                self._has_metadata = True
        except Exception:
            # Best-effort — a probe failure shouldn't block open().
            pass

    def close(self) -> None:
        """Release the collection + client references.

        Chroma caches the ``System`` (which owns sqlite + segment
        handles) in a process-wide ``SharedSystemClient`` singleton,
        so dropping our ``PersistentClient`` reference alone is not
        enough on Windows — the sqlite file stays locked. We also
        call ``SharedSystemClient.clear_system_cache()`` when we own
        the client (not a caller-supplied one) so tests / snapshot
        restores that ``rmtree`` the persist directory don't fail
        with a file-in-use error.
        """
        self._coll = None
        if not self._external_client:
            self._client = None
            import gc

            gc.collect()
            _clear_chromadb_system_cache()
            gc.collect()

    def clear(self) -> None:
        """Drop every vector but keep the collection usable.

        We go through ``delete_collection`` + ``get_or_create_collection``
        rather than deleting all ids one at a time — the former is O(1)
        in Chroma's internal index vs O(N) for the explicit-ids path.
        """
        assert self._client is not None
        # ``delete_collection`` raises when the collection is already
        # absent; suppress so ``clear`` stays idempotent.
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._collection_name)
        self._coll = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._has_metadata = False

    # ------------------------------------------------------------------
    # Vector ops
    # ------------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        if self._coll is None:
            return 0
        return int(self._coll.count())

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        # Fallback: introspect the first stored row's embedding.
        if self._coll is None or self._coll.count() == 0:
            return 0
        probe = self._coll.get(limit=1, include=["embeddings"])
        embs = probe.get("embeddings")
        if embs is None or len(embs) == 0:
            return 0
        self._dim = int(np.asarray(embs[0]).shape[0])
        return self._dim

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        *,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert ``len(ids)`` vectors.

        ``metadatas`` is a Chroma-specific extension beyond the
        ``VectorBackend`` Protocol: when supplied, each row is stored
        with metadata so downstream ``where`` filters can push down.
        MemoryLayer's default path never calls with ``metadatas=`` —
        metadata lives on MemoryLayer's Python side — but external
        callers who construct a ``ChromaBackend`` directly (or tests
        that exercise pushdown) can use this path to populate
        filterable fields.
        """
        if not ids:
            return
        assert self._coll is not None
        arr = np.asarray(vectors, dtype=np.float32)
        expected_dim = self._dim if self._dim is not None else arr.shape[-1]
        if arr.ndim != 2 or arr.shape[0] != len(ids) or arr.shape[1] != expected_dim:
            raise ValueError(
                f"vectors shape {arr.shape} mismatched; expected ({len(ids)}, {expected_dim})"
            )
        if self._dim is None:
            self._dim = int(arr.shape[1])
        embeddings = arr.tolist()
        if metadatas is not None:
            if len(metadatas) != len(ids):
                raise ValueError(f"metadatas length {len(metadatas)} != ids length {len(ids)}")
            # Chroma rejects empty metadata dicts on add; replace with
            # None cell-by-cell so the collection accepts the batch.
            cleaned: list[dict[str, Any] | None] = [(m if m else None) for m in metadatas]
            if any(m is not None for m in cleaned):
                self._has_metadata = True
            # Chroma 0.5 still requires every row have a metadata dict
            # when metadatas= is supplied; fall back to a placeholder
            # for empty-dict rows so the batch stays uniform.
            final_metas = [m if m is not None else {"_": 0} for m in cleaned]
            self._coll.upsert(
                ids=list(ids),
                embeddings=embeddings,
                metadatas=final_metas,
            )
        else:
            self._coll.upsert(ids=list(ids), embeddings=embeddings)

    def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        assert self._coll is not None
        # Chroma logs a warning for unknown ids but doesn't raise —
        # matches the Protocol's "missing ids are ignored" contract.
        self._coll.delete(ids=list(ids))

    def get_vectors(self, ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty((0, self.dim), dtype=np.float32)
        assert self._coll is not None
        result = self._coll.get(ids=list(ids), include=["embeddings"])
        got_ids = list(result.get("ids") or [])
        got_embs_raw = result.get("embeddings")
        got_embs = list(got_embs_raw) if got_embs_raw is not None else []
        # Chroma returns rows in its own internal order (sorted by
        # id), not in the order we asked for. Build a lookup and
        # reindex so the caller gets the requested order.
        id_to_vec: dict[str, np.ndarray] = {}
        for rid, emb in zip(got_ids, got_embs, strict=True):
            id_to_vec[str(rid)] = np.asarray(emb, dtype=np.float32)
        rows = np.zeros((len(ids), self.dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            vec = id_to_vec.get(nid)
            if vec is None:
                raise KeyError(f"id {nid!r} not in backend")
            rows[i] = vec
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
        assert self._coll is not None
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if self._dim is not None and q.shape[0] != self._dim:
            raise ValueError(f"query dim mismatch: backend={self._dim}, got={q.shape[0]}")
        # Translator raises FilterPushdownUnsupported for anything
        # we can't express; MemoryLayer catches + falls back.
        translated_where = to_chroma_where(where)
        if translated_where is not None and not self._has_metadata:
            # No metadata has ever been stored — Chroma's where
            # can't match anything by definition. Raise so MemoryLayer
            # falls back to its Python pre-filter path which inspects
            # MemoryLayer-side metadata.
            raise FilterPushdownUnsupported(
                op=None,
                field=None,
                message=(
                    "Chroma collection has no stored metadata; "
                    "MemoryLayer will use its Python pre-filter path"
                ),
            )
        # Over-fetch when exclusions are active so the post-filter
        # still has k rows to hand back.
        limit = min(
            k + (len(exclude_ids) if exclude_ids else 0),
            self.ntotal,
        )
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [q.tolist()],
            "n_results": limit,
        }
        if translated_where is not None:
            query_kwargs["where"] = translated_where
        result = self._coll.query(**query_kwargs)
        ids_nested = result.get("ids") or [[]]
        dists_nested = result.get("distances") or [[]]
        ids_row = list(ids_nested[0]) if ids_nested else []
        dists_row = list(dists_nested[0]) if dists_nested else []
        out: list[tuple[str, float]] = [
            (str(nid), _distance_to_similarity(float(d)))
            for nid, d in zip(ids_row, dists_row, strict=True)
        ]
        if exclude_ids:
            out = [(nid, s) for nid, s in out if nid not in exclude_ids]
        return out[:k]

    def search_subset(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        if not candidate_ids or k <= 0:
            return []
        assert self._coll is not None
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        # Chroma's query cannot filter on id directly, so we pull the
        # candidate vectors and rank them in-process. This is exactly
        # what MemoryLayer's default path does for adapters without
        # server-side subset search, and keeps the cost proportional
        # to the subset size rather than ntotal.
        result = self._coll.get(ids=list(candidate_ids), include=["embeddings"])
        got_ids = list(result.get("ids") or [])
        got_embs_raw = result.get("embeddings")
        got_embs = list(got_embs_raw) if got_embs_raw is not None else []
        if not got_ids:
            return []
        matrix = np.asarray(got_embs, dtype=np.float32)
        # Cosine similarity against the query.
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        row_norms = np.linalg.norm(matrix, axis=1) + 1e-12
        sims = (matrix @ q_norm) / row_norms
        order = np.argsort(-sims)[: min(k, len(got_ids))]
        return [(str(got_ids[i]), float(sims[i])) for i in order]

    def search_near_id(
        self,
        node_id: str,
        k: int,
        *,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Delegate to the Protocol default.

        Chroma has no server-side "find similar to stored id" call,
        so the default two-step ``get_vectors`` + ``search`` path is
        as good as any adapter-specific path would be — both round-
        trip the pivot vector through Python.
        """
        return _default_search_near_id(self, node_id, k, exclude_self=exclude_self)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self, bundle_dir: Path) -> None:
        """Copy the Chroma persist directory into the bundle.

        Chroma 0.5+ auto-persists, so the on-disk directory IS the
        canonical snapshot. We close the collection reference before
        copying to flush any in-memory buffers, then reopen so the
        caller can keep using the backend.
        """
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": "chroma",
            "dim": self._dim,
            "collection_name": self._collection_name,
            "chromadb_version": _chromadb_version(),
        }
        (bundle_dir / "backend.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self._path is None:
            # External client (e.g. HTTP) — we can't copy a directory
            # we don't own. Sidecar-only snapshot; restore() will
            # no-op on the data side and rely on WAL replay.
            return
        dest = bundle_dir / "chroma"
        if dest.exists():
            shutil.rmtree(dest)
        src = Path(self._path)
        if src.exists():
            # Drop the collection reference so any mmap handles flush
            # before we copy.
            self._coll = None
            shutil.copytree(src, dest)
            # Reopen so the caller can keep using the backend.
            self.open()

    def restore(self, bundle_dir: Path) -> None:
        """Inverse of :meth:`snapshot`.

        Copies the bundle's ``chroma/`` subdirectory over the live
        persist directory (wiping whatever was there) and reopens the
        collection. No-op when the bundle has no ``chroma/`` subdir
        (MemoryLayer will rebuild via WAL replay in that case).
        """
        bundle_dir = Path(bundle_dir)
        meta_path = bundle_dir / "backend.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("backend") == "chroma":
                self._collection_name = meta.get("collection_name", self._collection_name)
                if meta.get("dim") is not None:
                    self._dim = int(meta["dim"])
        src = bundle_dir / "chroma"
        if not src.exists():
            return
        if self._path is None:
            # External client — can't restore a directory for a
            # client we don't own.
            return
        # Fully release the client + Chroma's process-wide system
        # cache before rmtree — on Windows the underlying sqlite +
        # segment files stay locked until the singleton is cleared,
        # which would crash ``shutil.rmtree`` below.
        import gc

        self._coll = None
        self._client = None
        gc.collect()
        _clear_chromadb_system_cache()
        gc.collect()
        dst = Path(self._path)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        # Reopen — PersistentClient picks up the restored files.
        assert chromadb is not None
        self._client = chromadb.PersistentClient(path=self._path)
        self._coll = None
        self.open()


def _distance_to_similarity(distance: float) -> float:
    """Convert a Chroma cosine distance to a cosine-similarity score.

    Chroma with ``hnsw:space=cosine`` reports ``1 - cosine_similarity``
    in ``[0, 2]``. We return ``1 - distance`` so the range and
    direction match every other VectorBackend (higher = more similar,
    ``[-1, 1]``).
    """
    return 1.0 - float(distance)


def _clear_chromadb_system_cache() -> None:
    """Tear down Chroma's process-wide singleton.

    Chroma caches the underlying ``System`` (which owns sqlite +
    segment file handles) in ``chromadb.api.client.SharedSystemClient``
    keyed by persist directory. On Windows those handles outlive the
    Python-level ``PersistentClient`` reference, so a subsequent
    ``rmtree`` of the directory fails with ``WinError 32 — file in
    use``. Clearing the cache releases the handles deterministically.

    No-op if the helper isn't available (older chromadb, or the
    library reshuffles its internals in a future release).
    """
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def _chromadb_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("chromadb"))
    except Exception:
        return "unknown"


__all__ = ["ChromaBackend"]
