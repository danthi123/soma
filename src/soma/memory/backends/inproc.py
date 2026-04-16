"""InProcBackend — the default ``VectorBackend``.

Pure-Python + numpy vector store with a lazy FAISS accelerator. Reads
exactly like the pre-Phase-6 MemoryLayer's FAISS path, just decoupled
behind the protocol. Supports ``flat`` (exact) and ``hnsw``
(approximate) FAISS index types; linear brute-force stays as the
fallback for small stores and ``search_subset`` (which needs an
arbitrary subset mask FAISS can't express).

Vectors live in a ``(ntotal, dim)`` numpy array that grows on
``add`` and shrinks on ``remove``. A parallel id list keeps the
position ↔ node_id mapping. Any mutation invalidates the FAISS index
so the next ``search`` rebuilds lazily when the store is large
enough to warrant it.

Snapshot format is ``memory_embeddings.pt`` (a torch tensor of shape
``(ntotal, dim)``) — unchanged from pre-Phase-6 bundles, so old
bundles keep loading without migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from soma import metrics as _m


def _import_faiss() -> Any:
    try:
        import faiss

        return faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS backend requires faiss-cpu or faiss-gpu. "
            "Install with: pip install faiss-cpu"
        ) from exc


class InProcBackend:
    """In-process FAISS + numpy ``VectorBackend``.

    Parameters
    ----------
    dim:
        Embedding dimension. Fixed at construction; subsequent adds
        must match.
    faiss_threshold:
        Minimum number of entries before the FAISS index is built.
        Below that, search is a plain numpy cosine scan. Set to 0 to
        disable FAISS entirely (always use the linear path).
    faiss_index_type:
        ``"flat"`` for IndexFlatIP (exact, SIMD) or ``"hnsw"`` for
        IndexHNSWFlat (approximate, much faster at large N).
    faiss_hnsw_m / faiss_hnsw_ef_search / faiss_hnsw_ef_construction:
        HNSW tuning knobs. Ignored for ``flat``.
    bundle_name:
        Metric label for ``soma_faiss_index_size``. Defaults match
        MemoryLayer's ``"__default__"``.
    """

    supports_filter_pushdown: bool = False
    name: str = "inproc"

    def __init__(
        self,
        *,
        dim: int,
        faiss_threshold: int = 10_000,
        faiss_index_type: str = "flat",
        faiss_hnsw_m: int = 32,
        faiss_hnsw_ef_search: int = 64,
        faiss_hnsw_ef_construction: int = 80,
        bundle_name: str = "__default__",
    ) -> None:
        if faiss_index_type not in ("flat", "hnsw"):
            raise ValueError(
                f"faiss_index_type must be 'flat' or 'hnsw', got {faiss_index_type!r}"
            )
        self._dim = int(dim)
        self._faiss_threshold = int(faiss_threshold)
        self._faiss_index_type = faiss_index_type
        self._faiss_hnsw_m = int(faiss_hnsw_m)
        self._faiss_hnsw_ef_search = int(faiss_hnsw_ef_search)
        self._faiss_hnsw_ef_construction = int(faiss_hnsw_ef_construction)
        self._bundle_name = bundle_name
        # Parallel storage: ids in insertion order, vectors as one
        # numpy array that grows by concat on add and shrinks by a
        # row delete on remove.
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._vectors: np.ndarray = np.empty((0, self._dim), dtype=np.float32)
        self._faiss_index: Any = None

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return self._dim

    def open(self) -> None:
        """No-op for in-proc; kept so the protocol is uniform."""
        return None

    def close(self) -> None:
        """No-op for in-proc; kept so the protocol is uniform."""
        return None

    def clear(self) -> None:
        self._ids = []
        self._id_to_idx = {}
        self._vectors = np.empty((0, self._dim), dtype=np.float32)
        self._faiss_index = None
        _m.FAISS_INDEX_SIZE.labels(bundle=self._bundle_name).set(0)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(ids):
            raise ValueError(
                f"vectors must have shape (len(ids), dim); got {arr.shape} for {len(ids)} ids"
            )
        if arr.shape[1] != self._dim:
            raise ValueError(
                f"vector dim mismatch: backend={self._dim}, got={arr.shape[1]}"
            )
        start = len(self._ids)
        for offset, nid in enumerate(ids):
            self._id_to_idx[nid] = start + offset
            self._ids.append(nid)
        if self._vectors.size == 0:
            self._vectors = arr.copy()
        else:
            self._vectors = np.concatenate([self._vectors, arr], axis=0)
        self._faiss_index = None

    def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        to_drop_idx = sorted(
            {self._id_to_idx[nid] for nid in ids if nid in self._id_to_idx},
            reverse=True,
        )
        if not to_drop_idx:
            return
        # Drop rows in descending index order so earlier indices stay
        # valid during the iteration. Then rebuild the id list and
        # index map from scratch — O(N) but forgets are rare.
        mask = np.ones(self._vectors.shape[0], dtype=bool)
        for i in to_drop_idx:
            mask[i] = False
            self._ids.pop(i)
        self._vectors = self._vectors[mask]
        self._id_to_idx = {nid: i for i, nid in enumerate(self._ids)}
        self._faiss_index = None

    def get_vectors(self, ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty((0, self._dim), dtype=np.float32)
        rows = np.empty((len(ids), self._dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            idx = self._id_to_idx.get(nid)
            if idx is None:
                raise KeyError(f"id {nid!r} not in backend")
            rows[i] = self._vectors[idx]
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
        # InProcBackend doesn't push down ``where``; MemoryLayer falls
        # back to its Python pre-filter path when where is not None.
        # Accepting the kwarg here keeps the signature uniform for
        # tests and future adapters.
        _ = where  # intentional: not supported, caller uses search_subset.
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(
                f"query dim mismatch: backend={self._dim}, got={q.shape[1]}"
            )
        exclude = exclude_ids or set()
        self._maybe_build_faiss()
        if self._faiss_index is not None and not exclude:
            return self._rank_faiss(q, k=k)
        return self._rank_linear(q, k=k, exclude=exclude)

    def search_subset(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        if not candidate_ids or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(
                f"query dim mismatch: backend={self._dim}, got={q.shape[0]}"
            )
        idxs: list[int] = []
        seen: set[str] = set()
        for nid in candidate_ids:
            if nid in seen:
                continue
            pos = self._id_to_idx.get(nid)
            if pos is not None:
                idxs.append(pos)
                seen.add(nid)
        if not idxs:
            return []
        subset = self._vectors[idxs]
        scores = _cosine_sim_rows(q, subset)
        order = np.argsort(-scores)[:k]
        return [(self._ids[idxs[j]], float(scores[j])) for j in order]

    def snapshot(self, bundle_dir: Path) -> None:
        """Write ``memory_embeddings.pt`` with the stacked vector matrix.

        Bundle format is unchanged from pre-Phase-6: a single torch
        tensor of shape ``(ntotal, dim)``. MemoryLayer's own save
        writes ``memory_index.json`` alongside which holds the id
        list; on standalone restore (no MemoryLayer driving it) we
        also write ``inproc_ids.json`` so the backend can round-trip
        itself in isolation.
        """
        import json

        bundle_dir.mkdir(parents=True, exist_ok=True)
        tensor = torch.from_numpy(self._vectors.copy())
        torch.save(tensor, str(bundle_dir / "memory_embeddings.pt"))
        (bundle_dir / "inproc_ids.json").write_text(
            json.dumps(self._ids), encoding="utf-8"
        )

    def restore(self, bundle_dir: Path) -> None:
        """Inverse of :meth:`snapshot` for a standalone backend.

        MemoryLayer's bundle loader does not take this path — it
        repopulates the backend via explicit :meth:`add` calls from
        ``memory_index.json``. This method exists so the backend can
        be tested and persisted on its own (benchmarks, adapter
        matrix, ...).
        """
        import json

        emb_path = bundle_dir / "memory_embeddings.pt"
        if not emb_path.exists():
            return
        tensor = torch.load(
            str(emb_path), map_location="cpu", weights_only=True
        )
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        ids_path = bundle_dir / "inproc_ids.json"
        if ids_path.exists():
            self._ids = list(json.loads(ids_path.read_text(encoding="utf-8")))
        else:
            # Fallback for legacy bundles: synthesize ids from row index.
            self._ids = [f"__row_{i}__" for i in range(arr.shape[0])]
        self._id_to_idx = {nid: i for i, nid in enumerate(self._ids)}
        self._vectors = arr
        self._faiss_index = None

    # ------------------------------------------------------------------
    # FAISS / linear ranking
    # ------------------------------------------------------------------
    def _maybe_build_faiss(self) -> None:
        if self._faiss_threshold <= 0:
            self._faiss_index = None
            _m.FAISS_INDEX_SIZE.labels(bundle=self._bundle_name).set(0)
            return
        if self.ntotal < self._faiss_threshold:
            self._faiss_index = None
            _m.FAISS_INDEX_SIZE.labels(bundle=self._bundle_name).set(0)
            return
        if self._faiss_index is not None:
            return
        self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        faiss = _import_faiss()
        with _m.FAISS_REBUILD_SECONDS.time():
            matrix = self._vectors.copy()
            faiss.normalize_L2(matrix)
            if self._faiss_index_type == "hnsw":
                index = faiss.IndexHNSWFlat(
                    self._dim,
                    self._faiss_hnsw_m,
                    faiss.METRIC_INNER_PRODUCT,
                )
                index.hnsw.efConstruction = self._faiss_hnsw_ef_construction
                index.hnsw.efSearch = self._faiss_hnsw_ef_search
            else:
                index = faiss.IndexFlatIP(self._dim)
            index.add(matrix)
            self._faiss_index = index
        _m.FAISS_REBUILD_TOTAL.labels(
            index_type=self._faiss_index_type
        ).inc()
        _m.FAISS_INDEX_SIZE.labels(bundle=self._bundle_name).set(
            index.ntotal
        )

    def _rank_faiss(
        self, query: np.ndarray, *, k: int
    ) -> list[tuple[str, float]]:
        assert self._faiss_index is not None
        faiss = _import_faiss()
        q = query.copy().astype(np.float32)
        faiss.normalize_L2(q)
        actual_k = min(k, self._faiss_index.ntotal)
        if actual_k <= 0:
            return []
        scores, indices = self._faiss_index.search(q, actual_k)
        return [
            (self._ids[int(idx)], float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
            if idx >= 0
        ]

    def _rank_linear(
        self, query: np.ndarray, *, k: int, exclude: set[str]
    ) -> list[tuple[str, float]]:
        q = query.reshape(-1)
        scores = _cosine_sim_rows(q, self._vectors)
        if exclude:
            for nid in exclude:
                idx = self._id_to_idx.get(nid)
                if idx is not None:
                    scores[idx] = -np.inf
        eligible = int((scores > -np.inf).sum())
        actual_k = min(k, eligible)
        if actual_k <= 0:
            return []
        order = np.argsort(-scores)[:actual_k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]


def _cosine_sim_rows(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query and each row of matrix.

    Returns ``(matrix.shape[0],)`` float32. Uses numpy so we don't have
    to shuttle data through torch for a pure CPU-side scan.
    """
    q = query.reshape(-1)
    q_norm = np.linalg.norm(q) + 1e-12
    m_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    dots = matrix @ q
    return (dots / (m_norms * q_norm)).astype(np.float32)


__all__ = ["InProcBackend"]
