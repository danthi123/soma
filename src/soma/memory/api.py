"""Public memory-layer API: vector-DB-shaped surface over SOMA's substrate.

This is the product-facing entry point introduced by the 2026-04-15
pivot (see ``docs/plans/2026-04-15-memory-layer-pivot.md``). It gives
agent developers a familiar ``store``/``retrieve`` contract while
leaving room for the graph/plasticity differentiators to come online in
later stages.

Stage 2 (this module) ships the flat vector-store semantics plus
save/load. Stage 3 wires ``consolidate()`` into SOMA's growth engine so
stored entries become graph structure that prunes and reinforces with
use; today ``consolidate`` is a safe no-op so callers can include the
call in their loops now and not have to revisit when the plasticity
path lands.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import portalocker
import torch
from torch.nn import functional as F  # noqa: N812

from soma import metrics as _m
from soma.io.text_encoder import TextEncoder, load_tokenizer
from soma.memory.backend import FilterPushdownUnsupported, VectorBackend
from soma.memory.wal import WAL, WalRecord

__all__ = ["MemoryHit", "MemoryLayer"]

logger = logging.getLogger("soma.memory")

_VALID_DURABILITY = {"sync", "batch", "async"}

# Compaction defaults. When the WAL grows past ``max(size_floor,
# size_ratio * snapshot_size) OR record_count > record_threshold OR
# age > age_seconds``, a background compaction thread rewrites the
# snapshot and truncates the WAL.
_COMPACTION_SIZE_FLOOR = 4 * 1024 * 1024  # 4 MB
_COMPACTION_SIZE_RATIO = 1.0
_COMPACTION_RECORD_THRESHOLD = 10_000
_COMPACTION_AGE_SECONDS = 60 * 60  # 1 hour


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    Writes to ``<path>.tmp``, fsyncs the fd, then ``os.replace`` swaps it
    into place. On Unix we also fsync the parent directory so the rename
    itself is durable; on Windows that's a no-op (``os.open`` on a
    directory raises). Removes the ``.tmp`` file on any error so callers
    see a clean bundle dir when save() partially fails.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            # fd already handed off to fdopen; nothing else to close.
            raise
        os.replace(str(tmp), str(path))
        # Best-effort parent-dir fsync on POSIX; Windows doesn't let us
        # open a directory, so we just skip it there.
        if os.name != "nt":
            with contextlib.suppress(OSError):
                dfd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """torch.save wrapper that goes through ``<path>.tmp`` + os.replace.

    ``torch.save`` writes the file directly in place, so a crash mid-write
    leaves a half-file. We route it through a sibling ``.tmp`` and flip
    atomically. Errors delete the ``.tmp`` leftover.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, str(tmp))
        # Fsync the tmp before the rename so the bytes are on stable
        # storage before anyone can see the file at its final name.
        with contextlib.suppress(OSError):
            fd = os.open(str(tmp), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.replace(str(tmp), str(path))
        if os.name != "nt":
            with contextlib.suppress(OSError):
                dfd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise

EmbedFn = Callable[[str], torch.Tensor]


def _vec_to_np(v: torch.Tensor) -> np.ndarray[Any, Any]:
    """Convert a (dim,) torch tensor into a (1, dim) float32 ndarray.

    Protocol-boundary shim — every backend expects numpy float32, not
    torch tensors. Centralized here so the torch→numpy round-trip is
    consistent on every call site.
    """
    arr: np.ndarray[Any, Any] = (
        v.detach().cpu().numpy().astype(np.float32).reshape(1, -1)
    )
    return arr


def _batch_to_np(vs: list[torch.Tensor]) -> np.ndarray[Any, Any]:
    """Stack a list of (dim,) tensors into a (N, dim) float32 array."""
    if not vs:
        return np.empty((0, 0), dtype=np.float32)
    arr: np.ndarray[Any, Any] = (
        torch.stack(vs, dim=0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return arr


# ----------------------------------------------------------------------
# Metadata filter — Chroma-compatible subset of `where` semantics.
# ----------------------------------------------------------------------
_COMPARE_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda a, b: bool(a == b),
    "$ne": lambda a, b: bool(a != b),
    "$gt": lambda a, b: bool(a > b),
    "$gte": lambda a, b: bool(a >= b),
    "$lt": lambda a, b: bool(a < b),
    "$lte": lambda a, b: bool(a <= b),
}


def _matches_where(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    """Chroma-compatible subset of ``where`` filters:

    - ``{"f": v}`` exact match
    - ``{"f": {"$eq": v}}`` / ``$ne`` / ``$gt`` / ``$gte`` / ``$lt`` / ``$lte``
    - ``{"f": {"$in": [...]}}`` value-in-list

    Multiple fields = AND (same as Chroma's default). Missing fields
    on an entry fail the filter.
    """
    for field_name, spec in where.items():
        if isinstance(spec, dict):
            actual = meta.get(field_name)
            for op, expected in spec.items():
                if op == "$in":
                    if not isinstance(expected, (list, tuple, set)):
                        raise ValueError(f"$in expects a list, got {type(expected).__name__}")
                    if actual not in expected:
                        return False
                elif op == "$nin":
                    if not isinstance(expected, (list, tuple, set)):
                        raise ValueError(f"$nin expects a list, got {type(expected).__name__}")
                    if actual in expected:
                        return False
                elif op in _COMPARE_OPS:
                    # Special case: comparing explicitly against None / missing.
                    # {"f": {"$eq": None}} matches entries missing the field OR
                    # entries that have field=None (both are "no value here").
                    # {"f": {"$ne": None}} matches entries with a concrete value.
                    if expected is None:
                        if op == "$eq" and actual is not None:
                            return False
                        if op == "$ne" and actual is None:
                            return False
                        continue
                    fn = _COMPARE_OPS[op]
                    if actual is None:
                        return False
                    try:
                        if not fn(actual, expected):
                            return False
                    except TypeError:
                        return False
                else:
                    raise ValueError(f"unsupported operator {op!r} in where clause")
        else:
            if meta.get(field_name) != spec:
                return False
    return True


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved entry. Immutable so callers can pass them around safely."""

    node_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_step: int = 0


class MemoryLayer:
    """Local-first, learning agent-memory layer.

    Reads like a vector DB at the edge: ``store(text)`` appends,
    ``retrieve(query)`` ranks by cosine similarity. Behind the API,
    entries are kept alongside their pooled-token-embedding vectors in
    an in-memory tensor that scales linearly with the store size (fine
    for up to ~100K entries on consumer hardware; Stage 3 adds a
    chunked / on-disk path for larger stores).

    The embedder is caller-supplied — pass any
    :class:`soma.io.text_encoder.TextEncoder` plus its tokenizer. This
    keeps the memory layer independent of any specific embedding model,
    so callers can swap in sentence-transformers or an LLM's input
    embeddings once Stage 3's benchmark harness picks a default.

    Persistence writes a directory bundle compatible with
    ``SOMA.save_bundle`` naming: ``tokenizer.json``, ``encoder.pt``,
    ``memory_index.json``, ``memory_embeddings.pt``. A future
    MemoryLayer bundle CAN be dropped inside a SOMA brain bundle and
    the two will coexist without collision.
    """

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        encoder: TextEncoder | None = None,
        embed_fn: EmbedFn | None = None,
        embed_dim: int | None = None,
        device: torch.device | str | None = None,
        faiss_threshold: int = 10_000,
        faiss_index_type: str = "flat",
        faiss_hnsw_m: int = 32,
        faiss_hnsw_ef_search: int = 64,
        faiss_hnsw_ef_construction: int = 80,
        auto_consolidate_every: int = 0,
        graph_rerank_alpha: float = 0.0,
        graph_rerank_stable_capture: bool = True,
        bundle_path: str | Path | None = None,
        durability: Literal["sync", "batch", "async"] = "sync",
        backend: VectorBackend | None = None,
    ) -> None:
        if embed_fn is None and encoder is None:
            raise ValueError("MemoryLayer needs either (tokenizer + encoder) or embed_fn")
        if durability not in _VALID_DURABILITY:
            raise ValueError(
                f"durability must be one of {sorted(_VALID_DURABILITY)}, got {durability!r}"
            )
        self._tokenizer = tokenizer
        self._encoder = encoder
        self._custom_embed_fn = embed_fn
        if device is not None:
            self._device = torch.device(device)
        elif encoder is not None:
            self._device = encoder.embedding.weight.device
        else:
            self._device = torch.device("cpu")
        if embed_fn is not None:
            if embed_dim is None:
                raise ValueError("embed_dim required when using embed_fn")
            self._embed_dim: int = embed_dim
        else:
            assert encoder is not None
            self._embed_dim = int(encoder.embed_dim)

        # Optional SOMA attachment for graph-based consolidation.
        self._soma: Any = None
        self._soma_tokenizer: Any = None
        self._soma_encoder: Any = None

        # FAISS ANN params are propagated into the default InProcBackend.
        # Kept as kwargs on MemoryLayer so the old call sites (and the
        # bundle-save path, which reads them into backend.json) don't
        # have to be rewritten — the defaults match the pre-Phase-6
        # behavior.
        if faiss_index_type not in ("flat", "hnsw"):
            raise ValueError(f"faiss_index_type must be 'flat' or 'hnsw', got {faiss_index_type!r}")
        self._faiss_threshold: int = faiss_threshold
        self._faiss_index_type: str = faiss_index_type
        self._faiss_hnsw_m: int = int(faiss_hnsw_m)
        self._faiss_hnsw_ef_search: int = int(faiss_hnsw_ef_search)
        self._faiss_hnsw_ef_construction: int = int(faiss_hnsw_ef_construction)
        self._auto_consolidate_every: int = auto_consolidate_every
        self._stores_since_consolidation: int = 0

        # Parallel storage. Order is preserved across save/load so
        # ``get_recent`` stays stable. The vector matrix lives in the
        # backend — MemoryLayer only keeps the id/text/metadata side.
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._timestamps: list[int] = []
        # SOMA output activations captured during consolidate(), keyed by
        # node_id. Used for graph-aware re-ranking when SOMA is attached.
        # Keyed by id (not list position) so backends that soft-delete or
        # reorder can't desync us — see Phase 6 task 3.
        self._soma_activations: dict[str, torch.Tensor | None] = {}

        # VectorBackend adapter. Default = in-process FAISS for zero
        # behavior change; callers that opt in to Qdrant (or any other
        # adapter) pass backend= explicitly.
        if backend is None:
            from soma.memory.backends.inproc import InProcBackend

            backend = InProcBackend(
                dim=self._embed_dim,
                faiss_threshold=self._faiss_threshold,
                faiss_index_type=self._faiss_index_type,
                faiss_hnsw_m=self._faiss_hnsw_m,
                faiss_hnsw_ef_search=self._faiss_hnsw_ef_search,
                faiss_hnsw_ef_construction=self._faiss_hnsw_ef_construction,
            )
        if backend.dim != self._embed_dim:
            raise ValueError(
                f"backend dim {backend.dim} != embed_dim {self._embed_dim}"
            )
        self._backend: VectorBackend = backend
        self._backend.open()

        self._step: int = 0
        self._graph_rerank_alpha: float = float(graph_rerank_alpha)
        self._graph_rerank_stable_capture: bool = bool(graph_rerank_stable_capture)
        # Cursor: index of the next entry consolidate() should process.
        # Lets consolidate() be incremental — old entries already had
        # their growth-capture activations recorded in earlier calls.
        # Reset to 0 by attach_soma (a freshly-attached SOMA hasn't
        # seen any of the existing entries yet).
        self._consolidation_cursor: int = 0
        # Lazy stable-capture flag (Phase 13). ``consolidate()`` flips
        # this True when its growth pass updated weights / topology;
        # the next ``retrieve()`` with ``graph_rerank_alpha > 0`` runs
        # the O(N) stable-capture pass and clears the flag. Mutations
        # (``store``, ``forget``, WAL replay) also set it so the next
        # retrieve refreshes. Cost moves off the write path; with the
        # default ``graph_rerank_alpha=0.0`` the flag is never acted on
        # and stable-capture never runs.
        self._stable_capture_dirty: bool = False

        # Optional recall boosters — lazy, opt-in at retrieve time.
        # BM25 lexical index for hybrid search; rebuilt when stale.
        self._bm25_index: Any = None
        self._bm25_version: int = -1
        # Cross-encoder (or any Reranker) for re-ranking top-N.
        self._reranker: Any = None

        # Bundle name for metric labels. Defaulted to "__default__" so
        # in-memory MemoryLayers (bundle_path=None) still produce stable
        # labelled series. Callers running multi-tenant servers set this
        # explicitly when constructing or loading a bundle.
        self._bundle_name: str = "__default__"

        # Durability / persistence state. When bundle_path is None the
        # MemoryLayer runs in-memory only and keeps its pre-WAL behavior.
        self._bundle_path: Path | None = (
            Path(bundle_path) if bundle_path is not None else None
        )
        self._durability: Literal["sync", "batch", "async"] = durability
        self._wal: WAL | None = None
        self._lock_path: Path | None = None
        # Compaction state. ``_compaction_lock`` guards the spawn-at-most-
        # one-thread invariant; ``_compaction_thread`` is the in-flight
        # worker (or None/finished). Thresholds are instance-mutable so
        # tests can tune them down without waiting for the 10K default.
        self._compaction_lock: threading.Lock = threading.Lock()
        self._compaction_thread: threading.Thread | None = None
        self._compaction_size_floor: int = _COMPACTION_SIZE_FLOOR
        self._compaction_size_ratio: float = _COMPACTION_SIZE_RATIO
        self._compaction_record_threshold: int = _COMPACTION_RECORD_THRESHOLD
        self._compaction_age_seconds: float = _COMPACTION_AGE_SECONDS
        self._compaction_last_ts: float = time.monotonic()
        # Cursor + snapshot-size used by reload_if_stale and compaction
        # triggers. These are updated as the WAL grows / compacts.
        self._last_wal_offset: int = 0
        if self._bundle_path is not None:
            self._bundle_path.mkdir(parents=True, exist_ok=True)
            self._lock_path = self._bundle_path / "bundle.lock"
            # Ensure the lock file exists so portalocker.Lock can open it
            # in "r+" mode on both Windows and POSIX.
            self._lock_path.touch(exist_ok=True)
            self._wal = WAL(
                self._bundle_path,
                embed_dim=self._embed_dim,
                durability=self._durability,
            )
            self._wal.open()
            # For a freshly-opened WAL on a shared bundle, any existing
            # records belong to a peer. Start the cursor at 0 so the
            # first reload_if_stale() catches up; our own appends advance
            # it lazily via the same path.
            self._last_wal_offset = 0

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def with_sbert(
        cls,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: torch.device | str | None = None,
    ) -> MemoryLayer:
        """Create a MemoryLayer backed by a sentence-transformers model.

        Requires ``sentence-transformers`` to be installed (optional dep).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "MemoryLayer.with_sbert() requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        model = SentenceTransformer(model_name)
        raw_dim = model.get_sentence_embedding_dimension()
        if raw_dim is None:
            raise ValueError(
                f"SentenceTransformer({model_name!r}) has no reported "
                "embedding dimension; cannot build MemoryLayer."
            )
        dim = int(raw_dim)

        def _embed(text: str) -> torch.Tensor:
            return torch.tensor(model.encode(text, convert_to_numpy=True))

        return cls(embed_fn=_embed, embed_dim=dim, device=device)

    # ------------------------------------------------------------------
    # SOMA graph attachment (optional)
    # ------------------------------------------------------------------
    def attach_soma(
        self,
        soma: Any,
        tokenizer: Any,
        encoder: Any,
    ) -> None:
        """Attach a SOMA instance for graph-based consolidation.

        When attached, :meth:`consolidate` feeds stored entries through
        the SOMA graph, triggering structural plasticity (edge
        formation, pruning, myelination). Without attachment,
        ``consolidate`` remains a safe no-op.
        """
        self._soma = soma
        self._soma_tokenizer = tokenizer
        self._soma_encoder = encoder
        # New SOMA hasn't processed any existing entries yet. Reset
        # cursor so the next consolidate() catches up the whole store.
        self._consolidation_cursor = 0

    # ------------------------------------------------------------------
    # Durability helpers
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _bundle_lock(self) -> Any:
        """Acquire the bundle.lock sidecar while mutating the WAL.

        No-op when the MemoryLayer has no bundle attached (in-memory
        mode). Uses portalocker so two processes pointing at the same
        bundle dir serialize their writes without stepping on each
        other's WAL offsets.
        """
        if self._lock_path is None:
            yield
            return
        with portalocker.Lock(
            str(self._lock_path),
            mode="r+",
            timeout=30,
        ):
            yield

    def flush(self) -> None:
        """Force-sync the WAL to stable storage.

        No-op when no bundle is attached. Callers on ``durability="async"``
        or ``"batch"`` use this before a planned shutdown to close the
        durability gap.
        """
        if self._wal is None:
            return
        with self._bundle_lock(), _m.WAL_FLUSH_SECONDS.time():
            self._wal.flush()

    def reload_if_stale(self) -> int:
        """Apply any WAL records a peer writer appended since our last read.

        When two processes share a bundle (the multi-worker uvicorn case),
        the reader process's in-memory state can lag a writer's commits.
        This method re-reads the WAL tail past ``_last_wal_offset`` and
        applies each record on top of the in-memory state. If the WAL
        shrank (i.e. a peer ran compaction), we conservatively reset the
        cursor to zero and do a full re-replay from the snapshot.

        Returns the number of records applied. No-op (returns 0) when
        no bundle is attached or the WAL is unchanged.

        Meant to be called on retrieve paths in ``serve.py`` so every
        read sees the freshest committed state. Stores already take the
        bundle lock and see their own writes immediately, so we do NOT
        call this on the write path.
        """
        if self._wal is None:
            return 0
        with self._bundle_lock():
            if self._wal is None:
                return 0
            current_size = self._wal.ops_size_on_disk()
            if current_size == self._last_wal_offset:
                return 0
            if current_size < self._last_wal_offset:
                # WAL shrank (compaction by a peer, or truncate). We
                # can't trust our in-memory state anymore; reset and
                # do a full replay. This is rare — only fires when a
                # peer process compacts the shared bundle.
                self._last_wal_offset = 0
            applied = 0
            for rec in self._wal.replay_tail(self._last_wal_offset):
                self._apply_record(rec)
                applied += 1
            self._last_wal_offset = current_size
            if applied:
                # Backend already invalidated on each add/remove inside
                # _apply_record; we just emit the reload metric here.
                _m.RELOAD_TOTAL.labels(bundle=_m._bundle_label(self._bundle_name)).inc()
            return applied

    def _apply_record(self, rec: WalRecord) -> None:
        """Apply one WAL record to the in-memory state. Used by
        reload_if_stale; mirrors the branches in :meth:`load`."""
        if rec.op == "store":
            if rec.node_id in self._id_to_idx:
                return  # already applied — idempotent on repeated replay
            self._id_to_idx[rec.node_id] = len(self._ids)
            self._ids.append(rec.node_id)
            self._texts.append(rec.text or "")
            self._metadatas.append(dict(rec.metadata))
            self._timestamps.append(int(rec.timestamp_step))
            emb = rec.embedding
            assert emb is not None
            self._backend.add([rec.node_id], _vec_to_np(emb))
            self._soma_activations[rec.node_id] = None
            self._step = max(self._step, int(rec.timestamp_step) + 1)
            if self._graph_rerank_stable_capture:
                self._stable_capture_dirty = True
        elif rec.op == "forget":
            idx = self._id_to_idx.pop(rec.node_id, None)
            if idx is None:
                return
            self._ids.pop(idx)
            self._texts.pop(idx)
            self._metadatas.pop(idx)
            self._timestamps.pop(idx)
            self._backend.remove([rec.node_id])
            self._soma_activations.pop(rec.node_id, None)
            for later_id in self._ids[idx:]:
                self._id_to_idx[later_id] -= 1
            if self._consolidation_cursor > len(self._texts):
                self._consolidation_cursor = len(self._texts)
            self._step = max(self._step, int(rec.timestamp_step) + 1)
            if self._graph_rerank_stable_capture:
                self._stable_capture_dirty = True
        elif rec.op == "update_metadata":
            idx = self._id_to_idx.get(rec.node_id)
            if idx is None:
                return
            patch = rec.metadata.get("patch", {})
            if isinstance(patch, dict):
                self._metadatas[idx].update(patch)
            self._step = max(self._step, int(rec.timestamp_step) + 1)

    def close(self) -> None:
        """Close the WAL, flushing any buffered state. Safe to call
        multiple times; safe when no bundle is attached.

        Joins any in-flight compaction thread first so the on-disk
        bundle is in a consistent state by the time close() returns.
        """
        thread = self._compaction_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=30.0)
        # Backend.close is idempotent — safe to call every time.
        with contextlib.suppress(Exception):
            self._backend.close()
        if self._wal is None:
            return
        with self._bundle_lock():
            self._wal.close()
        self._wal = None

    # ------------------------------------------------------------------
    # Compaction — background snapshot rewrite + WAL truncate
    # ------------------------------------------------------------------
    def _maybe_compact(self) -> None:
        """Spawn a compaction thread when the WAL exceeds bounds.

        Triggers on any of:
          - record count > ``_compaction_record_threshold`` (10 000 default)
          - WAL size bytes > ``max(4 MB, 1.0 x snapshot)``
          - time since last compaction > ``_compaction_age_seconds`` (1 h)

        Only one compaction runs at a time. A second trigger while the
        first is in flight is a no-op — ``_compaction_lock`` is
        non-blocking and ``_compaction_thread.is_alive()`` gates the
        spawn. The running thread re-reads the WAL at the point it
        grabs the bundle lock, so a trigger missed during in-flight
        compaction is caught by the next store/store_batch/forget.

        Call from within the bundle lock (so the trigger snapshot is
        coherent with the mutation that just landed).
        """
        if self._wal is None or self._bundle_path is None:
            return
        # Only one in-flight compaction at a time. Non-blocking; a
        # concurrent trigger sees the first still running and returns.
        if not self._compaction_lock.acquire(blocking=False):
            return
        try:
            thread = self._compaction_thread
            if thread is not None and thread.is_alive():
                return
            if not self._compaction_should_fire():
                return
            self._compaction_thread = threading.Thread(
                target=self._run_compaction,
                name="soma-memory-compaction",
                daemon=True,
            )
            self._compaction_thread.start()
        finally:
            self._compaction_lock.release()

    def _compaction_should_fire(self) -> bool:
        """Evaluate the three trigger conditions against the live WAL."""
        if self._wal is None or self._bundle_path is None:
            return False
        wal = self._wal
        if wal.record_count > self._compaction_record_threshold:
            return True
        snap_path = self._bundle_path / "memory_embeddings.pt"
        snap_size = snap_path.stat().st_size if snap_path.exists() else 0
        size_floor = max(self._compaction_size_floor, int(self._compaction_size_ratio * snap_size))
        if wal.size_bytes > size_floor:
            return True
        if time.monotonic() - self._compaction_last_ts > self._compaction_age_seconds:
            # Only fire on age if there's actually something to compact.
            return wal.record_count > 0
        return False

    def _run_compaction(self) -> None:
        """Background worker: copy state under lock, write snapshot
        outside the lock, then re-acquire to swap + truncate WAL.

        The snapshot write uses :func:`_atomic_torch_save` and
        :func:`_atomic_write_bytes` so a crash mid-compaction leaves
        the previous snapshot intact. The WAL is only truncated AFTER
        the new snapshot is in place.
        """
        try:
            # --- Step 1: snapshot the in-memory state refs. -----------
            # We hold the bundle lock just long enough to copy references
            # (O(N) on list copies, but no embedding or encoder work).
            # Readers see the old snapshot + WAL until we flip atomically.
            with self._bundle_lock():
                if self._wal is None or self._bundle_path is None:
                    return
                ids_snap = list(self._ids)
                texts_snap = list(self._texts)
                meta_snap = [dict(m) for m in self._metadatas]
                ts_snap = list(self._timestamps)
                # Pull vectors out of the backend while still under the
                # lock so the snapshot is coherent with the id list.
                if ids_snap:
                    emb_np_snap = self._backend.get_vectors(ids_snap)
                else:
                    emb_np_snap = np.empty((0, self._embed_dim), dtype=np.float32)
                step_snap = self._step
                encoder_snap = self._encoder
                has_encoder = encoder_snap is not None
                encoder_state = (
                    encoder_snap.state_dict() if encoder_snap is not None else None
                )
                encoder_max_seq = (
                    int(encoder_snap.max_seq_len) if encoder_snap is not None else None
                )

            # --- Step 2: write the new snapshot outside the lock. -----
            # Each file goes to a sibling .tmp + os.replace so readers
            # never observe a half-written bundle.
            bundle = self._bundle_path
            assert bundle is not None
            if has_encoder:
                # Tokenizer save path already writes atomically.
                assert self._encoder is not None
                self._encoder.save_tokenizer(bundle / "tokenizer.json")
                assert encoder_state is not None
                _atomic_torch_save(encoder_state, bundle / "encoder.pt")
            if emb_np_snap.size > 0:
                stacked = torch.from_numpy(emb_np_snap.copy())
            else:
                stacked = torch.empty((0, self._embed_dim))
            _atomic_torch_save(stacked, bundle / "memory_embeddings.pt")
            index: dict[str, Any] = {
                "schema_version": 2,
                "embed_dim": self._embed_dim,
                "embed_type": "text_encoder" if has_encoder else "custom",
                "step": step_snap,
                "entries": [
                    {
                        "node_id": nid,
                        "text": txt,
                        "metadata": md,
                        "timestamp_step": ts,
                    }
                    for nid, txt, md, ts in zip(
                        ids_snap, texts_snap, meta_snap, ts_snap, strict=True
                    )
                ],
            }
            if encoder_max_seq is not None:
                index["max_seq_len"] = encoder_max_seq
            _atomic_write_bytes(
                bundle / "memory_index.json",
                json.dumps(index, indent=2).encode("utf-8"),
            )

            # --- Step 3: re-acquire lock, truncate the WAL. -----------
            # The snapshot is now on disk. Anything the WAL held prior
            # to our Step 1 snapshot is redundant. WAL appends that
            # landed during Steps 1-2 are lost on truncate — so we MUST
            # snapshot the WAL's live state before truncating too.
            with self._bundle_lock():
                if self._wal is None:
                    return
                # Replay any WAL records appended after our snapshot
                # and re-append them on top of the freshly truncated
                # WAL so no committed store is dropped. We keep every
                # record whose node_id was NOT in the snapshot (= new
                # store since Step 1) plus every forget tombstone whose
                # target id appeared in the snapshot (so a forget of a
                # pre-snapshot id still replays on next load). Forgets
                # of post-snapshot ids are preserved too — their
                # matching store record is already in the tail list, so
                # replay order will store-then-forget correctly.
                # update_metadata records whose target is in the snapshot
                # were already materialized into meta_snap above, so they
                # are redundant and we drop them. Post-snapshot ones
                # (target id not in snapshot) must be kept; they'll
                # replay after the corresponding store in the tail.
                existing_ids = set(ids_snap)
                tail_records: list[WalRecord] = []
                for rec in self._wal.replay():
                    if rec.op == "store" and rec.node_id in existing_ids:
                        # Already in snapshot; skip.
                        continue
                    if (
                        rec.op == "update_metadata"
                        and rec.node_id in existing_ids
                    ):
                        # Patch is already applied to snapshot metadata; skip.
                        continue
                    tail_records.append(rec)
                self._wal.truncate()
                for rec in tail_records:
                    self._wal.append(rec)
                self._last_wal_offset = 0
                self._compaction_last_ts = time.monotonic()
        except Exception:
            # Best-effort: log via the background thread's failure; we
            # don't want compaction crashes to kill the main process.
            # The next store/forget will retry the trigger.
            pass

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def store(self, text: str, *, metadata: dict[str, Any] | None = None) -> str:
        """Add an entry. Returns a stable node_id (uuid4 hex)."""
        if not text or not text.strip():
            raise ValueError("MemoryLayer.store rejects empty text")
        node_id = uuid.uuid4().hex
        embedding = self._embed(text)
        meta_dict = dict(metadata) if metadata else {}
        ts_step = self._step
        # Under a bundle lock: WAL append first (committed on disk before
        # we mutate in-memory state), then in-memory mutation. If the
        # append raises, we leave the in-memory state untouched.
        with self._bundle_lock():
            if self._wal is not None:
                self._wal.append(
                    WalRecord(
                        op="store",
                        node_id=node_id,
                        text=text,
                        metadata=meta_dict,
                        timestamp_step=ts_step,
                        embedding=embedding,
                        emb_offset=None,
                    )
                )
                _m.WAL_APPEND_TOTAL.labels(op="store").inc()
            self._id_to_idx[node_id] = len(self._ids)
            self._ids.append(node_id)
            self._texts.append(text)
            self._metadatas.append(meta_dict)
            self._timestamps.append(ts_step)
            self._backend.add([node_id], _vec_to_np(embedding))
            self._soma_activations[node_id] = None
            self._step += 1
            self._stores_since_consolidation += 1
            # New entry — prior stable-capture no longer covers the
            # full store. Next retrieve with graph re-rank will refresh.
            if self._graph_rerank_stable_capture:
                self._stable_capture_dirty = True
            if self._wal is not None:
                self._last_wal_offset = self._wal.ops_size_on_disk()
            self._maybe_compact()
        bundle_label = _m._bundle_label(self._bundle_name)
        _m.STORE_TOTAL.labels(bundle=bundle_label).inc()
        _m.ENTRIES.labels(bundle=bundle_label).set(len(self._ids))
        if (
            self._auto_consolidate_every > 0
            and self._soma is not None
            and self._stores_since_consolidation >= self._auto_consolidate_every
        ):
            self.consolidate()
            self._stores_since_consolidation = 0
        return node_id

    def store_batch(
        self,
        texts: list[str],
        *,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add many entries in one shot. Returns a list of node_ids in input order.

        When the encoder exposes ``encode_batch_many`` (sbert does), we call
        it once for the whole batch instead of N round-trips; this is the
        main reason to prefer ``store_batch`` over a loop of ``store``. The
        FAISS index is invalidated once at the end rather than per-entry.

        Acquires the bundle lock ONCE for the whole batch, so N records
        cost one fsync cycle under ``durability="sync"`` rather than N.
        """
        if not texts:
            return []
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length {len(metadatas)} != texts length {len(texts)}"
            )
        for t in texts:
            if not t or not t.strip():
                raise ValueError("MemoryLayer.store_batch rejects empty text")
        embeddings = self._embed_batch(texts)
        node_ids: list[str] = []
        with self._bundle_lock():
            for i, text in enumerate(texts):
                nid = uuid.uuid4().hex
                meta_dict = dict(metadatas[i]) if metadatas is not None else {}
                ts_step = self._step
                if self._wal is not None:
                    self._wal.append(
                        WalRecord(
                            op="store",
                            node_id=nid,
                            text=text,
                            metadata=meta_dict,
                            timestamp_step=ts_step,
                            embedding=embeddings[i],
                            emb_offset=None,
                        )
                    )
                    _m.WAL_APPEND_TOTAL.labels(op="store").inc()
                self._id_to_idx[nid] = len(self._ids)
                self._ids.append(nid)
                self._texts.append(text)
                self._metadatas.append(meta_dict)
                self._timestamps.append(ts_step)
                self._soma_activations[nid] = None
                self._step += 1
                node_ids.append(nid)
            # One batch insert into the backend — cheaper than
            # N individual add() calls when the backend builds/rebuilds
            # an index per call.
            self._backend.add(node_ids, _batch_to_np(embeddings))
            self._stores_since_consolidation += len(texts)
            # Batch grew the store — any prior stable-capture is now
            # incomplete. Next retrieve with graph re-rank refreshes.
            if self._graph_rerank_stable_capture:
                self._stable_capture_dirty = True
            if self._wal is not None:
                self._last_wal_offset = self._wal.ops_size_on_disk()
            self._maybe_compact()
        bundle_label = _m._bundle_label(self._bundle_name)
        _m.STORE_BATCH_TOTAL.labels(bundle=bundle_label).inc()
        _m.ENTRIES.labels(bundle=bundle_label).set(len(self._ids))
        if (
            self._auto_consolidate_every > 0
            and self._soma is not None
            and self._stores_since_consolidation >= self._auto_consolidate_every
        ):
            self.consolidate()
            self._stores_since_consolidation = 0
        return node_ids

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        where: dict[str, Any] | None = None,
        hybrid_alpha: float | None = None,
        rerank_top_n: int | None = None,
    ) -> list[MemoryHit]:
        """Return up to k entries most similar to ``query`` by cosine.

        When a SOMA graph is attached AND consolidation has been run
        (producing stored SOMA output activations), retrieval uses a
        two-stage pipeline: cosine candidates are re-ranked by a blend
        of cosine score and graph-proximity score derived from SOMA's
        output activations.

        Optional parameters:

        - ``where``: metadata filter applied *before* ranking so we
          don't run out of candidates on selective filters. Supported
          forms::

              {"field": "value"}                 # exact match
              {"field1": "a", "field2": "b"}     # AND across fields
              {"field": {"$in": ["a", "b"]}}     # value-list match
              {"field": {"$ne": "x"}}            # not-equal
              {"field": {"$gt": 3}}              # comparisons ($gt/$lt/$gte/$lte)

          Matches Chroma's ``where`` semantics for the subset people
          actually use. Missing fields on an entry fail the filter.
        - ``hybrid_alpha`` in [0, 1]: blend cosine with BM25 lexical
          scores. ``0.0`` = pure BM25, ``1.0`` = pure cosine, ``0.5``
          is a reasonable default. BM25 index is built lazily.
        - ``rerank_top_n``: over-fetch ``rerank_top_n`` candidates and
          re-score with the attached :class:`Reranker` (see
          :meth:`attach_reranker`). Typical: ``3-5× k``.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self._ids:
            return []
        if hybrid_alpha is not None and not 0.0 <= hybrid_alpha <= 1.0:
            raise ValueError(f"hybrid_alpha must be in [0,1], got {hybrid_alpha}")

        backend = self._backend.name
        started = time.monotonic()
        q_vec = self._embed(query)
        # Lazy stable-capture (Phase 13). If the graph re-rank is
        # active and a prior consolidate() / mutation left the capture
        # stale, refresh it before reading activations. alpha=0 keeps
        # the dirty flag set and skips the O(N) pass entirely — the
        # production default path. ``stable_capture()`` is the public
        # primitive and early-returns if the feature is disabled / no
        # SOMA attached, so the gate here is just an alpha+dirty check.
        if self._graph_rerank_alpha > 0.0 and self._stable_capture_dirty:
            self.stable_capture()
        has_graph_signal = (
            self._graph_rerank_alpha > 0.0
            and self._soma is not None
            and any(a is not None for a in self._soma_activations.values())
        )

        # Pre-rerank candidate pool: hybrid > graph rerank > plain cosine.
        candidate_k = rerank_top_n if rerank_top_n else k
        if where is not None:
            # Filter-pushdown dispatch: backends that can translate the
            # Chroma-style ``where`` clause into their native filter
            # language run the query server-side in one round-trip. If
            # the backend rejects an operator we fall through to the
            # Python pre-filter + subset path that every backend can do.
            candidates = self._retrieve_with_filter(
                query,
                q_vec,
                where=where,
                k=candidate_k,
                hybrid_alpha=hybrid_alpha,
            )
        elif hybrid_alpha is not None:
            candidates = self._retrieve_hybrid(
                query, q_vec, k=candidate_k, alpha=hybrid_alpha
            )
        elif has_graph_signal:
            candidates = self._retrieve_with_rerank(query, q_vec, k=candidate_k)
        else:
            candidates = self._rank(q_vec, k=candidate_k, exclude_idx=None)

        if rerank_top_n and self._reranker is not None and candidates:
            candidates = self._apply_reranker(query, candidates, top_k=k)
        elif rerank_top_n and self._reranker is None:
            raise ValueError(
                "rerank_top_n set but no reranker attached. "
                "Call mem.attach_reranker(CrossEncoderReranker()) first."
            )
        results = candidates[:k]
        elapsed = time.monotonic() - started
        bundle_label = _m._bundle_label(self._bundle_name)
        _m.RETRIEVE_TOTAL.labels(bundle=bundle_label, backend=backend).inc()
        _m.RETRIEVE_LATENCY.labels(bundle=bundle_label, backend=backend).observe(elapsed)
        # Single structured-log line per retrieve so operators can trace
        # every lookup in Loki/Datadog/CloudWatch without parsing format
        # strings. Schema is pinned in ``docs/observability.md`` and
        # ``tests/test_memory/test_retrieve_log_line.py``; adding keys is
        # safe, removing/renaming is a breaking change.
        logger.info(
            "retrieve",
            extra={
                "event": "retrieve",
                "bundle": self._bundle_name,
                "query_len": len(query),
                "k": k,
                "has_where": where is not None,
                "hybrid_alpha": hybrid_alpha,
                "rerank_top_n": rerank_top_n,
                "n_hits": len(results),
                "backend": backend,
                "latency_ms": round(elapsed * 1000.0, 3),
                "cache_miss": False,
            },
        )
        return results

    def related(self, node_id: str, k: int = 5) -> list[MemoryHit]:
        """Return up to k entries most similar to the entry at ``node_id``."""
        if node_id not in self._id_to_idx:
            raise KeyError(f"node_id {node_id!r} not found in MemoryLayer")
        # Fetch the stored vector from the backend and use it as the
        # query. For HTTP adapters this is one extra round-trip; we
        # accept that for v1 (see plan §Risks).
        q_np = self._backend.get_vectors([node_id])[0]
        pairs = self._backend.search(q_np, k=k, exclude_ids={node_id})
        return [self._hit_for_id(nid, score=s) for nid, s in pairs]

    # ------------------------------------------------------------------
    # Recall boosters — hybrid lexical + cross-encoder re-ranking
    # ------------------------------------------------------------------
    def attach_reranker(self, reranker: Any) -> None:
        """Attach any object satisfying the :class:`Reranker` protocol
        (``score(query, candidates) -> list[float]``). After attaching,
        pass ``rerank_top_n=N`` to :meth:`retrieve` to over-fetch N
        cosine candidates and re-rank them with this model."""
        self._reranker = reranker

    def _maybe_build_bm25(self) -> None:
        """Lazily (re)build the BM25 index when stale."""
        if self._bm25_index is not None and self._bm25_version == self._step:
            return
        from soma.memory.bm25 import BM25Index

        with _m.BM25_REBUILD_SECONDS.time():
            idx = BM25Index()
            idx.build(list(self._texts))
            self._bm25_index = idx
            self._bm25_version = self._step
        _m.BM25_REBUILD_TOTAL.inc()

    def _retrieve_hybrid(
        self, query: str, q_vec: torch.Tensor, *, k: int, alpha: float
    ) -> list[MemoryHit]:
        """Blend cosine + BM25 with weight ``alpha`` on cosine.

        Both score streams are normalized to [0, 1] via max-normalization
        before blending so a heavy-tailed BM25 score doesn't swamp the
        bounded cosine score (or vice versa).
        """
        self._maybe_build_bm25()
        # Over-fetch a union from both sides so we don't lose items that
        # one side ranks high and the other ignores.
        pool_k = min(max(k * 3, 20), len(self._ids))
        cosine_hits = self._rank(q_vec, k=pool_k, exclude_idx=None)
        bm25_hits = self._bm25_index.search(query, pool_k) if self._bm25_index else []

        cos_max = max((h.score for h in cosine_hits), default=1e-9)
        bm25_max = max((s for _, s in bm25_hits), default=1e-9)

        cos_scores: dict[str, float] = {
            h.node_id: (h.score / cos_max if cos_max > 0 else 0.0)
            for h in cosine_hits
        }
        bm25_scores: dict[str, float] = {
            self._ids[i]: (s / bm25_max if bm25_max > 0 else 0.0)
            for i, s in bm25_hits
        }

        blended: dict[str, float] = {}
        for node_id in set(cos_scores) | set(bm25_scores):
            c = cos_scores.get(node_id, 0.0)
            b = bm25_scores.get(node_id, 0.0)
            blended[node_id] = alpha * c + (1.0 - alpha) * b

        # Materialize MemoryHits with blended scores, sorted descending.
        id_to_idx = {nid: i for i, nid in enumerate(self._ids)}
        ranked = sorted(blended.items(), key=lambda kv: -kv[1])[:k]
        out: list[MemoryHit] = []
        for nid, score in ranked:
            i = id_to_idx[nid]
            out.append(
                MemoryHit(
                    node_id=nid,
                    text=self._texts[i],
                    score=float(score),
                    metadata=dict(self._metadatas[i]),
                    timestamp_step=self._timestamps[i],
                )
            )
        return out

    def _apply_reranker(
        self, query: str, candidates: list[MemoryHit], *, top_k: int
    ) -> list[MemoryHit]:
        """Re-score ``candidates`` with the attached reranker and return
        them in descending rerank order. The original cosine/hybrid
        score is preserved in ``metadata['_pre_rerank_score']`` so
        callers can compare if they want."""
        if not candidates or self._reranker is None:
            return candidates
        texts = [h.text for h in candidates]
        scores = self._reranker.score(query, texts)
        rescored = []
        for h, s in zip(candidates, scores, strict=True):
            meta = dict(h.metadata)
            meta["_pre_rerank_score"] = h.score
            rescored.append(
                MemoryHit(
                    node_id=h.node_id,
                    text=h.text,
                    score=float(s),
                    metadata=meta,
                    timestamp_step=h.timestamp_step,
                )
            )
        rescored.sort(key=lambda h: -h.score)
        return rescored[:top_k]

    def _rank_subset(
        self, q_vec: torch.Tensor, indices: list[int], *, k: int
    ) -> list[MemoryHit]:
        """Brute-force cosine over a pre-filtered subset of entries.

        Delegates the numeric work to ``backend.search_subset`` so the
        operation stays adapter-agnostic. Translates list-positions to
        node_ids on the way in and back again on the way out.
        """
        if not indices:
            return []
        candidate_ids = [self._ids[i] for i in indices]
        q_np = _vec_to_np(q_vec).reshape(-1)
        pairs = self._backend.search_subset(q_np, candidate_ids, k=k)
        return [self._hit_for_id(nid, score=s) for nid, s in pairs]

    def _blend_bm25_subset(
        self,
        query: str,
        cos_hits: list[MemoryHit],
        indices: list[int],
        *,
        alpha: float,
        k: int,
    ) -> list[MemoryHit]:
        """Hybrid cosine+BM25 restricted to the filtered subset."""
        self._maybe_build_bm25()
        assert self._bm25_index is not None
        bm25_all = self._bm25_index.search(query, len(self._ids))
        allowed = set(indices)
        bm25_filtered = [(i, s) for i, s in bm25_all if i in allowed]
        cos_max = max((h.score for h in cos_hits), default=1e-9)
        bm25_max = max((s for _, s in bm25_filtered), default=1e-9)
        cos_scores: dict[str, float] = {
            h.node_id: (h.score / cos_max if cos_max > 0 else 0.0)
            for h in cos_hits
        }
        bm25_scores: dict[str, float] = {
            self._ids[i]: (s / bm25_max if bm25_max > 0 else 0.0)
            for i, s in bm25_filtered
        }
        blended: dict[str, float] = {}
        for node_id in set(cos_scores) | set(bm25_scores):
            c = cos_scores.get(node_id, 0.0)
            b = bm25_scores.get(node_id, 0.0)
            blended[node_id] = alpha * c + (1.0 - alpha) * b
        id_to_idx = {nid: i for i, nid in enumerate(self._ids)}
        ranked = sorted(blended.items(), key=lambda kv: -kv[1])[:k]
        return [
            self._hit_for_index(id_to_idx[nid], score=float(score))
            for nid, score in ranked
        ]

    def get(self, node_id: str) -> MemoryHit | None:
        """Fetch an entry by id; ``None`` if unknown. Score is self-cosine (1.0)."""
        idx = self._id_to_idx.get(node_id)
        if idx is None:
            return None
        return MemoryHit(
            node_id=node_id,
            text=self._texts[idx],
            score=1.0,
            metadata=dict(self._metadatas[idx]),
            timestamp_step=self._timestamps[idx],
        )

    def get_recent(self, n: int) -> list[MemoryHit]:
        """Return the n most recently stored entries, newest first."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        start = max(0, len(self._ids) - n)
        recent_indices = list(range(start, len(self._ids)))[::-1]
        return [self._hit_for_index(i, score=1.0) for i in recent_indices]

    def update_metadata(
        self, node_id: str, patch: dict[str, Any]
    ) -> None:
        """Merge ``patch`` into the metadata of an existing entry.

        Keys in ``patch`` overwrite existing keys; keys not in ``patch``
        are preserved. WAL-replay compatible: appends an
        ``update_metadata`` record before mutating in-memory state, so a
        crash after append + before the in-memory mutation replays the
        patch on next load.

        Raises :class:`KeyError` if ``node_id`` is unknown.

        This is the load-bearing primitive behind
        :class:`soma.memory.conversational.ConversationalMemory`'s
        SUPERSEDE op (which sets ``superseded_by`` on the old entry
        instead of deleting it, preserving history for audit).
        """
        patch_dict = dict(patch)
        with self._bundle_lock():
            idx = self._id_to_idx.get(node_id)
            if idx is None:
                raise KeyError(f"node_id {node_id!r} not found in MemoryLayer")
            if self._wal is not None:
                self._wal.append(
                    WalRecord(
                        op="update_metadata",
                        node_id=node_id,
                        text=None,
                        metadata={"patch": patch_dict},
                        timestamp_step=self._step,
                        embedding=None,
                        emb_offset=None,
                    )
                )
                _m.WAL_APPEND_TOTAL.labels(op="update_metadata").inc()
            self._metadatas[idx].update(patch_dict)
            if self._wal is not None:
                self._last_wal_offset = self._wal.ops_size_on_disk()
            self._maybe_compact()

    def forget(self, node_id: str) -> bool:
        """Remove an entry. Returns True if removed, False if unknown.

        Appends a ``forget`` tombstone to the WAL (if attached) before
        mutating in-memory state, so a crash after append + before the
        in-memory pop still has the tombstone on disk for replay.
        """
        with self._bundle_lock():
            idx = self._id_to_idx.get(node_id)
            if idx is None:
                return False
            if self._wal is not None:
                self._wal.append(
                    WalRecord(
                        op="forget",
                        node_id=node_id,
                        text=None,
                        metadata={},
                        timestamp_step=self._step,
                        embedding=None,
                        emb_offset=None,
                    )
                )
                _m.WAL_APPEND_TOTAL.labels(op="forget").inc()
            self._id_to_idx.pop(node_id, None)
            self._ids.pop(idx)
            self._texts.pop(idx)
            self._metadatas.pop(idx)
            self._timestamps.pop(idx)
            self._backend.remove([node_id])
            self._soma_activations.pop(node_id, None)
            for later_id in self._ids[idx:]:
                self._id_to_idx[later_id] -= 1
            # Keep the consolidation cursor valid: it's an int into the
            # texts list, and forget() just shrank that list by one.
            if self._consolidation_cursor > len(self._texts):
                self._consolidation_cursor = len(self._texts)
            # Forget mutates the stored set — mark stable-capture stale
            # (remaining entries' captures are still valid individually,
            # but the dirty flag is cheap to clear on the next retrieve).
            if self._graph_rerank_stable_capture:
                self._stable_capture_dirty = True
            if self._wal is not None:
                self._last_wal_offset = self._wal.ops_size_on_disk()
            self._maybe_compact()
        bundle_label = _m._bundle_label(self._bundle_name)
        _m.FORGET_TOTAL.labels(bundle=bundle_label).inc()
        _m.ENTRIES.labels(bundle=bundle_label).set(len(self._ids))
        return True

    def consolidate(self) -> int:
        """Push stored entries through SOMA's graph to trigger plasticity.

        When a SOMA instance is attached via :meth:`attach_soma`, this
        feeds each stored text through the graph (one ``step()`` per
        entry) so structural plasticity — synaptogenesis, pruning,
        myelination — fires based on the content patterns. Returns
        the number of entries processed.

        Without an attached SOMA, this is a safe no-op (returns 0).
        Callers should include ``consolidate()`` in their loops now; it
        becomes load-bearing once a SOMA is attached.

        Emits both the legacy ``soma_consolidate_*`` metrics and the
        Phase 8 ``soma_compaction_*`` labelled variants. The compaction
        counter records ``outcome="ok" | "error"`` so operators can
        alert on a rising error ratio without parsing logs.
        """
        _m.CONSOLIDATE_TOTAL.inc()
        bundle_label = _m._bundle_label(self._bundle_name)
        started = time.monotonic()
        outcome = "ok"
        try:
            with _m.CONSOLIDATE_SECONDS.time():
                return self._consolidate_impl()
        except BaseException:
            outcome = "error"
            raise
        finally:
            elapsed = time.monotonic() - started
            _m.COMPACTION_TOTAL.labels(bundle=bundle_label, outcome=outcome).inc()
            _m.COMPACTION_SECONDS.labels(bundle=bundle_label).observe(elapsed)

    def _consolidate_impl(self) -> int:
        """Internal body of consolidate(), wrapped by metrics in the caller.

        Phase 13: stable-capture moved off the write path. We run the
        growth pass here (SOMA.step under ``eval_mode=False`` for new
        entries), then flip ``_stable_capture_dirty`` so the next
        retrieve that actually consumes graph activations refreshes
        them. Callers that want to pay the cost eagerly (benchmarks,
        cold-start warmup) invoke :meth:`stable_capture` directly.
        """
        if self._soma is None:
            return 0
        self._stores_since_consolidation = 0
        from soma.io.verbalizer import SomaAggregator

        soma_output_dim = int(self._soma.config.sensor_output_dim)

        # Incremental growth pass: only process entries past the cursor.
        # Old entries already had their growth-time activations recorded
        # in prior consolidate() calls — re-running them would do
        # redundant SOMA steps and balloon cost from O(new_entries) to
        # O(N) per call.
        processed = 0
        start = self._consolidation_cursor
        for entry_idx in range(start, len(self._texts)):
            text = self._texts[entry_idx]
            nid = self._ids[entry_idx]
            token_embeddings = self._soma_encoder.encode(text)
            if len(token_embeddings) < 2:
                continue
            detached = [e.detach() for e in token_embeddings]
            for i in range(len(detached) - 1):
                inputs = {"text": detached[i]}
                targets = {"text": detached[i + 1]}
                self._soma.step(inputs, targets=targets, eval_mode=False)
            output_acts = self._soma._current_output_activations()
            pooled = SomaAggregator.collapse(
                output_acts,
                soma_output_dim=soma_output_dim,
            )
            self._soma_activations[nid] = pooled.detach().cpu()
            processed += 1
        self._consolidation_cursor = len(self._texts)

        # Stable-capture is now lazy (Phase 13). If the growth pass
        # processed anything, prior stored activations are no longer
        # comparable to queries under the current graph + weights —
        # even without nodes/edges changing, Hebbian + backprop run on
        # every ``soma.step()`` so weights drift. Mark dirty so the
        # next retrieve that needs graph activations refreshes them.
        # With ``graph_rerank_alpha=0.0`` (shipping default) the
        # retrieve path never reads the capture, so the flag stays
        # set and the O(N) pass simply never runs.
        if self._graph_rerank_stable_capture and processed > 0:
            self._stable_capture_dirty = True
        return processed

    def stable_capture(self) -> None:
        """Re-run every stored entry through the post-growth graph (eval_mode).

        Public entry point for the stable-capture pass. The ordinary
        ``retrieve`` path calls this lazily when
        ``graph_rerank_alpha > 0`` and the dirty flag is set;
        benchmarks and warmup loops that want predictable retrieve
        latency invoke it explicitly instead (``mem.stable_capture()``
        before the first hot retrieve).

        Safe no-op when no SOMA is attached or
        ``graph_rerank_stable_capture=False``. On success the dirty
        flag is cleared so subsequent retrieves (with no intervening
        mutation) skip the pass entirely.

        Cost is O(N_stored) SOMA.step calls in ``eval_mode=True`` — no
        weight updates, no growth. See
        ``docs/plans/2026-04-16-lazy-stable-capture.md`` for the
        rationale behind hoisting this off the consolidate() write
        path.
        """
        if self._soma is None or not self._graph_rerank_stable_capture:
            return
        soma_output_dim = int(self._soma.config.sensor_output_dim)
        self._recapture_activations_stable(soma_output_dim)
        self._stable_capture_dirty = False

    def _recapture_activations_stable(self, soma_output_dim: int) -> None:
        """Internal stable-capture worker — called by :meth:`stable_capture`.

        During growth (``eval_mode=False``) the graph mutates between
        entries, so activations captured inline are snapshots of
        *different* graphs — not directly comparable to the activation
        computed for a query at retrieval time, which sees the post-growth
        graph. A second pass in eval_mode (no growth, no weight updates)
        re-captures every stored entry's activation under the same graph
        that queries will encounter, restoring the comparability the
        cosine-style re-rank blend implicitly assumes.
        """
        from soma.io.verbalizer import SomaAggregator

        for entry_idx, text in enumerate(self._texts):
            nid = self._ids[entry_idx]
            token_embeddings = self._soma_encoder.encode(text)
            if len(token_embeddings) < 2:
                continue
            detached = [e.detach() for e in token_embeddings]
            for i in range(len(detached) - 1):
                inputs = {"text": detached[i]}
                targets = {"text": detached[i + 1]}
                self._soma.step(inputs, targets=targets, eval_mode=True)
            output_acts = self._soma._current_output_activations()
            pooled = SomaAggregator.collapse(
                output_acts,
                soma_output_dim=soma_output_dim,
            )
            self._soma_activations[nid] = pooled.detach().cpu()

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._id_to_idx

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write a self-contained memory bundle to ``path`` (a directory).

        Each file is written to a sibling ``.tmp`` and swapped into
        place with ``os.replace`` so a crash mid-save leaves the old
        (consistent) bundle intact, never a half-written mix. Any
        leftover ``.tmp`` file is deleted on error.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        encoder = self._encoder
        has_encoder = encoder is not None
        if encoder is not None:
            # save_tokenizer goes through its own write path; torch.save
            # we route through the atomic wrapper.
            encoder.save_tokenizer(out / "tokenizer.json")
            _atomic_torch_save(encoder.state_dict(), out / "encoder.pt")
        if self._ids:
            # Pull stacked vectors out of the backend in id order so
            # snapshot rows line up with memory_index.json entries.
            stacked_np = self._backend.get_vectors(self._ids)
            stacked = torch.from_numpy(stacked_np.copy())
        else:
            stacked = torch.empty((0, self._embed_dim))
        _atomic_torch_save(stacked, out / "memory_embeddings.pt")
        index: dict[str, Any] = {
            # v1 = snapshot-only (pre-WAL). v2 = snapshot + optional WAL
            # sidecar. We emit v2 unconditionally now; v1 bundles still
            # load because load() accepts both.
            "schema_version": 2,
            "embed_dim": self._embed_dim,
            "embed_type": "text_encoder" if has_encoder else "custom",
            "step": self._step,
            "entries": [
                {
                    "node_id": nid,
                    "text": txt,
                    "metadata": md,
                    "timestamp_step": ts,
                }
                for nid, txt, md, ts in zip(
                    self._ids,
                    self._texts,
                    self._metadatas,
                    self._timestamps,
                    strict=True,
                )
            ],
        }
        if encoder is not None:
            index["max_seq_len"] = int(encoder.max_seq_len)
        _atomic_write_bytes(
            out / "memory_index.json",
            json.dumps(index, indent=2).encode("utf-8"),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embed_fn: EmbedFn | None = None,
        device: torch.device | str | None = None,
        durability: Literal["sync", "batch", "async"] = "sync",
    ) -> MemoryLayer:
        """Rehydrate a MemoryLayer from a ``save()``-produced directory.

        Bundles saved with the TextEncoder path include ``tokenizer.json``
        and ``encoder.pt``; those are reloaded automatically. Bundles saved
        with a custom ``embed_fn`` only store embeddings + index — pass the
        same ``embed_fn`` at load time so new stores can be embedded.

        If the bundle also has WAL sidecar files (``memory_ops.wal.jsonl``
        + ``memory_embeddings.wal.bin``), their records are replayed on
        top of the snapshot and the WAL stays open for subsequent writes.
        A bundle that only has WAL files (no snapshot yet — the common
        case for a brand-new store that never called ``save()``) loads
        from the WAL header for ``embed_dim`` and replays from there.
        """
        src = Path(path)
        index_path = src / "memory_index.json"
        embeddings_path = src / "memory_embeddings.pt"
        has_snapshot = index_path.exists() and embeddings_path.exists()
        wal_ops_path = src / "memory_ops.wal.jsonl"
        has_wal = wal_ops_path.exists()

        if not has_snapshot and not has_wal:
            raise FileNotFoundError(
                f"MemoryLayer bundle missing {index_path.name} (and no WAL found)"
            )

        # --- Determine the embedder + embed_dim. --------------------------
        # Preference: snapshot index.json (richer metadata). Fallback:
        # WAL header.
        index: dict[str, Any] | None = None
        if has_snapshot:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            schema = index.get("schema_version")
            if schema not in (1, 2):
                raise ValueError(
                    f"Unsupported MemoryLayer schema version {schema!r}"
                )
            embed_dim = int(index["embed_dim"])
            embed_type = index.get("embed_type", "text_encoder")
        else:
            # No snapshot — must derive embed_dim from the WAL header.
            first_line = wal_ops_path.read_text(encoding="utf-8").splitlines()[0]
            header = json.loads(first_line)
            embed_dim = int(header["embed_dim"])
            # No snapshot means no tokenizer/encoder files either →
            # caller must supply embed_fn.
            embed_type = "custom"

        if embed_type == "text_encoder":
            tokenizer_path = src / "tokenizer.json"
            encoder_path = src / "encoder.pt"
            for f in (tokenizer_path, encoder_path):
                if not f.exists():
                    raise FileNotFoundError(f"MemoryLayer bundle missing {f.name}")
            tokenizer = load_tokenizer(tokenizer_path)
            encoder = TextEncoder(
                tokenizer,
                embed_dim=embed_dim,
                max_seq_len=int((index or {}).get("max_seq_len", 512)),
                device=device,
            )
            encoder_state = torch.load(
                encoder_path,
                map_location=device or "cpu",
                weights_only=True,
            )
            encoder.load_state_dict(encoder_state)
            instance = cls(
                tokenizer=tokenizer,
                encoder=encoder,
                device=device,
                bundle_path=src,
                durability=durability,
            )
        else:
            if embed_fn is None:
                raise ValueError(
                    "This bundle was saved with a custom embed_fn; pass the "
                    "same embed_fn to load()."
                )
            instance = cls(
                embed_fn=embed_fn,
                embed_dim=embed_dim,
                device=device,
                bundle_path=src,
                durability=durability,
            )

        # --- Replay snapshot (if any). ------------------------------------
        if has_snapshot:
            assert index is not None
            instance._step = int(index.get("step", 0))
            embeddings = torch.load(
                embeddings_path, map_location=device or "cpu", weights_only=True
            )
            # Batch-insert into the backend in one call so adapters that
            # build per-call indices only do it once. Populate the
            # MemoryLayer lists in parallel.
            snap_ids: list[str] = []
            snap_vecs: list[torch.Tensor] = []
            for entry, vec in zip(index["entries"], embeddings, strict=True):
                nid = str(entry["node_id"])
                instance._id_to_idx[nid] = len(instance._ids)
                instance._ids.append(nid)
                instance._texts.append(str(entry["text"]))
                instance._metadatas.append(dict(entry.get("metadata", {})))
                instance._timestamps.append(int(entry.get("timestamp_step", 0)))
                instance._soma_activations[nid] = None
                snap_ids.append(nid)
                snap_vecs.append(vec)
            if snap_ids:
                instance._backend.add(snap_ids, _batch_to_np(snap_vecs))

        # --- Replay WAL on top of snapshot. -------------------------------
        # The WAL was opened during __init__; replay re-reads from disk.
        if instance._wal is not None:
            for rec in instance._wal.replay():
                if rec.op == "store":
                    if rec.node_id in instance._id_to_idx:
                        # Snapshot already had this id — WAL append was
                        # the same record, don't double-apply.
                        continue
                    instance._id_to_idx[rec.node_id] = len(instance._ids)
                    instance._ids.append(rec.node_id)
                    instance._texts.append(rec.text or "")
                    instance._metadatas.append(dict(rec.metadata))
                    instance._timestamps.append(int(rec.timestamp_step))
                    emb = rec.embedding
                    assert emb is not None
                    instance._backend.add([rec.node_id], _vec_to_np(emb))
                    instance._soma_activations[rec.node_id] = None
                    instance._step = max(
                        instance._step, int(rec.timestamp_step) + 1
                    )
                elif rec.op == "forget":
                    idx = instance._id_to_idx.pop(rec.node_id, None)
                    if idx is None:
                        continue
                    instance._ids.pop(idx)
                    instance._texts.pop(idx)
                    instance._metadatas.pop(idx)
                    instance._timestamps.pop(idx)
                    instance._backend.remove([rec.node_id])
                    instance._soma_activations.pop(rec.node_id, None)
                    for later_id in instance._ids[idx:]:
                        instance._id_to_idx[later_id] -= 1
                    instance._step = max(
                        instance._step, int(rec.timestamp_step) + 1
                    )
                elif rec.op == "update_metadata":
                    idx = instance._id_to_idx.get(rec.node_id)
                    if idx is None:
                        continue
                    patch = rec.metadata.get("patch", {})
                    if isinstance(patch, dict):
                        instance._metadatas[idx].update(patch)
                    instance._step = max(
                        instance._step, int(rec.timestamp_step) + 1
                    )
        # Mark where we've caught up to; reload_if_stale() starts scanning
        # from here so a peer writer's tail appends are cheap to detect.
        if instance._wal is not None:
            instance._last_wal_offset = instance._wal.ops_size_on_disk()
        return instance

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> torch.Tensor:
        """Embed ``text`` into a (embed_dim,) vector."""
        with _m.EMBED_LATENCY.labels(batch_size_bucket=_m.batch_bucket(1)).time():
            return self._embed_raw(text)

    def _embed_raw(self, text: str) -> torch.Tensor:
        """Raw embed without metric instrumentation; used inside batch paths
        that already own the ``EMBED_LATENCY`` observation so we don't
        double-count."""
        if self._custom_embed_fn is not None:
            vec = self._custom_embed_fn(text)
            return vec.detach().to(self._device)
        assert self._encoder is not None
        with torch.no_grad():
            stacked = self._encoder.encode_batch(text)  # (T, embed_dim)
        if stacked.numel() == 0:
            return torch.zeros(self._embed_dim, device=self._device)
        pooled = stacked.mean(dim=0)
        return pooled.detach().to(self._device)

    def _embed_batch(self, texts: list[str]) -> list[torch.Tensor]:
        """Embed a list of texts, using a fused path if the backend has one."""
        bucket = _m.batch_bucket(len(texts))
        with _m.EMBED_LATENCY.labels(batch_size_bucket=bucket).time():
            fn = self._custom_embed_fn
            batch_fn = getattr(fn, "encode_many", None) if fn is not None else None
            if batch_fn is not None:
                stacked = batch_fn(texts)  # (N, embed_dim) tensor
                return [row.detach().to(self._device) for row in stacked]
            return [self._embed_raw(t) for t in texts]

    def _retrieve_with_rerank(
        self,
        query_text: str,
        query_vec: torch.Tensor,
        *,
        k: int,
        oversample: int = 3,
    ) -> list[MemoryHit]:
        """Two-stage retrieval: cosine candidates → graph-score re-rank."""
        from soma.training.verbalizer_bootstrap import text_to_state

        q_np = _vec_to_np(query_vec).reshape(-1)
        pool_k = min(k * oversample, len(self._ids))
        pairs = self._backend.search(q_np, k=pool_k)
        candidates = [self._hit_for_id(nid, score=s) for nid, s in pairs]
        if not candidates:
            return []

        soma_output_dim = int(self._soma.config.sensor_output_dim)
        q_act = text_to_state(
            text=query_text,
            soma=self._soma,
            tokenizer=self._soma_tokenizer,
            encoder=self._soma_encoder,
            soma_output_dim=soma_output_dim,
        )

        alpha = self._graph_rerank_alpha
        scored: list[tuple[float, MemoryHit]] = []
        for hit in candidates:
            stored_act = self._soma_activations.get(hit.node_id)
            if stored_act is not None and q_act is not None:
                graph_score = float(
                    F.cosine_similarity(
                        q_act.view(1, -1).cpu(),
                        stored_act.view(1, -1).cpu(),
                        dim=-1,
                    ).item()
                )
                blended = (1.0 - alpha) * hit.score + alpha * graph_score
            else:
                blended = hit.score
            scored.append(
                (
                    blended,
                    MemoryHit(
                        node_id=hit.node_id,
                        text=hit.text,
                        score=blended,
                        metadata=hit.metadata,
                        timestamp_step=hit.timestamp_step,
                    ),
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in scored[:k]]

    def _rank(
        self,
        query_vec: torch.Tensor,
        *,
        k: int,
        exclude_idx: int | None,
    ) -> list[MemoryHit]:
        """Delegate ranking to the backend.

        ``exclude_idx`` is kept as a parameter for call-site back-compat
        (older ``related``/``_retrieve_with_rerank`` passed it); we
        translate it to the id set the backend expects. New call sites
        should prefer passing ``exclude_idx=None`` and letting the
        caller filter if needed.
        """
        exclude_ids: set[str] | None = None
        if exclude_idx is not None:
            exclude_ids = {self._ids[exclude_idx]}
        q_np = _vec_to_np(query_vec).reshape(-1)
        pairs = self._backend.search(q_np, k=k, exclude_ids=exclude_ids)
        return [self._hit_for_id(nid, score=s) for nid, s in pairs]

    def _retrieve_with_filter(
        self,
        query: str,
        q_vec: torch.Tensor,
        *,
        where: dict[str, Any],
        k: int,
        hybrid_alpha: float | None,
    ) -> list[MemoryHit]:
        """Run a ``where``-filtered retrieve.

        Attempts filter pushdown first when the backend declares
        support; on :class:`FilterPushdownUnsupported` falls back to
        the Python pre-filter path that every backend can do.
        """
        if self._backend.supports_filter_pushdown and hybrid_alpha is None:
            try:
                q_np = _vec_to_np(q_vec).reshape(-1)
                pairs = self._backend.search(q_np, k=k, where=where)
                return [self._hit_for_id(nid, score=s) for nid, s in pairs]
            except FilterPushdownUnsupported:
                pass  # fall through to Python pre-filter
        # Python pre-filter path: restrict to entries matching ``where``
        # then brute-force cosine (and optionally BM25) over the subset.
        filter_idx = [
            i
            for i, meta in enumerate(self._metadatas)
            if _matches_where(meta, where)
        ]
        if not filter_idx:
            return []
        candidates = self._rank_subset(q_vec, filter_idx, k=k)
        if hybrid_alpha is not None:
            candidates = self._blend_bm25_subset(
                query, candidates, filter_idx, alpha=hybrid_alpha, k=k
            )
        return candidates

    def _hit_for_index(self, idx: int, *, score: float) -> MemoryHit:
        return MemoryHit(
            node_id=self._ids[idx],
            text=self._texts[idx],
            score=score,
            metadata=dict(self._metadatas[idx]),
            timestamp_step=self._timestamps[idx],
        )

    def _hit_for_id(self, node_id: str, *, score: float) -> MemoryHit:
        idx = self._id_to_idx[node_id]
        return MemoryHit(
            node_id=node_id,
            text=self._texts[idx],
            score=float(score),
            metadata=dict(self._metadatas[idx]),
            timestamp_step=self._timestamps[idx],
        )
