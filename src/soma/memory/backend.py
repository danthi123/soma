"""VectorBackend protocol — the pluggable query-accelerator surface.

MemoryLayer keeps ids/texts/metadata/timestamps/WAL in Python and
delegates every vector op to a ``VectorBackend``. Backends may live
in-process (see :mod:`soma.memory.backends.inproc`), connect to a
remote server (see :mod:`soma.memory.backends.qdrant`), or wrap any
other store that supports approximate nearest-neighbour search.

Design choices pinned by this module:

- **Numpy at the boundary.** Vectors enter/exit as ``np.ndarray``
  ``float32`` — never torch tensors. This lets adapters that speak
  arrow/C/HTTP stay out of the torch import graph and keeps the
  contract dtype-explicit. MemoryLayer is responsible for the
  ``.detach().cpu().numpy().astype(np.float32)`` conversion.

- **Filter pushdown is opt-in.** Backends that can translate the
  Chroma-compatible ``where`` dict into their native filter language
  set ``supports_filter_pushdown=True`` and accept ``where=`` in
  :meth:`VectorBackend.search`. Unsupported ops should raise
  :class:`FilterPushdownUnsupported` so MemoryLayer can cleanly fall
  back to its Python pre-filter + ``search_subset`` path.

- **Runtime-checkable.** ``isinstance(obj, VectorBackend)`` works on
  any duck-typed object that exposes the right attributes. That way
  a user can hand MemoryLayer a third-party adapter without importing
  from this module at construction time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


class FilterPushdownUnsupported(Exception):
    """Raised by a backend when a ``where`` clause can't be translated.

    MemoryLayer catches this and falls back to its Python pre-filter
    path (``_matches_where`` + ``search_subset``). Carries the offending
    ``op`` and ``field`` so logs/operators can see which clause caused
    the refusal without having to re-parse the filter.
    """

    def __init__(
        self,
        *,
        op: str | None = None,
        field: str | None = None,
        message: str | None = None,
    ) -> None:
        self.op = op
        self.field = field
        detail = message or (
            f"filter pushdown not supported for op={op!r} field={field!r}"
        )
        super().__init__(detail)


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol every vector-store adapter implements.

    All vectors are ``np.ndarray`` ``float32``. Shapes:

    - ``add`` / ``get_vectors`` take ``(N, dim)``.
    - ``search`` / ``search_subset`` take ``(dim,)`` queries.

    Identifiers are opaque strings (MemoryLayer uses ``uuid4().hex``);
    backends translate to native point ids internally and keep the
    mapping private.

    Scores returned by :meth:`search` and :meth:`search_subset` must be
    cosine similarity in ``[-1, 1]`` — higher is more similar.
    """

    @property
    def ntotal(self) -> int:
        """Number of vectors currently indexed."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimension declared at construction time."""
        ...

    @property
    def supports_filter_pushdown(self) -> bool:
        """``True`` iff the backend honours ``search(where=...)``.

        When ``False``, MemoryLayer skips the pushdown attempt and
        does Python pre-filter + ``search_subset`` instead.
        """
        ...

    @property
    def name(self) -> str:
        """Short label for metrics (``"inproc"``, ``"qdrant"``, ...)."""
        ...

    def open(self) -> None:
        """Idempotent — bring the backend to a usable state.

        Called by MemoryLayer after construction and by ``restore``.
        Cheap for in-proc; may open a client connection or collection
        for remote adapters.
        """
        ...

    def close(self) -> None:
        """Release any resources (clients, file handles, collections).

        Safe to call multiple times. MemoryLayer calls this from its
        own :meth:`close`.
        """
        ...

    def clear(self) -> None:
        """Drop every indexed vector. Keeps ``dim`` and any static
        configuration (distance metric, HNSW params, collection name)
        intact so a subsequent ``add`` just works."""
        ...

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Index ``len(ids)`` new vectors. ``vectors`` has shape
        ``(len(ids), dim)`` and dtype ``float32``.

        Repeated ids are undefined behaviour — MemoryLayer never
        re-adds an id (it uses uuid4 per entry).
        """
        ...

    def remove(self, ids: list[str]) -> None:
        """Drop these ids from the index. Missing ids are ignored so
        this is idempotent on replay."""
        ...

    def get_vectors(self, ids: list[str]) -> np.ndarray:
        """Fetch the stored vectors for these ids, in id order.

        Returns ``(len(ids), dim)`` ``float32``. Used by
        :meth:`MemoryLayer.related` to pivot off a stored node's vector.
        Missing ids should raise (MemoryLayer always asks for ids it
        knows exist).
        """
        ...

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude_ids: set[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Return the top-``k`` ``(id, score)`` pairs by cosine similarity.

        - ``query`` shape ``(dim,)`` float32.
        - ``exclude_ids`` omits specific ids (used by ``related`` to
          drop the self-match).
        - ``where`` is the Chroma-style filter dict. Backends that
          declare ``supports_filter_pushdown=True`` should attempt to
          translate and execute the filter server-side, raising
          :class:`FilterPushdownUnsupported` for any operator they
          can't express.
        """
        ...

    def search_subset(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """Rank the cosine match of ``query`` against a caller-chosen
        set of ``candidate_ids``. Used by MemoryLayer when the Python
        pre-filter path is active (backend doesn't push down, or the
        filter is too exotic).
        """
        ...

    def snapshot(self, bundle_dir: Path) -> None:
        """Write any backend-specific state into ``bundle_dir`` so
        ``restore(bundle_dir)`` can rebuild equivalent state.

        For in-proc this is ``memory_embeddings.pt``; for Qdrant local
        it's a ``qdrant.snapshot`` file plus a ``backend.json`` sidecar;
        for Qdrant HTTP it's just the sidecar pointing at the server.
        """
        ...

    def restore(self, bundle_dir: Path) -> None:
        """Inverse of :meth:`snapshot`. MemoryLayer calls this after
        it has replayed the WAL, so an adapter that stores nothing
        on disk (HTTP) can safely no-op; the WAL already replayed the
        upserts."""
        ...


__all__ = ["FilterPushdownUnsupported", "VectorBackend"]
