"""Sparse k-hot codes for biology-inspired retrieval — Path B Phase 2.

Design reference:
    docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md § Phase 2

Primitives:

- :class:`SparseCode` — immutable (active_indices, dim) pair, sorted and
  validated on construction. Hashable, serialises cleanly to JSON via
  ``active.tolist()``.
- :func:`kwta` — project a dense embedding into ``dim`` with a
  seeded random-normal projection, then pick the top-``k`` as active.
- :func:`pattern_separate` — reduce pairwise overlap between codes by
  removing a fraction of shared active dimensions. Strength 0 is an
  identity; strength 1 removes every shared dimension.
- :func:`code_similarity` — Jaccard similarity in [0, 1].

These are numpy-only, device-free, and allocation-light — they sit on top
of the existing SOMA dense embedding path and add a third retrieval
score (sparse-overlap) alongside cosine and BM25.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SparseCode:
    """Immutable sparse k-hot code.

    Attributes:
        active: Sorted int32 array of active dimension indices. Must be
            strictly increasing (no duplicates) and all in ``[0, dim)``.
        dim: Ambient dimensionality of the code space.
    """
    active: np.ndarray
    dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.active, np.ndarray):
            raise TypeError(f"active must be np.ndarray, got {type(self.active).__name__}")
        if self.active.ndim != 1:
            raise ValueError(f"active must be 1-D, got shape {self.active.shape}")
        if self.active.size > 0:
            if self.active.min() < 0 or self.active.max() >= self.dim:
                raise ValueError(
                    f"active indices out of range [0, {self.dim}): "
                    f"min={int(self.active.min())}, max={int(self.active.max())}"
                )
            diffs = np.diff(self.active)
            if diffs.size > 0 and diffs.min() <= 0:
                raise ValueError("active indices must be sorted and unique")

    @property
    def k(self) -> int:
        """Number of active dimensions."""
        return int(self.active.size)


def _make_projection(rng: np.random.Generator, dim: int, in_features: int) -> np.ndarray:
    """Random-normal projection matrix, shape (dim, in_features)."""
    return rng.standard_normal((dim, in_features)).astype(np.float32)


def kwta(
    dense: np.ndarray,
    k: int,
    dim: int,
    *,
    projection: np.ndarray | None = None,
    seed: int = 0,
) -> SparseCode:
    """k-Winner-Take-All sparse code from a dense embedding.

    Args:
        dense: 1-D dense embedding, shape ``(in_features,)``.
        k: Number of dimensions to activate.
        dim: Ambient code dimensionality (size of the sparse code space).
        projection: Optional pre-computed projection matrix of shape
            ``(dim, in_features)``. If provided, ``seed`` is ignored. If
            ``None``, a random-normal projection is generated from
            ``seed``.
        seed: Seed for the projection RNG when ``projection`` is None.

    Returns:
        :class:`SparseCode` with exactly ``min(k, dim)`` active indices,
        chosen as the top-k projected activations.
    """
    if dense.ndim != 1:
        raise ValueError(f"dense must be 1-D, got shape {dense.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    if projection is None:
        rng = np.random.default_rng(seed)
        projection = _make_projection(rng, dim, dense.size)
    else:
        if projection.shape != (dim, dense.size):
            raise ValueError(
                f"projection shape {projection.shape} does not match (dim, in_features) "
                f"= ({dim}, {dense.size})"
            )

    activations = projection @ dense.astype(np.float32)  # shape (dim,)
    k_eff = min(k, dim)
    # argpartition gives indices of the k largest (unsorted). We then sort
    # ascending for the SparseCode invariant.
    top_idx = np.argpartition(-activations, k_eff - 1)[:k_eff]
    top_idx = np.sort(top_idx).astype(np.int32)
    return SparseCode(active=top_idx, dim=dim)


def code_similarity(a: SparseCode, b: SparseCode) -> float:
    """Jaccard similarity between two sparse codes.

    Returns 1.0 if both codes are empty (vacuous agreement).
    """
    if a.active.size == 0 and b.active.size == 0:
        return 1.0
    intersection = np.intersect1d(a.active, b.active, assume_unique=True).size
    union = a.active.size + b.active.size - intersection
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def pattern_separate(
    codes: Sequence[SparseCode],
    strength: float = 0.3,
) -> list[SparseCode]:
    """Reduce pairwise overlap between sparse codes (DG-like separation).

    For each pair ``(i, j)`` with ``i < j``, removes
    ``floor(strength * |shared|)`` shared active dimensions from the
    lower-index code. Deterministic (no RNG). The strategy is
    asymmetric but monotone: larger ``strength`` can only decrease
    pairwise overlap, and ``strength == 0`` is an identity.

    Args:
        codes: Sequence of SparseCodes to separate. All must share the
            same ``dim``.
        strength: Fraction of shared dimensions to remove per pair, in
            ``[0, 1]``.

    Returns:
        New list of SparseCodes. Input codes are not mutated.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if len(codes) < 2 or strength == 0.0:
        return [SparseCode(active=c.active.copy(), dim=c.dim) for c in codes]

    dims = {c.dim for c in codes}
    if len(dims) > 1:
        raise ValueError(f"all codes must share the same dim, got {dims}")

    working = [c.active.copy() for c in codes]

    for i in range(len(working)):
        for j in range(i + 1, len(working)):
            shared = np.intersect1d(working[i], working[j], assume_unique=True)
            n_remove = int(np.floor(strength * shared.size))
            if n_remove == 0:
                continue
            to_remove = shared[:n_remove]
            working[i] = np.setdiff1d(working[i], to_remove, assume_unique=True)

    dim = codes[0].dim
    return [SparseCode(active=a.astype(np.int32), dim=dim) for a in working]
