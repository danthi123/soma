"""Unit tests for the VectorBackend protocol + FilterPushdownUnsupported.

These tests pin the contract that every backend adapter must satisfy.
They don't construct a real backend — Task 2 does that — they just
verify the Protocol is importable and runtime-checkable and the
sentinel exception carries the metadata callers need.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from soma.memory.backend import FilterPushdownUnsupported, VectorBackend


class _DummyBackend:
    """Minimal stub that should pass ``isinstance(..., VectorBackend)``.

    Protocol checks at runtime look only at attribute presence, not
    signature compatibility, so having each method defined is enough.
    """

    name = "dummy"
    supports_filter_pushdown = False

    @property
    def ntotal(self) -> int:
        return 0

    @property
    def dim(self) -> int:
        return 4

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def clear(self) -> None:
        return None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        return None

    def remove(self, ids: list[str]) -> None:
        return None

    def get_vectors(self, ids: list[str]) -> np.ndarray:
        return np.zeros((len(ids), self.dim), dtype=np.float32)

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude_ids: set[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        return []

    def search_subset(
        self, query: np.ndarray, candidate_ids: list[str], k: int
    ) -> list[tuple[str, float]]:
        return []

    def snapshot(self, bundle_dir: Path) -> None:
        return None

    def restore(self, bundle_dir: Path) -> None:
        return None


def test_protocol_is_runtime_checkable() -> None:
    """Protocol must be marked ``@runtime_checkable`` so ``isinstance``
    works — that's how MemoryLayer will accept user-supplied adapters."""
    assert isinstance(_DummyBackend(), VectorBackend)


def test_non_conforming_object_fails_isinstance() -> None:
    """A plain object without the required attributes should not pass."""

    class _Empty:
        pass

    assert not isinstance(_Empty(), VectorBackend)


def test_filter_pushdown_unsupported_carries_fields() -> None:
    """Exception records which op and field caused the refusal so
    MemoryLayer (and logs) can surface the mismatch."""
    err = FilterPushdownUnsupported(op="$regex", field="tag")
    assert err.op == "$regex"
    assert err.field == "tag"
    assert isinstance(err, Exception)


def test_filter_pushdown_unsupported_allows_empty_construction() -> None:
    """Both fields are optional — caller may know only one of them."""
    err = FilterPushdownUnsupported()
    assert err.op is None
    assert err.field is None


def test_filter_pushdown_unsupported_is_catchable() -> None:
    with pytest.raises(FilterPushdownUnsupported) as exc_info:
        raise FilterPushdownUnsupported(op="$foo")
    assert exc_info.value.op == "$foo"
