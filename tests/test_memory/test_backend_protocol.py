"""Unit tests for the VectorBackend protocol + FilterPushdownUnsupported.

These tests pin the contract that every backend adapter must satisfy.
They don't construct a real backend unit-by-unit — the dedicated
adapter test modules do that — they just verify the Protocol is
importable and runtime-checkable, the sentinel exception carries the
metadata callers need, AND every shipped adapter round-trips the same
basic invariants (the parametrized contract suite).
"""

from __future__ import annotations

from collections.abc import Callable
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


# ----------------------------------------------------------------------
# Parametrized contract suite — every shipped adapter must pass the same
# basic invariants. Adding a new backend means adding a factory here.
# ----------------------------------------------------------------------

BackendFactory = Callable[[int], Any]


def _inproc_factory(dim: int) -> Any:
    from soma.memory.backends.inproc import InProcBackend

    return InProcBackend(dim=dim, faiss_threshold=3)


def _qdrant_factory(dim: int) -> Any:
    pytest.importorskip("qdrant_client")
    from soma.memory.backends.qdrant import QdrantBackend

    return QdrantBackend(mode="memory", dim=dim)


def _lancedb_factory(dim: int, tmp_path: Path) -> Any:
    pytest.importorskip("lancedb")
    from soma.memory.backends.lancedb import LanceDBBackend

    return LanceDBBackend(
        path=tmp_path / f"lancedb_{dim}",
        table_name="contract",
        dim=dim,
    )


@pytest.fixture(
    params=[
        pytest.param("inproc", id="inproc"),
        pytest.param("qdrant", id="qdrant"),
        pytest.param("lancedb", id="lancedb"),
    ]
)
def shipped_backend(request: pytest.FixtureRequest, tmp_path: Path) -> BackendFactory:
    """Factory returning a freshly-constructed backend for each shipped
    adapter. Dependencies missing = skip, not fail.
    """
    kind = request.param
    if kind == "inproc":
        return _inproc_factory
    if kind == "qdrant":
        return lambda d: _qdrant_factory(d)
    if kind == "lancedb":
        return lambda d: _lancedb_factory(d, tmp_path)
    raise AssertionError(f"unhandled backend param {kind!r}")


def _rand_vecs(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_contract_initial_state(shipped_backend: BackendFactory) -> None:
    """Fresh backend reports zero ntotal + correct dim."""
    b = shipped_backend(8)
    try:
        assert b.ntotal == 0
        assert b.dim == 8
        assert isinstance(b.name, str) and b.name
    finally:
        if hasattr(b, "close"):
            b.close()


def test_contract_add_then_search(shipped_backend: BackendFactory) -> None:
    b = shipped_backend(8)
    try:
        ids = [f"id-{i}" for i in range(6)]
        vecs = _rand_vecs(6, 8)
        b.add(ids, vecs)
        assert b.ntotal == 6
        hits = b.search(vecs[0], k=3)
        assert len(hits) == 3
        # Self-match is the top cosine hit regardless of adapter.
        assert hits[0][0] == "id-0"
    finally:
        if hasattr(b, "close"):
            b.close()


def test_contract_get_vectors_roundtrip(
    shipped_backend: BackendFactory,
) -> None:
    b = shipped_backend(8)
    try:
        ids = [f"id-{i}" for i in range(4)]
        vecs = _rand_vecs(4, 8, seed=17)
        b.add(ids, vecs)
        got = b.get_vectors(["id-2", "id-0"])
        assert got.shape == (2, 8)
        # Qdrant COSINE renormalizes vectors; compare via cosine sim
        # to stay adapter-agnostic.
        for i, nid in enumerate(["id-2", "id-0"]):
            original = vecs[int(nid.split("-")[1])]
            dot = float(np.dot(got[i], original))
            norms = float(np.linalg.norm(got[i]) * np.linalg.norm(original))
            sim = dot / (norms + 1e-12)
            assert sim > 0.99, f"{nid}: cosine sim {sim}"
    finally:
        if hasattr(b, "close"):
            b.close()


def test_contract_remove_idempotent(
    shipped_backend: BackendFactory,
) -> None:
    b = shipped_backend(8)
    try:
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8)
        b.add(ids, vecs)
        b.remove(["id-2"])
        assert b.ntotal == 4
        hits = b.search(vecs[0], k=5)
        returned = {nid for nid, _ in hits}
        assert "id-2" not in returned
        # Removing an unknown id must not raise.
        b.remove(["does-not-exist"])
        assert b.ntotal == 4
    finally:
        if hasattr(b, "close"):
            b.close()


def test_contract_clear_empties(shipped_backend: BackendFactory) -> None:
    b = shipped_backend(8)
    try:
        b.add(["a", "b", "c"], _rand_vecs(3, 8))
        assert b.ntotal == 3
        b.clear()
        assert b.ntotal == 0
    finally:
        if hasattr(b, "close"):
            b.close()


def test_contract_snapshot_restore_roundtrip(
    shipped_backend: BackendFactory, tmp_path: Path
) -> None:
    b = shipped_backend(8)
    try:
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8, seed=31)
        b.add(ids, vecs)
        snap_dir = tmp_path / "snap"
        snap_dir.mkdir()
        b.snapshot(snap_dir)
    finally:
        if hasattr(b, "close"):
            b.close()

    b2 = shipped_backend(8)
    try:
        b2.restore(snap_dir)
        # After restore, every backend should be usable — either it
        # repopulates from the snapshot on disk (InProc, LanceDB local
        # dir) or it was already pointing at the authoritative store
        # and the WAL-replay path in MemoryLayer brings it up
        # (Qdrant HTTP). The contract: ``search`` works without error.
        _ = b2.search(vecs[0], k=3)
    finally:
        if hasattr(b2, "close"):
            b2.close()


def test_contract_protocol_isinstance(
    shipped_backend: BackendFactory,
) -> None:
    b = shipped_backend(4)
    try:
        assert isinstance(b, VectorBackend)
    finally:
        if hasattr(b, "close"):
            b.close()
