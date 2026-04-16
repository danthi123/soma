"""Tests for InProcBackend — the default, zero-new-deps VectorBackend.

Runs the same set of round-trips against both FAISS configurations
(``flat`` and ``hnsw``) so the lazy-rebuild semantics stay pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from soma.memory.backend import VectorBackend
from soma.memory.backends.inproc import InProcBackend


def _rand_vecs(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


@pytest.fixture(params=["flat", "hnsw"])
def backend_factory(request: pytest.FixtureRequest):
    """Construct an InProcBackend in the requested FAISS mode.

    faiss_threshold=5 is low enough that any test with more than a
    handful of adds trips the FAISS build path.
    """
    index_type = request.param

    def _make(*, dim: int = 8, threshold: int = 5) -> InProcBackend:
        return InProcBackend(
            dim=dim,
            faiss_threshold=threshold,
            faiss_index_type=index_type,
        )

    return _make


def test_add_then_search_top_k(backend_factory) -> None:
    b = backend_factory()
    ids = [f"id-{i}" for i in range(6)]
    vecs = _rand_vecs(6, 8)
    b.add(ids, vecs)
    assert b.ntotal == 6
    hits = b.search(vecs[0], k=3)
    assert len(hits) == 3
    # Self-match should be at the top under cosine similarity.
    assert hits[0][0] == "id-0"


def test_search_respects_exclude_ids(backend_factory) -> None:
    b = backend_factory()
    ids = [f"id-{i}" for i in range(6)]
    vecs = _rand_vecs(6, 8)
    b.add(ids, vecs)
    hits = b.search(vecs[0], k=5, exclude_ids={"id-0"})
    returned = {nid for nid, _ in hits}
    assert "id-0" not in returned
    assert len(hits) == 5


def test_search_subset_restricts_to_candidates(backend_factory) -> None:
    b = backend_factory()
    ids = [f"id-{i}" for i in range(8)]
    vecs = _rand_vecs(8, 8)
    b.add(ids, vecs)
    subset = ["id-2", "id-5", "id-7"]
    hits = b.search_subset(vecs[0], subset, k=10)
    returned = {nid for nid, _ in hits}
    assert returned <= set(subset)
    assert len(hits) == 3


def test_remove_triggers_rebuild_on_next_search(backend_factory) -> None:
    b = backend_factory()
    ids = [f"id-{i}" for i in range(6)]
    vecs = _rand_vecs(6, 8)
    b.add(ids, vecs)
    # Force FAISS build via first search.
    b.search(vecs[0], k=3)
    b.remove(["id-3"])
    assert b.ntotal == 5
    hits = b.search(vecs[0], k=5)
    returned = {nid for nid, _ in hits}
    assert "id-3" not in returned


def test_get_vectors_roundtrips_stored_vectors(backend_factory) -> None:
    b = backend_factory()
    ids = [f"id-{i}" for i in range(4)]
    vecs = _rand_vecs(4, 8)
    b.add(ids, vecs)
    got = b.get_vectors(["id-2", "id-0"])
    assert got.shape == (2, 8)
    np.testing.assert_allclose(got[0], vecs[2])
    np.testing.assert_allclose(got[1], vecs[0])


def test_snapshot_restore_roundtrip_preserves_ntotal(
    backend_factory, tmp_path
) -> None:
    b = backend_factory()
    ids = [f"id-{i}" for i in range(6)]
    vecs = _rand_vecs(6, 8)
    b.add(ids, vecs)
    b.snapshot(tmp_path)

    b2 = backend_factory()
    b2.restore(tmp_path)
    # Restore sets up the vector store from the snapshot; subsequent
    # search sees the same ntotal.
    assert b2.ntotal == 6
    got = b2.get_vectors(["id-3"])
    np.testing.assert_allclose(got[0], vecs[3])


def test_clear_empties_store(backend_factory) -> None:
    b = backend_factory()
    ids = [f"id-{i}" for i in range(3)]
    b.add(ids, _rand_vecs(3, 8))
    b.clear()
    assert b.ntotal == 0


def test_supports_filter_pushdown_false(backend_factory) -> None:
    assert backend_factory().supports_filter_pushdown is False


def test_name_is_inproc(backend_factory) -> None:
    assert backend_factory().name == "inproc"


def test_protocol_isinstance(backend_factory) -> None:
    assert isinstance(backend_factory(), VectorBackend)
