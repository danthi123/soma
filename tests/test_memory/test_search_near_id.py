"""Adapter-specific tests for ``VectorBackend.search_near_id``.

The parametrized contract suite in ``test_backend_protocol.py`` pins
the behaviour every adapter must satisfy. This module adds
adapter-specific *implementation* tests — specifically, that
``QdrantBackend`` takes the ``recommend`` fast path instead of the
default ``get_vectors`` + ``search`` pair.

Philosophy: the contract tests verify the public observable
behaviour; these tests verify we actually skipped the round-trip we
claimed to skip. Without them a regression that silently reverts to
the default would ship green CI.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def qdrant_backend():
    """Fresh in-memory QdrantBackend with 5 seeded vectors."""
    pytest.importorskip("qdrant_client")
    from soma.memory.backends.qdrant import QdrantBackend

    b = QdrantBackend(mode="memory", dim=8)
    rng = np.random.default_rng(0)
    ids = [f"id-{i}" for i in range(5)]
    vecs = rng.standard_normal((5, 8)).astype(np.float32)
    b.add(ids, vecs)
    try:
        yield b, ids
    finally:
        b.close()


def test_qdrant_search_near_id_calls_recommend(qdrant_backend) -> None:
    """Fast path: ``search_near_id`` MUST go through ``client.recommend``.

    Guards against a regression where a future edit reverts the
    override to call the default helper — which would silently double
    the HTTP round-trips for ``MemoryLayer.related()``.
    """
    b, ids = qdrant_backend
    assert b._client is not None
    real_recommend = b._client.recommend
    with patch.object(
        b._client, "recommend", wraps=real_recommend
    ) as mock_rec:
        _ = b.search_near_id(ids[0], k=3)
    assert mock_rec.called, "search_near_id must call client.recommend"
    # Positive should be the int point id for the pivot — proves we're
    # using the server-side id-based recommend rather than re-uploading
    # a vector.
    call_kwargs = mock_rec.call_args.kwargs
    pos = call_kwargs.get("positive")
    assert pos is not None and len(pos) == 1
    assert pos[0] == b._id_to_point[ids[0]]


def test_qdrant_search_near_id_does_not_call_get_vectors(
    qdrant_backend,
) -> None:
    """Fast path: ``search_near_id`` MUST NOT fall through
    ``get_vectors`` + ``search``.

    This is the load-bearing claim of Phase 16 for HTTP Qdrant: one
    round-trip, not two. The override hits ``client.recommend`` and
    nothing else touches the retrieve/search API surface.
    """
    b, ids = qdrant_backend
    assert b._client is not None
    with (
        patch.object(
            b._client, "retrieve", wraps=b._client.retrieve
        ) as mock_retrieve,
        patch.object(
            b._client, "search", wraps=b._client.search
        ) as mock_search,
    ):
        _ = b.search_near_id(ids[0], k=3)
    assert not mock_retrieve.called, (
        "search_near_id must not call client.retrieve — that's the "
        "slow path Phase 16 skipped."
    )
    assert not mock_search.called, (
        "search_near_id must not call client.search — recommend "
        "replaces it."
    )


def test_qdrant_search_near_id_handles_missing_point(qdrant_backend) -> None:
    """Unknown pivot ids return ``[]`` without touching ``recommend``.

    The id-map lookup catches the miss before we'd waste an HTTP call
    on a guaranteed-404 ``recommend``.
    """
    b, _ = qdrant_backend
    assert b._client is not None
    with patch.object(b._client, "recommend") as mock_rec:
        assert b.search_near_id("definitely-not-here", k=3) == []
    assert not mock_rec.called


def test_qdrant_search_near_id_include_self_injects_pivot(
    qdrant_backend,
) -> None:
    """``exclude_self=False`` must surface the pivot id at the top.

    Qdrant's ``recommend`` API never returns the positive points
    themselves, so the override synthesizes a self-hit at score 1.0
    and requests ``k - 1`` true recommendations.
    """
    b, ids = qdrant_backend
    hits = b.search_near_id(ids[0], k=3, exclude_self=False)
    assert len(hits) == 3
    assert hits[0][0] == ids[0]
    assert hits[0][1] == pytest.approx(1.0)
    # The remaining slots are real neighbours — none of them the pivot.
    assert all(nid != ids[0] for nid, _ in hits[1:])
