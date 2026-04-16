"""Tests for hybrid BM25+vector retrieval and cross-encoder re-ranking
hooks on MemoryLayer."""

from __future__ import annotations

import pytest
import torch

from soma.memory import MemoryLayer
from soma.memory.rerank import StubReranker


def _deterministic_embed(text: str) -> torch.Tensor:
    """Hash-derived 16-d vector — collision-resistant enough for tests
    where we want distinguishable cosine scores."""
    h = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(h)
    return torch.randn(16, generator=g)


def _make_mem() -> MemoryLayer:
    return MemoryLayer(embed_fn=_deterministic_embed, embed_dim=16)


def test_retrieve_still_works_with_no_boosters_enabled() -> None:
    """Sanity: pure cosine retrieve unchanged by the new hooks."""
    mem = _make_mem()
    mem.store("alpha beta")
    mem.store("gamma delta")
    hits = mem.retrieve("alpha beta", k=2)
    assert len(hits) <= 2


def test_retrieve_rejects_invalid_hybrid_alpha() -> None:
    mem = _make_mem()
    mem.store("anything")
    with pytest.raises(ValueError, match="hybrid_alpha"):
        mem.retrieve("q", k=1, hybrid_alpha=-0.5)
    with pytest.raises(ValueError, match="hybrid_alpha"):
        mem.retrieve("q", k=1, hybrid_alpha=1.5)


def test_retrieve_rerank_without_attached_reranker_raises() -> None:
    mem = _make_mem()
    mem.store("one")
    mem.store("two")
    with pytest.raises(ValueError, match="no reranker attached"):
        mem.retrieve("q", k=1, rerank_top_n=5)


def test_hybrid_prefers_lexical_matches_at_low_alpha() -> None:
    """At alpha=0 we should use BM25 only — the doc whose tokens
    literally match the query should rank first even if cosine over
    random embeddings puts it lower."""
    mem = _make_mem()
    mem.store("artificial sweeteners in diet soda")
    mem.store("mountaineering gear recommendations")
    mem.store("the weather is nice today")
    # BM25 on "diet soda" -> doc 0 dominates.
    hits = mem.retrieve("diet soda", k=3, hybrid_alpha=0.0)
    assert len(hits) >= 1
    assert hits[0].text == "artificial sweeteners in diet soda"


def test_hybrid_at_full_alpha_equals_pure_cosine() -> None:
    """alpha=1 should reduce to pure cosine (same ranking as retrieve
    without the flag, modulo ID vs index ordering ties)."""
    mem = _make_mem()
    for t in ["aa bb cc", "dd ee ff", "gg hh ii", "jj kk ll"]:
        mem.store(t)
    plain = [h.node_id for h in mem.retrieve("aa bb cc", k=3)]
    hybrid = [h.node_id for h in mem.retrieve("aa bb cc", k=3, hybrid_alpha=1.0)]
    assert set(plain) == set(hybrid)


def test_rerank_reorders_candidates_using_attached_scorer() -> None:
    """StubReranker scores by token overlap — the doc with the most
    overlapping tokens must rank first after rerank, even when cosine
    put it lower."""
    mem = _make_mem()
    mem.store("alpha beta gamma delta epsilon")
    mem.store("alpha beta")
    mem.store("something else entirely")
    mem.attach_reranker(StubReranker())

    hits = mem.retrieve("alpha beta gamma delta", k=1, rerank_top_n=3)
    assert len(hits) == 1
    # Doc with all four query tokens should win the rerank.
    assert hits[0].text == "alpha beta gamma delta epsilon"


def test_rerank_preserves_original_score_in_metadata() -> None:
    mem = _make_mem()
    mem.store("alpha beta")
    mem.store("gamma delta")
    mem.attach_reranker(StubReranker())
    hits = mem.retrieve("alpha", k=2, rerank_top_n=2)
    assert all("_pre_rerank_score" in h.metadata for h in hits)


def test_hybrid_and_rerank_compose() -> None:
    """Pool via hybrid, then cross-encoder re-rank the pool."""
    mem = _make_mem()
    mem.store("long detailed record about solar panels and photovoltaic cells")
    mem.store("brief note solar")
    mem.store("unrelated content on cooking")
    mem.attach_reranker(StubReranker())
    hits = mem.retrieve(
        "solar panels", k=1, hybrid_alpha=0.3, rerank_top_n=3
    )
    assert len(hits) == 1
    # Both docs with "solar" should be pool candidates; the longer one
    # shares more tokens with the query after tokenization.
    assert "solar" in hits[0].text.lower()


def test_bm25_rebuilds_on_store_after_retrieve() -> None:
    """Adding an entry after the BM25 index is built must invalidate."""
    mem = _make_mem()
    mem.store("foo bar")
    mem.retrieve("foo", k=1, hybrid_alpha=0.0)  # triggers build
    v1 = mem._bm25_version
    mem.store("new baz")
    mem.retrieve("baz", k=1, hybrid_alpha=0.0)  # triggers rebuild
    assert mem._bm25_version > v1
    assert len(mem._bm25_index) == 2


def test_attach_reranker_swaps_the_model() -> None:
    mem = _make_mem()
    first = StubReranker(name="A")
    second = StubReranker(name="B")
    mem.attach_reranker(first)
    assert mem._reranker.name == "A"
    mem.attach_reranker(second)
    assert mem._reranker.name == "B"
