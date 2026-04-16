"""Tests for in-index metadata filtering on MemoryLayer.retrieve(where=...).

Mirrors the Chroma `where` semantics subset most callers actually use:
exact match, AND across fields, $in/$nin, and numeric comparisons."""

from __future__ import annotations

import pytest
import torch

from soma.memory import MemoryLayer
from soma.memory.api import _matches_where
from soma.memory.rerank import StubReranker


def _deterministic_embed(text: str) -> torch.Tensor:
    h = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(h)
    return torch.randn(16, generator=g)


def _make_mem() -> MemoryLayer:
    mem = MemoryLayer(embed_fn=_deterministic_embed, embed_dim=16)
    mem.store("alex loves tea", metadata={"user": "alex", "tag": "pref", "rating": 5})
    mem.store("bobbi prefers coffee", metadata={"user": "bobbi", "tag": "pref", "rating": 3})
    mem.store("alex lives in portland", metadata={"user": "alex", "tag": "fact", "rating": 4})
    mem.store("bobbi lives in boston", metadata={"user": "bobbi", "tag": "fact", "rating": 4})
    mem.store("orphan entry with no user")
    return mem


# ------------------------------------------------------------------
# _matches_where unit tests
# ------------------------------------------------------------------


def test_matches_where_exact_match_true() -> None:
    assert _matches_where({"k": "v"}, {"k": "v"})
    assert _matches_where({"a": 1, "b": 2}, {"a": 1})


def test_matches_where_exact_match_false() -> None:
    assert not _matches_where({"k": "v"}, {"k": "x"})
    assert not _matches_where({}, {"k": "v"})  # missing field fails


def test_matches_where_multi_field_is_and() -> None:
    assert _matches_where({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert not _matches_where({"a": 1, "b": 2}, {"a": 1, "b": 3})


def test_matches_where_eq_operator() -> None:
    assert _matches_where({"k": 5}, {"k": {"$eq": 5}})
    assert not _matches_where({"k": 5}, {"k": {"$eq": 6}})


def test_matches_where_ne_operator() -> None:
    assert _matches_where({"k": 5}, {"k": {"$ne": 6}})
    assert not _matches_where({"k": 5}, {"k": {"$ne": 5}})


def test_matches_where_numeric_comparisons() -> None:
    meta = {"rating": 4}
    assert _matches_where(meta, {"rating": {"$gt": 3}})
    assert not _matches_where(meta, {"rating": {"$gt": 4}})
    assert _matches_where(meta, {"rating": {"$gte": 4}})
    assert _matches_where(meta, {"rating": {"$lt": 5}})
    assert _matches_where(meta, {"rating": {"$lte": 4}})


def test_matches_where_in_operator() -> None:
    assert _matches_where({"k": "a"}, {"k": {"$in": ["a", "b", "c"]}})
    assert not _matches_where({"k": "z"}, {"k": {"$in": ["a", "b", "c"]}})


def test_matches_where_nin_operator() -> None:
    assert _matches_where({"k": "z"}, {"k": {"$nin": ["a", "b"]}})
    assert not _matches_where({"k": "a"}, {"k": {"$nin": ["a", "b"]}})


def test_matches_where_missing_field_with_op_fails() -> None:
    assert not _matches_where({}, {"k": {"$gt": 5}})


def test_matches_where_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        _matches_where({"k": 1}, {"k": {"$regex": ".*"}})


def test_matches_where_in_requires_list() -> None:
    with pytest.raises(ValueError, match="\\$in expects a list"):
        _matches_where({"k": "a"}, {"k": {"$in": "not-a-list"}})


# ------------------------------------------------------------------
# retrieve(where=...) integration tests
# ------------------------------------------------------------------


def test_retrieve_with_where_filters_by_user() -> None:
    mem = _make_mem()
    hits = mem.retrieve("anything", k=10, where={"user": "alex"})
    users = {h.metadata.get("user") for h in hits}
    assert users == {"alex"}
    assert len(hits) == 2


def test_retrieve_with_where_no_matches_returns_empty() -> None:
    mem = _make_mem()
    hits = mem.retrieve("q", k=5, where={"user": "ghost"})
    assert hits == []


def test_retrieve_with_where_combines_multiple_fields_and() -> None:
    mem = _make_mem()
    hits = mem.retrieve("q", k=5, where={"user": "alex", "tag": "fact"})
    assert len(hits) == 1
    assert hits[0].text == "alex lives in portland"


def test_retrieve_with_where_in_operator() -> None:
    mem = _make_mem()
    hits = mem.retrieve("q", k=10, where={"user": {"$in": ["alex", "bobbi"]}})
    assert len(hits) == 4  # orphan entry has no user field, excluded


def test_retrieve_with_where_numeric_gte() -> None:
    mem = _make_mem()
    hits = mem.retrieve("q", k=10, where={"rating": {"$gte": 4}})
    ratings = {h.metadata.get("rating") for h in hits}
    assert ratings == {4, 5}
    assert len(hits) == 3


def test_retrieve_with_where_composes_with_hybrid() -> None:
    """Filter + hybrid BM25: only alex-tagged, lexical rescoring of the
    pre-filtered pool."""
    mem = _make_mem()
    hits = mem.retrieve(
        "tea", k=5, where={"user": "alex"}, hybrid_alpha=0.0
    )
    # BM25-only should surface the alex+tea entry first among alex's entries.
    assert len(hits) == 2
    assert hits[0].text == "alex loves tea"


def test_retrieve_with_where_composes_with_rerank() -> None:
    """Filter narrows the pool, rerank reorders within it."""
    mem = _make_mem()
    mem.attach_reranker(StubReranker())
    hits = mem.retrieve(
        "alex portland", k=1,
        where={"user": "alex"},
        rerank_top_n=5,
    )
    assert len(hits) == 1
    assert hits[0].metadata["user"] == "alex"
    # Token overlap picks the portland entry over the tea entry.
    assert "portland" in hits[0].text


def test_retrieve_with_where_and_hybrid_and_rerank() -> None:
    """Triple compose: filter → hybrid pool → cross-encoder rerank."""
    mem = _make_mem()
    mem.attach_reranker(StubReranker())
    hits = mem.retrieve(
        "fact", k=1,
        where={"tag": "fact"},
        hybrid_alpha=0.3,
        rerank_top_n=5,
    )
    assert len(hits) == 1
    assert hits[0].metadata["tag"] == "fact"


def test_retrieve_with_where_skips_missing_field() -> None:
    """Orphan entry with no user field should be excluded by user filter."""
    mem = _make_mem()
    hits = mem.retrieve("orphan", k=10, where={"user": "alex"})
    texts = [h.text for h in hits]
    assert "orphan entry with no user" not in texts


def test_retrieve_with_where_rejects_unknown_operator_on_query() -> None:
    mem = _make_mem()
    with pytest.raises(ValueError, match="unsupported operator"):
        mem.retrieve("q", k=1, where={"k": {"$regex": ".*"}})
