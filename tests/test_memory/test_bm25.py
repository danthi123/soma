"""Tests for the standalone BM25Index module."""

from __future__ import annotations

from soma.memory.bm25 import BM25Index, tokenize


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("it's 2026 — yes.") == ["it", "s", "2026", "yes"]
    assert tokenize("") == []


def test_bm25_returns_top_k_by_match_strength() -> None:
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "a quick cat chased the brown mouse",
        "elephants are large gray mammals",
        "brown paper packages tied up with string",
    ]
    idx = BM25Index()
    idx.build(docs)
    top = idx.search("quick brown fox", k=3)
    # Document 0 literally contains all three tokens; should rank first.
    assert top[0][0] == 0
    # Docs 1 and 3 share "quick"/"brown" in varying degrees; both should
    # appear before the unrelated elephants doc (2).
    assert 2 not in [i for i, _ in top]


def test_bm25_skips_unknown_query_terms() -> None:
    idx = BM25Index()
    idx.build(["dogs bark loudly", "cats meow softly"])
    # Only "unknown" in query — nothing matches, empty result.
    assert idx.search("unknown", k=5) == []
    # Mixed: the known term still surfaces relevant doc(s).
    top = idx.search("unknown dogs", k=5)
    assert top[0][0] == 0


def test_bm25_drops_zero_score_docs_from_top_k() -> None:
    idx = BM25Index()
    idx.build(["alpha beta", "gamma delta"])
    top = idx.search("alpha", k=5)
    assert len(top) == 1
    assert top[0][0] == 0


def test_bm25_empty_corpus_returns_empty() -> None:
    idx = BM25Index()
    idx.build([])
    assert idx.search("anything", k=5) == []
    assert len(idx) == 0


def test_bm25_len_reflects_corpus_size() -> None:
    idx = BM25Index()
    idx.build(["a", "b", "c"])
    assert len(idx) == 3


def test_bm25_rebuild_replaces_contents() -> None:
    idx = BM25Index()
    idx.build(["first corpus"])
    idx.build(["new content one", "new content two"])
    assert len(idx) == 2
    top = idx.search("content", k=5)
    assert len(top) == 2
