"""Tests for soma.memory.api.MemoryLayer — the public vector-DB-shaped surface."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryHit, MemoryLayer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "to be or not to be that is the question",
    "the rain in spain falls mainly on the plain",
    "all happy families are alike",
]


@pytest.fixture
def embedder() -> tuple[object, TextEncoder]:
    """Tiny deterministic tokenizer+encoder for fast unit tests."""
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    return tokenizer, encoder


def test_store_returns_unique_node_ids(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    ids = [mem.store(f"fact {i}") for i in range(5)]
    assert len(set(ids)) == 5, "node_ids must be unique"
    assert all(isinstance(i, str) for i in ids), "node_ids must be strings"


def test_store_accepts_metadata(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("user prefers dark mode", metadata={"source": "settings"})
    hit = mem.get(nid)
    assert hit is not None
    assert hit.metadata == {"source": "settings"}


def test_retrieve_returns_most_similar_first(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("the cat sat on the mat")
    mem.store("quantum mechanics describes subatomic physics")
    mem.store("the dog chased the ball")

    hits = mem.retrieve("feline on a rug", k=3)
    assert len(hits) == 3
    assert all(isinstance(h, MemoryHit) for h in hits)
    # Scores must be monotonically non-increasing.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_empty_store_returns_empty_list(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    assert mem.retrieve("anything", k=5) == []


def test_retrieve_k_clamps_to_store_size(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("one")
    mem.store("two")
    hits = mem.retrieve("something", k=10)
    assert len(hits) == 2


def test_get_recent_orders_by_insertion(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for i in range(5):
        mem.store(f"fact {i}")
    recent = mem.get_recent(3)
    assert len(recent) == 3
    assert [h.text for h in recent] == ["fact 4", "fact 3", "fact 2"]


def test_forget_removes_entry(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("ephemeral fact")
    mem.store("persistent fact")
    assert mem.forget(nid) is True
    assert mem.get(nid) is None
    assert len(mem) == 1
    assert mem.retrieve("ephemeral", k=5) != []  # retrieves persistent, not ephemeral
    assert all("ephemeral" not in h.text for h in mem.retrieve("anything", k=5))


def test_forget_unknown_id_returns_false(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("some fact")
    assert mem.forget("nonexistent-uuid") is False


def test_related_excludes_self(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("the cat sat on the mat")
    mem.store("quantum mechanics")
    mem.store("the dog chased the ball")

    neighbours = mem.related(nid, k=5)
    assert all(h.node_id != nid for h in neighbours)
    assert len(neighbours) == 2  # all other entries


def test_related_unknown_id_raises(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    with pytest.raises(KeyError, match="nonexistent"):
        mem.related("nonexistent-uuid")


def test_save_load_round_trip(embedder, tmp_path: Path) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    ids = [mem.store(f"fact {i}", metadata={"idx": i}) for i in range(5)]

    bundle = tmp_path / "mem"
    mem.save(bundle)
    assert (bundle / "memory_index.json").exists()
    assert (bundle / "memory_embeddings.pt").exists()
    assert (bundle / "tokenizer.json").exists()
    assert (bundle / "encoder.pt").exists()

    restored = MemoryLayer.load(bundle)
    assert len(restored) == 5
    for i, nid in enumerate(ids):
        hit = restored.get(nid)
        assert hit is not None
        assert hit.text == f"fact {i}"
        assert hit.metadata == {"idx": i}


def test_save_load_retrieve_matches_pre_save(embedder, tmp_path: Path) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for text in CORPUS:
        mem.store(text)
    query = "tell me about cats and foxes"
    pre_hits = mem.retrieve(query, k=3)

    bundle = tmp_path / "mem"
    mem.save(bundle)
    restored = MemoryLayer.load(bundle)
    post_hits = restored.retrieve(query, k=3)

    assert [h.node_id for h in pre_hits] == [h.node_id for h in post_hits]
    for pre, post in zip(pre_hits, post_hits, strict=True):
        assert pre.text == post.text
        assert pre.score == pytest.approx(post.score, abs=1e-5)


def test_consolidate_is_safe_noop(embedder) -> None:
    """Stage 2 stub: consolidate must be callable but does not mutate index."""
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("before consolidate")
    before = [h.node_id for h in mem.get_recent(10)]
    mem.consolidate()
    after = [h.node_id for h in mem.get_recent(10)]
    assert before == after


def test_store_empty_text_raises(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    with pytest.raises(ValueError, match="empty"):
        mem.store("")


def test_len_reflects_store_forget(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    assert len(mem) == 0
    a = mem.store("one")
    mem.store("two")
    assert len(mem) == 2
    mem.forget(a)
    assert len(mem) == 1


def test_memory_hit_fields(embedder) -> None:
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    nid = mem.store("hello world", metadata={"k": "v"})
    hit = mem.get(nid)
    assert hit is not None
    assert hit.node_id == nid
    assert hit.text == "hello world"
    assert hit.metadata == {"k": "v"}
    assert isinstance(hit.timestamp_step, int)
    assert hit.score == pytest.approx(1.0, abs=1e-5)  # self-retrieval → cosine 1.0
