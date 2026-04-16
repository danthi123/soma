"""Tests for soma.integrations.llamaindex.SomaRetriever."""

from __future__ import annotations

import pytest
import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer

llama_index = pytest.importorskip("llama_index")
from llama_index.core.schema import NodeWithScore  # noqa: E402

from soma.integrations.llamaindex import SomaRetriever  # noqa: E402

CORPUS = [
    "the quick brown fox",
    "a stitch in time saves nine",
    "all happy families are alike",
]


@pytest.fixture
def mem() -> MemoryLayer:
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    m = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for text in CORPUS:
        m.store(text)
    return m


def test_retriever_returns_nodes(mem: MemoryLayer) -> None:
    retriever = SomaRetriever(memory=mem, k=2)
    nodes = retriever.retrieve("fox")
    assert len(nodes) == 2
    assert all(isinstance(n, NodeWithScore) for n in nodes)


def test_retriever_node_has_text_and_score(mem: MemoryLayer) -> None:
    retriever = SomaRetriever(memory=mem, k=1)
    nodes = retriever.retrieve("stitch")
    assert len(nodes) >= 1
    node = nodes[0]
    assert node.node.text is not None
    assert isinstance(node.score, float)
    assert "node_id" in node.node.metadata


def test_retriever_k_clamps(mem: MemoryLayer) -> None:
    retriever = SomaRetriever(memory=mem, k=100)
    nodes = retriever.retrieve("anything")
    assert len(nodes) == 3


def test_retriever_empty_memory() -> None:
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(["hello"], vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    m = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    retriever = SomaRetriever(memory=m, k=5)
    nodes = retriever.retrieve("anything")
    assert nodes == []
