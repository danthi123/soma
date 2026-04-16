"""Tests for soma.integrations.langchain.SomaRetriever."""

from __future__ import annotations

import pytest
import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer

langchain_core = pytest.importorskip("langchain_core")
from langchain_core.documents import Document  # noqa: E402

from soma.integrations.langchain import SomaRetriever  # noqa: E402

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


def test_retriever_returns_documents(mem: MemoryLayer) -> None:
    retriever = SomaRetriever(memory=mem, k=2)
    docs = retriever.invoke("fox")
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)


def test_retriever_documents_have_metadata(mem: MemoryLayer) -> None:
    retriever = SomaRetriever(memory=mem, k=1)
    docs = retriever.invoke("stitch")
    assert len(docs) >= 1
    doc = docs[0]
    assert "node_id" in doc.metadata
    assert "score" in doc.metadata
    assert isinstance(doc.metadata["score"], float)


def test_retriever_k_clamps(mem: MemoryLayer) -> None:
    retriever = SomaRetriever(memory=mem, k=100)
    docs = retriever.invoke("anything")
    assert len(docs) == 3


def test_retriever_empty_memory() -> None:
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(["hello"], vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    m = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    retriever = SomaRetriever(memory=m, k=5)
    docs = retriever.invoke("anything")
    assert docs == []


def test_retriever_passes_where_to_memory_layer() -> None:
    """SomaRetriever with where= should restrict results to the filter."""
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(["alex bobbi fact pref"], vocab_size=64)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    m = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    m.store("alex fact", metadata={"user": "alex"})
    m.store("bobbi fact", metadata={"user": "bobbi"})
    retriever = SomaRetriever(memory=m, k=5, where={"user": "alex"})
    docs = retriever.invoke("fact")
    assert len(docs) == 1
    assert docs[0].metadata.get("user") == "alex"


def test_retriever_passes_hybrid_alpha() -> None:
    """hybrid_alpha=0 (pure BM25) routes through SOMA's BM25 index."""
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(["portland boston"], vocab_size=64)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    m = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    m.store("alex lives in portland")
    m.store("bobbi lives in boston")
    retriever = SomaRetriever(memory=m, k=1, hybrid_alpha=0.0)
    docs = retriever.invoke("portland")
    assert len(docs) == 1
    assert "portland" in docs[0].page_content
