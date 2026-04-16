"""Tests for the RAG session helper."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from soma.llm import DryRunBackend, RAGSession
from soma.llm.rag import format_context
from soma.memory import MemoryLayer
from soma.memory.api import MemoryHit


def _stub_embed(text: str) -> torch.Tensor:
    h = hash(text)
    return torch.tensor([(h >> i) & 0xF for i in range(0, 32, 4)], dtype=torch.float32)


@dataclass
class CapturingBackend:
    name: str = "capturing"
    last_prompt: str = ""
    reply: str = "captured-reply"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.last_prompt = prompt
        return self.reply


def test_format_context_uses_path_and_heading_metadata() -> None:
    hits = [
        MemoryHit(
            node_id="x",
            text="body of chunk one",
            score=0.9,
            metadata={"path": "doc.md", "heading": "Section A"},
        ),
        MemoryHit(
            node_id="y",
            text="body two",
            score=0.5,
            metadata={"source": "chat"},
        ),
    ]
    out = format_context(hits)
    assert "[1] doc.md; heading: Section A" in out
    assert "body of chunk one" in out
    assert "[2] chat" in out


def test_format_context_falls_back_to_node_id_when_no_metadata() -> None:
    hits = [MemoryHit(node_id="abcdef1234", text="body", score=1.0)]
    out = format_context(hits)
    assert "id=abcdef12" in out


def test_rag_session_threads_question_into_prompt() -> None:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    mem.store("Alex lives in Portland", metadata={"path": "alex.md"})
    backend = CapturingBackend()
    chat = RAGSession(memory=mem, llm=backend, k=1)
    answer = chat.ask("where does Alex live?")
    assert answer.text == "captured-reply"
    assert "where does Alex live?" in backend.last_prompt
    assert "Alex lives in Portland" in backend.last_prompt
    assert answer.backend_name == "capturing"
    assert len(answer.hits) == 1


def test_rag_session_uses_no_context_template_when_empty() -> None:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    backend = CapturingBackend()
    chat = RAGSession(memory=mem, llm=backend, k=5)
    answer = chat.ask("any question")
    assert "Nothing relevant was retrieved" in backend.last_prompt
    assert answer.hits == []


def test_rag_answer_cite_lines_renders_top_k() -> None:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    mem.store("a", metadata={"path": "a.md", "heading": "H1"})
    mem.store("b", metadata={"path": "b.md"})
    chat = RAGSession(memory=mem, llm=DryRunBackend(), k=2)
    answer = chat.ask("?")
    lines = answer.cite_lines()
    assert len(lines) == 2
    assert "a.md (H1)" in lines[0] or "b.md" in lines[0]


def test_rag_session_is_stateless_across_asks() -> None:
    """Two ask calls should each retrieve fresh — no hidden state."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    mem.store("first", metadata={"path": "1.md"})
    backend = CapturingBackend()
    chat = RAGSession(memory=mem, llm=backend, k=5)
    chat.ask("q1")
    p1 = backend.last_prompt
    chat.ask("q2")
    p2 = backend.last_prompt
    assert p1 != p2
    assert "q1" in p1 and "q2" in p2
    assert "first" in p1 and "first" in p2  # memory persists
