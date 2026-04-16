"""RAG glue: turns a MemoryLayer + LLMBackend into a one-call ``ask``.

Most callers just want::

    chat = RAGSession(memory=mem, llm=backend)
    chat.ask("what does the user prefer for dinner?")

This file is the small layer that does retrieve → format prompt with
citations → backend.generate → return answer + sources. Prompt template
is overridable (pass ``prompt_template=`` or subclass) but defaults to
a citation-friendly format that works across most chat-tuned models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from soma.llm.backends import LLMBackend
from soma.llm.query_expand import QueryExpander, rrf_merge
from soma.memory import MemoryLayer

DEFAULT_TEMPLATE = (
    "You are a helpful assistant answering questions using the user's "
    "stored memory. Use only the context below; if the context doesn't "
    "contain the answer, say so rather than guessing. Cite sources "
    "inline like [1], [2], etc.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer (cite sources like [1]):"
)

NO_CONTEXT_TEMPLATE = (
    "You are a helpful assistant. Nothing relevant was retrieved from "
    "the user's memory for this question. Answer honestly and say so "
    "if appropriate.\n\n"
    "Question: {question}\nAnswer:"
)


def format_context(hits: list[Any]) -> str:
    """Default citation-friendly context block."""
    lines = []
    for i, h in enumerate(hits, 1):
        src_bits = []
        if "path" in h.metadata:
            src_bits.append(str(h.metadata["path"]))
        if "heading" in h.metadata and h.metadata["heading"]:
            src_bits.append(f"heading: {h.metadata['heading']}")
        if "source" in h.metadata:
            src_bits.append(str(h.metadata["source"]))
        src = "; ".join(src_bits) if src_bits else f"id={h.node_id[:8]}"
        lines.append(f"[{i}] {src}\n    {h.text}")
    return "\n\n".join(lines)


@dataclass
class RAGAnswer:
    text: str
    hits: list[Any]
    backend_name: str

    def cite_lines(self) -> list[str]:
        """Render hits as ``[i] short-source`` lines for display."""
        out: list[str] = []
        for i, h in enumerate(self.hits, 1):
            path = h.metadata.get("path")
            heading = h.metadata.get("heading")
            src = path or h.metadata.get("source") or h.node_id[:8]
            if heading:
                src = f"{src} ({heading})"
            out.append(f"[{i}] {src}  score={h.score:.3f}")
        return out


@dataclass
class RAGSession:
    """Stateless retrieve-then-generate wrapper.

    ``ask(question)`` retrieves top-k from ``memory``, builds a prompt
    via ``prompt_template``, calls ``llm.generate``, returns
    :class:`RAGAnswer` with the model's text + the hits used.

    The session is intentionally stateless — for multi-turn chat where
    the model should see prior turns, push each turn back into the
    memory layer (``memory.store(turn)``) and the next ``ask`` will
    retrieve relevant turns alongside the static knowledge.
    """

    memory: MemoryLayer
    llm: LLMBackend
    k: int = 5
    max_tokens: int = 256
    prompt_template: str = DEFAULT_TEMPLATE
    no_context_template: str = NO_CONTEXT_TEMPLATE
    format_context_fn: Any = field(default=None)
    query_expander: QueryExpander | None = None
    retrieve_kwargs: dict[str, Any] = field(default_factory=dict)

    def _build_prompt(self, question: str, hits: list[Any]) -> str:
        if not hits:
            return self.no_context_template.format(question=question)
        ctx = (self.format_context_fn or format_context)(hits)
        return self.prompt_template.format(context=ctx, question=question)

    def _retrieve(self, question: str) -> list[Any]:
        """Retrieve with optional LLM query expansion + RRF merge."""
        if self.query_expander is None:
            return self.memory.retrieve(question, k=self.k, **self.retrieve_kwargs)
        variants = self.query_expander.expand(question)
        ranked_lists = [
            self.memory.retrieve(v, k=self.k, **self.retrieve_kwargs)
            for v in variants
        ]
        return rrf_merge(ranked_lists, top_k=self.k)

    def ask(self, question: str) -> RAGAnswer:
        hits = self._retrieve(question)
        prompt = self._build_prompt(question, hits)
        text = self.llm.generate(prompt, max_tokens=self.max_tokens)
        return RAGAnswer(text=text, hits=hits, backend_name=self.llm.name)
