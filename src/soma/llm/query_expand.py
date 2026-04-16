"""LLM-driven query expansion for RAG recall.

When a user's question is terse / under-specified / ambiguous, the
first-k retrieval often misses relevant context because cosine
similarity over the raw query doesn't match how the *answer* is
phrased in memory. A cheap fix: ask the LLM to rewrite the question
into several paraphrases + sub-questions, retrieve top-k for each,
then merge the result lists with Reciprocal Rank Fusion (RRF).

Expected lift: +3-8% Recall@5 on hard / under-specified queries.
Cost: one extra LLM call per ask(), proportional to `n_variants`.

Usage::

    from soma.llm import OllamaBackend, QueryExpander, RAGSession

    backend = OllamaBackend(model="llama3.2")
    chat = RAGSession(
        memory=mem,
        llm=backend,
        query_expander=QueryExpander(llm=backend, n_variants=3),
    )
    chat.ask("what was that thing I mentioned about dinner?")
    # Internally expands to 3 variants, retrieves k per variant,
    # RRF-merges into the final context sent to the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from soma.llm.backends import LLMBackend

DEFAULT_EXPAND_PROMPT = (
    "You will rewrite a user question into several alternative phrasings "
    "plus sub-questions that together cover likely ways the answer might "
    "appear in a knowledge base. Return a bullet list, one variant per "
    "line, each starting with '- '. Keep variants short (<= 15 words). "
    "Do not include the original question; generate only alternatives.\n\n"
    "Original question: {question}\n\n"
    "{n} alternative variants:"
)

_BULLET_RE = re.compile(r"^\s*[-*\u2022]\s*(.+?)\s*$")


@dataclass
class QueryExpander:
    """Rewrites one question into N variant queries via the provided LLM.

    The prompt template is overridable; the default asks the LLM for a
    bullet list that the expander parses back into variants. Any
    backend that satisfies :class:`LLMBackend` works (Ollama, OpenAI,
    Anthropic, OpenAI-compatible, HF).
    """

    llm: LLMBackend
    n_variants: int = 3
    prompt_template: str = DEFAULT_EXPAND_PROMPT
    max_tokens: int = 200
    include_original: bool = True

    def expand(self, question: str) -> list[str]:
        """Return a list of variant queries (possibly including the
        original, depending on ``include_original``)."""
        if self.n_variants <= 0:
            return [question] if self.include_original else []
        prompt = self.prompt_template.format(
            question=question, n=self.n_variants
        )
        raw = self.llm.generate(prompt, max_tokens=self.max_tokens)
        variants = _parse_bullets(raw)[: self.n_variants]
        if self.include_original:
            return [question, *variants]
        return variants or [question]


def _parse_bullets(text: str) -> list[str]:
    """Pull bullet-prefixed lines out of an LLM reply."""
    out: list[str] = []
    for line in text.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            variant = m.group(1).strip().strip('"').strip("'")
            if variant:
                out.append(variant)
    # Fallback: if the model ignored the bullet-format instruction,
    # take short non-empty lines as variants.
    if not out:
        out = [
            s.strip().strip('"').strip("'")
            for s in text.splitlines()
            if s.strip() and len(s.strip()) < 200
        ]
    return out


def rrf_merge(
    ranked_lists: list[list[Any]], *, k_rrf: int = 60, top_k: int | None = None
) -> list[Any]:
    """Reciprocal Rank Fusion: combine multiple ranked hit lists.

    Each item's fused score is ``sum(1 / (k_rrf + rank))`` across
    lists that contain it. Ties broken by first appearance across
    lists. Works on any object with a ``node_id`` attribute.
    """
    scores: dict[str, float] = {}
    first_hit: dict[str, Any] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits):
            raw = getattr(hit, "node_id", None)
            nid: str = str(raw) if raw is not None else f"__id_{id(hit)}"
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k_rrf + rank + 1)
            first_hit.setdefault(nid, hit)
    order = sorted(scores.items(), key=lambda kv: -kv[1])
    merged = [first_hit[nid] for nid, _s in order]
    return merged[:top_k] if top_k else merged
