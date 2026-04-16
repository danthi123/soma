"""Tests for LLM query expansion + RRF merge."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from soma.llm import QueryExpander, RAGSession, rrf_merge
from soma.llm.query_expand import _parse_bullets
from soma.memory import MemoryLayer
from soma.memory.api import MemoryHit


def _stub_embed(text: str) -> torch.Tensor:
    h = hash(text)
    return torch.tensor(
        [(h >> i) & 0xF for i in range(0, 32, 4)], dtype=torch.float32
    )


@dataclass
class ScriptedBackend:
    """LLMBackend stub that replays pre-canned responses in order."""

    replies: list[str]
    name: str = "scripted"
    _prompts: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._prompts = []

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self._prompts.append(prompt)
        return self.replies.pop(0)


# ------------------------------------------------------------------
# _parse_bullets
# ------------------------------------------------------------------


def test_parse_bullets_recognizes_dash_star_and_unicode() -> None:
    text = """- first variant
* second variant
\u2022 third variant
not a bullet line
- fourth
"""
    out = _parse_bullets(text)
    assert out == ["first variant", "second variant", "third variant", "fourth"]


def test_parse_bullets_strips_surrounding_quotes() -> None:
    text = '- "hello"\n- \'world\''
    assert _parse_bullets(text) == ["hello", "world"]


def test_parse_bullets_falls_back_to_lines_when_no_bullets() -> None:
    text = "one rephrase\ntwo rephrase\nthree rephrase"
    out = _parse_bullets(text)
    assert "one rephrase" in out


# ------------------------------------------------------------------
# rrf_merge
# ------------------------------------------------------------------


def _h(node_id: str) -> MemoryHit:
    return MemoryHit(node_id=node_id, text=node_id, score=1.0, metadata={})


def test_rrf_gives_higher_fused_score_to_items_ranked_high_in_many_lists() -> None:
    a, b, c, d = _h("a"), _h("b"), _h("c"), _h("d")
    lists = [
        [a, b, c],  # a@1, b@2, c@3
        [a, c, b],  # a@1, c@2, b@3
        [b, a, d],  # b@1, a@2, d@3
    ]
    merged = rrf_merge(lists)
    nids = [h.node_id for h in merged]
    # 'a' appears rank-1 twice, rank-2 once → highest fused score.
    assert nids[0] == "a"
    # 'd' appears once late → lowest of the four.
    assert nids[-1] == "d"


def test_rrf_preserves_unique_items() -> None:
    a, b = _h("a"), _h("b")
    merged = rrf_merge([[a], [b]])
    assert {h.node_id for h in merged} == {"a", "b"}


def test_rrf_top_k_truncates() -> None:
    hits = [_h(str(i)) for i in range(10)]
    merged = rrf_merge([hits], top_k=3)
    assert len(merged) == 3


# ------------------------------------------------------------------
# QueryExpander
# ------------------------------------------------------------------


def test_query_expander_parses_bullet_reply() -> None:
    backend = ScriptedBackend(
        replies=["- variant one\n- variant two\n- variant three"]
    )
    expander = QueryExpander(llm=backend, n_variants=3, include_original=False)
    variants = expander.expand("original question")
    assert variants == ["variant one", "variant two", "variant three"]


def test_query_expander_includes_original_by_default() -> None:
    backend = ScriptedBackend(replies=["- paraphrase"])
    expander = QueryExpander(llm=backend, n_variants=1)
    variants = expander.expand("orig")
    assert variants == ["orig", "paraphrase"]


def test_query_expander_truncates_to_n_variants() -> None:
    backend = ScriptedBackend(
        replies=["- a\n- b\n- c\n- d\n- e"]
    )
    expander = QueryExpander(llm=backend, n_variants=2, include_original=False)
    variants = expander.expand("q")
    assert len(variants) == 2
    assert variants == ["a", "b"]


def test_query_expander_zero_variants_returns_only_original() -> None:
    backend = ScriptedBackend(replies=[])
    expander = QueryExpander(llm=backend, n_variants=0)
    assert expander.expand("only-me") == ["only-me"]


def test_query_expander_zero_variants_no_original_returns_empty() -> None:
    backend = ScriptedBackend(replies=[])
    expander = QueryExpander(llm=backend, n_variants=0, include_original=False)
    assert expander.expand("ignored") == []


def test_query_expander_falls_back_when_llm_ignores_bullet_format() -> None:
    backend = ScriptedBackend(
        replies=["paraphrase one\nparaphrase two"]
    )
    expander = QueryExpander(llm=backend, n_variants=2, include_original=False)
    variants = expander.expand("q")
    assert "paraphrase one" in variants


# ------------------------------------------------------------------
# RAGSession + QueryExpander composition
# ------------------------------------------------------------------


def test_rag_session_with_expander_retrieves_per_variant() -> None:
    """When an expander is attached, ask() should call retrieve()
    once per variant and RRF-merge. Use a scripted backend whose
    expansion reply AND answer reply are both fixed."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    mem.store("user lives in Portland, OR", metadata={"path": "bio.md"})
    mem.store("user works at ArcMotion", metadata={"path": "bio.md"})
    mem.store("user eats vegetarian food", metadata={"path": "bio.md"})

    expander_backend = ScriptedBackend(replies=["- where does the user live?\n- city of residence"])
    ask_backend = ScriptedBackend(replies=["final answer [1]"])

    expander = QueryExpander(
        llm=expander_backend, n_variants=2, include_original=False
    )
    chat = RAGSession(memory=mem, llm=ask_backend, k=3, query_expander=expander)
    answer = chat.ask("where does the user live?")
    # The expander's LLM was called once (to produce variants).
    assert len(expander_backend._prompts) == 1
    # The ask backend was called once (with merged context).
    assert len(ask_backend._prompts) == 1
    assert answer.text == "final answer [1]"
    # Context prompt should include one of the stored entries.
    assert any(
        frag in ask_backend._prompts[0]
        for frag in ("Portland", "ArcMotion", "vegetarian")
    )


def test_rag_session_without_expander_still_works() -> None:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    mem.store("something")
    backend = ScriptedBackend(replies=["reply"])
    chat = RAGSession(memory=mem, llm=backend, k=1)
    assert chat.ask("q").text == "reply"


def test_rag_session_passes_retrieve_kwargs_through() -> None:
    """retrieve_kwargs on RAGSession should forward to mem.retrieve."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    mem.store("a", metadata={"u": "alex"})
    mem.store("b", metadata={"u": "bobbi"})
    backend = ScriptedBackend(replies=["r"])
    chat = RAGSession(
        memory=mem, llm=backend, k=5, retrieve_kwargs={"where": {"u": "alex"}}
    )
    answer = chat.ask("anything")
    # Only alex-tagged hits reach the prompt.
    assert len(answer.hits) == 1
    assert answer.hits[0].metadata["u"] == "alex"
