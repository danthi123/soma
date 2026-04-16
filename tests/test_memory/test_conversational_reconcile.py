"""Tests for ConversationalMemory._reconcile — the threshold short-circuit
+ LLM fallback.

Contract (per plan §3):
- Max-score >= 0.92 → skip (near-duplicate), LLM NEVER called.
- Max-score < 0.75 → ADD unconditionally, LLM NEVER called.
- 0.75 <= max-score < 0.92 → ambiguous; LLM decides
  ADD / UPDATE / SUPERSEDE / NOOP.

SUPERSEDE writes ``metadata.superseded_by = new_id`` on the OLD entry
(via MemoryLayer.update_metadata), not a forget. UPDATE forgets the
old entry and stores the new one with metadata.supersedes pointer.
NOOP stores nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory, ExtractedFact


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic per-text vector. Small overlap between similar phrasings."""
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


@dataclass
class _CountingBackend:
    """LLM backend that returns scripted replies and counts calls."""

    replies: list[str] = field(default_factory=list)
    name: str = "reconcile-counting"
    call_count: int = 0
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        idx = self.call_count - 1
        if idx < len(self.replies):
            return self.replies[idx]
        if self.replies:
            return self.replies[-1]
        return "[]"


class _FixedScoreLayer(MemoryLayer):
    """MemoryLayer subclass whose retrieve() returns scripted scores.

    Used to pin the threshold short-circuit behaviour without fighting
    cosine over randomly-seeded vectors.
    """

    _scripted_scores: list[float]

    def __init__(
        self,
        *,
        scripted_scores: list[float],
        embed_fn: Any,
        embed_dim: int,
    ) -> None:
        super().__init__(embed_fn=embed_fn, embed_dim=embed_dim)
        self._scripted_scores = list(scripted_scores)

    def retrieve(self, query: str, k: int = 5, **kwargs: Any):  # type: ignore[override]
        # Reuse the real ranking, then rewrite scores per the scripted list.
        hits = super().retrieve(query, k=k, **kwargs)
        if not hits or not self._scripted_scores:
            return hits
        from soma.memory.api import MemoryHit

        out = []
        for i, h in enumerate(hits):
            score = self._scripted_scores[min(i, len(self._scripted_scores) - 1)]
            out.append(
                MemoryHit(
                    node_id=h.node_id,
                    text=h.text,
                    score=float(score),
                    metadata=dict(h.metadata),
                    timestamp_step=h.timestamp_step,
                )
            )
        return out


_FACT_META = {"type": "fact", "session_id": "s"}


def test_near_dup_skips_without_llm_call() -> None:
    """Top-1 score >= 0.92 → reconcile returns None, LLM never called."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.95, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    mem.store("User lives in Boston", metadata=dict(_FACT_META))
    llm = _CountingBackend()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="location", text="User lives in Boston")

    before = len(mem)
    result = cm._reconcile(fact)
    assert result is None, "near-dup should short-circuit to None"
    assert llm.call_count == 0, "LLM must not be called in near-dup case"
    assert len(mem) == before, "no new entry should be added"


def test_low_similarity_adds_unconditionally() -> None:
    """Max score < 0.75 → ADD without calling the LLM."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.3, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    mem.store("totally unrelated stuff", metadata=dict(_FACT_META))
    llm = _CountingBackend()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="location", text="User lives in Boston")

    result = cm._reconcile(fact)
    assert result is not None, "low-similarity must return the new node_id"
    assert llm.call_count == 0
    assert len(mem) == 2
    new_hit = mem.get(result)
    assert new_hit is not None
    assert new_hit.text == "User lives in Boston"


def test_empty_store_adds() -> None:
    """Empty retrieve result → treated as low-similarity, ADD without LLM."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _CountingBackend()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="identity", text="User's name is Alex")

    result = cm._reconcile(fact)
    assert result is not None
    assert llm.call_count == 0
    assert len(mem) == 1


def test_ambiguous_range_calls_llm_and_adds() -> None:
    """0.75 <= max-score < 0.92: LLM called exactly once; ADD op honoured."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.8, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    mem.store("Something tangentially related", metadata=dict(_FACT_META))
    llm = _CountingBackend(
        replies=[json.dumps({"op": "ADD", "target_id": None, "reason": "new"})]
    )
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="preference", text="User prefers Thai food")

    result = cm._reconcile(fact)
    assert llm.call_count == 1, "ambiguous case must call LLM exactly once"
    assert result is not None
    assert len(mem) == 2


def test_supersede_writes_metadata_pointer_not_forget() -> None:
    """SUPERSEDE: old entry stays with metadata.superseded_by; new entry
    stored with metadata.supersedes pointing back."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.82, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    old_id = mem.store(
        "User lives in Portland",
        metadata={"type": "fact", "session_id": "s"},
    )
    llm = _CountingBackend(
        replies=[
            json.dumps(
                {"op": "SUPERSEDE", "target_id": old_id, "reason": "user moved"}
            )
        ]
    )
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="location", text="User lives in Boston")

    new_id = cm._reconcile(fact)
    assert new_id is not None
    assert new_id != old_id
    # Old entry still present, but flagged.
    old_hit = mem.get(old_id)
    assert old_hit is not None, "SUPERSEDE must NOT delete the old entry"
    assert old_hit.metadata.get("superseded_by") == new_id
    # New entry carries the back-pointer.
    new_hit = mem.get(new_id)
    assert new_hit is not None
    assert new_hit.metadata.get("supersedes") == old_id


def test_update_replaces_text_but_keeps_history() -> None:
    """UPDATE: forget the old entry, store the new with supersedes pointer."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.85, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    old_id = mem.store(
        "User likes pizza",
        metadata={"type": "fact", "session_id": "s"},
    )
    llm = _CountingBackend(
        replies=[
            json.dumps(
                {"op": "UPDATE", "target_id": old_id, "reason": "more detail"}
            )
        ]
    )
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(
        category="preference", text="User likes pepperoni pizza"
    )

    new_id = cm._reconcile(fact)
    assert new_id is not None
    # Old entry is gone (UPDATE forgets old).
    assert mem.get(old_id) is None
    new_hit = mem.get(new_id)
    assert new_hit is not None
    assert new_hit.metadata.get("supersedes") == old_id


def test_noop_stores_nothing() -> None:
    """NOOP: memory unchanged, reconcile returns None."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.80, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    old_id = mem.store(
        "User loves Thai food", metadata={"type": "fact", "session_id": "s"},
    )
    before_count = len(mem)
    llm = _CountingBackend(
        replies=[
            json.dumps(
                {"op": "NOOP", "target_id": old_id, "reason": "already covered"}
            )
        ]
    )
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="preference", text="User enjoys Thai")

    result = cm._reconcile(fact)
    assert result is None
    assert len(mem) == before_count


def test_reconcile_bad_json_falls_back_to_add() -> None:
    """LLM returns garbage in the ambiguous range → safe fallback: ADD."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.80, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    mem.store("something", metadata=dict(_FACT_META))
    llm = _CountingBackend(replies=["not valid json at all"])
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="location", text="User lives in Boston")

    before = len(mem)
    result = cm._reconcile(fact)
    # Safer to ADD than to lose the fact on parse failure.
    assert result is not None
    assert len(mem) == before + 1


def test_reconcile_unknown_op_falls_back_to_add() -> None:
    """LLM returns a valid JSON but unknown op → ADD fallback."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.80, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    mem.store("something", metadata=dict(_FACT_META))
    llm = _CountingBackend(
        replies=[json.dumps({"op": "MAYBE", "target_id": None, "reason": "?"})]
    )
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="location", text="User lives in Boston")

    before = len(mem)
    result = cm._reconcile(fact)
    assert result is not None
    assert len(mem) == before + 1


def test_update_with_missing_target_falls_back_to_add() -> None:
    """UPDATE without a valid target_id → can't resolve → ADD fallback."""
    mem = _FixedScoreLayer(
        scripted_scores=[0.80, 0.1],
        embed_fn=_stub_embed,
        embed_dim=16,
    )
    mem.store("anchor", metadata=dict(_FACT_META))
    llm = _CountingBackend(
        replies=[
            json.dumps({"op": "UPDATE", "target_id": None, "reason": "huh"})
        ]
    )
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    fact = ExtractedFact(category="location", text="User lives in Boston")

    before = len(mem)
    result = cm._reconcile(fact)
    assert result is not None
    assert len(mem) == before + 1
