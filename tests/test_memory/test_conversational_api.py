"""Tests for ConversationalMemory's public API:
add_message, retrieve, list_facts, supersede, clear_session,
get_summary, flush.

Session summaries roll every N turns (``summary_every=``, default 20).
Raw turns are stored so LoCoMo-style eval stays compatible, but
retrieve() scopes to this session and filters out superseded facts
by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic but non-trivially-overlapping vectors."""
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


@dataclass
class _ScriptedLLM:
    """Returns scripted replies per call, captures prompts for inspection."""

    extract_replies: list[str] = field(default_factory=list)
    reconcile_replies: list[str] = field(default_factory=list)
    summary_reply: str = "rolling summary line"
    name: str = "scripted"
    extract_calls: int = 0
    reconcile_calls: int = 0
    summary_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        if "You extract atomic facts" in prompt:
            self.extract_calls += 1
            i = self.extract_calls - 1
            if i < len(self.extract_replies):
                return self.extract_replies[i]
            return "[]"
        if "You reconcile a new fact" in prompt:
            self.reconcile_calls += 1
            i = self.reconcile_calls - 1
            if i < len(self.reconcile_replies):
                return self.reconcile_replies[i]
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        if "summarizing a short segment" in prompt:
            self.summary_calls += 1
            return self.summary_reply
        return "{}"


def _fact(category: str, text: str) -> dict[str, Any]:
    return {"category": category, "text": text}


def test_add_user_message_extracts_and_reconciles() -> None:
    """User turn → facts extracted + reconciled + raw turn stored with type=turn."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[
            json.dumps(
                [
                    _fact("identity", "User's name is Alex"),
                    _fact("location", "User lives in Boston"),
                ]
            )
        ],
    )
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="alex")
    cm.add_message("user", "I'm Alex and live in Boston")

    # 2 facts + 1 raw turn stored.
    entries = [mem.get(nid) for nid in mem._ids]
    facts = [e for e in entries if e and e.metadata.get("type") == "fact"]
    turns = [e for e in entries if e and e.metadata.get("type") == "turn"]
    assert len(facts) == 2
    assert len(turns) == 1
    assert turns[0].text == "I'm Alex and live in Boston"
    assert turns[0].metadata["session_id"] == "alex"
    assert turns[0].metadata["role"] == "user"


def test_add_assistant_message_skipped_by_default() -> None:
    """role="assistant" messages: raw turn stored, no fact extraction."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="alex")
    cm.add_message("assistant", "Hello, how can I help?")

    assert llm.extract_calls == 0, "assistant turns must not trigger extraction"
    entries = [mem.get(nid) for nid in mem._ids]
    turns = [e for e in entries if e and e.metadata.get("type") == "turn"]
    assert len(turns) == 1
    assert turns[0].metadata["role"] == "assistant"


def test_summary_rolls_every_n_turns() -> None:
    """With summary_every=3, a summary entry is stored at turn 3."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=["[]", "[]", "[]"],
        summary_reply="Summary of turns 1-3",
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s", summary_every=3
    )
    cm.add_message("user", "msg 1")
    cm.add_message("user", "msg 2")
    assert llm.summary_calls == 0
    cm.add_message("user", "msg 3")
    assert llm.summary_calls == 1
    # Find the summary entry.
    summaries = [
        mem.get(nid)
        for nid in mem._ids
        if (hit := mem.get(nid)) and hit.metadata.get("type") == "summary"
    ]
    assert len(summaries) == 1
    assert summaries[0].text == "Summary of turns 1-3"
    assert summaries[0].metadata["session_id"] == "s"


def test_retrieve_returns_facts_and_summaries_and_turns_together() -> None:
    """retrieve() competes facts, summaries, and turns by cosine."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[
            json.dumps([_fact("location", "User lives in Boston")]),
        ],
        summary_reply="User introduced themselves.",
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s", summary_every=1,
    )
    cm.add_message("user", "I live in Boston")

    hits = cm.retrieve("Boston", k=5)
    assert len(hits) >= 1
    # At least one of the hits should be the fact about Boston.
    assert any(
        "Boston" in h.text and h.metadata.get("type") == "fact" for h in hits
    )


def test_retrieve_excludes_superseded_by_default() -> None:
    """Superseded facts must not appear in default retrieve()."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")

    # Hand-seed a superseded fact.
    old_id = mem.store(
        "User lives in Portland",
        metadata={"type": "fact", "session_id": "s"},
    )
    new_id = mem.store(
        "User lives in Boston",
        metadata={
            "type": "fact",
            "session_id": "s",
            "supersedes": old_id,
        },
    )
    mem.update_metadata(old_id, {"superseded_by": new_id})

    hits = cm.retrieve("where does the user live?", k=5)
    hit_ids = {h.node_id for h in hits}
    assert new_id in hit_ids
    assert old_id not in hit_ids, (
        "default retrieve must exclude superseded entries"
    )


def test_retrieve_include_superseded_true_returns_history() -> None:
    """include_superseded=True lets callers see invalidated entries."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")

    old_id = mem.store(
        "User lives in Portland",
        metadata={"type": "fact", "session_id": "s"},
    )
    new_id = mem.store(
        "User lives in Boston",
        metadata={"type": "fact", "session_id": "s", "supersedes": old_id},
    )
    mem.update_metadata(old_id, {"superseded_by": new_id})

    hits = cm.retrieve(
        "where does the user live?", k=10, include_superseded=True,
    )
    hit_ids = {h.node_id for h in hits}
    assert old_id in hit_ids
    assert new_id in hit_ids


def test_clear_session_keeps_summaries_by_default() -> None:
    """clear_session() removes turns + facts but keeps summaries by default."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=["[]"],
        summary_reply="accumulated summary",
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s", summary_every=1,
    )
    cm.add_message("user", "hello")

    summaries_before = [
        mem.get(nid) for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "summary"
    ]
    assert len(summaries_before) >= 1

    cm.clear_session()
    # All turns + facts gone; summaries remain.
    remaining = [mem.get(nid) for nid in mem._ids]
    remaining_by_type: dict[str, int] = {}
    for e in remaining:
        if e is None:
            continue
        t = e.metadata.get("type", "unknown")
        remaining_by_type[t] = remaining_by_type.get(t, 0) + 1
    assert remaining_by_type.get("turn", 0) == 0
    assert remaining_by_type.get("fact", 0) == 0
    assert remaining_by_type.get("summary", 0) >= 1


def test_list_facts_filters_by_session_id() -> None:
    """Multi-session isolation: list_facts() only returns this session's facts."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    mem.store(
        "User A fact", metadata={"type": "fact", "session_id": "a"},
    )
    mem.store(
        "User B fact", metadata={"type": "fact", "session_id": "b"},
    )
    llm = _ScriptedLLM()
    cm_a = ConversationalMemory(memory=mem, llm=llm, session_id="a")
    cm_b = ConversationalMemory(memory=mem, llm=llm, session_id="b")

    a_facts = cm_a.list_facts()
    b_facts = cm_b.list_facts()
    assert len(a_facts) == 1 and a_facts[0].text == "User A fact"
    assert len(b_facts) == 1 and b_facts[0].text == "User B fact"


def test_get_summary_returns_most_recent() -> None:
    """get_summary() returns the last stored summary (by step)."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=["[]", "[]"],
        summary_reply="summary A",
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s", summary_every=1,
    )
    cm.add_message("user", "first")
    llm.summary_reply = "summary B"
    cm.add_message("user", "second")

    latest = cm.get_summary()
    assert latest is not None
    assert latest.text == "summary B"


def test_get_summary_none_when_no_summaries() -> None:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    assert cm.get_summary() is None


def test_flush_is_noop_in_sync_mode() -> None:
    """flush() must not raise when no bundle is attached."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    cm.flush()  # should be a no-op and not raise


def test_supersede_public_helper_marks_entry() -> None:
    """supersede(old_id, new_text) writes the same pointers as the LLM path."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")
    old_id = mem.store(
        "old fact", metadata={"type": "fact", "session_id": "s"},
    )
    new_id = cm.supersede(old_id, "new fact")
    old_hit = mem.get(old_id)
    new_hit = mem.get(new_id)
    assert old_hit is not None
    assert new_hit is not None
    assert old_hit.metadata.get("superseded_by") == new_id
    assert new_hit.metadata.get("supersedes") == old_id


def test_default_session_id_is_default_string() -> None:
    """No session_id kwarg -> internally uses "default"."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm)
    assert cm._session_id == "default"
