"""Tests for ConversationalMemory's Phase 12 multi-user scoping.

Contract (per ``docs/plans/2026-04-16-phase-12-multi-user-scoping.md``):

- ``ConversationalMemory(memory=mem, llm=llm, user_id="alice", ...)``
  tags every stored turn, fact, and summary with
  ``metadata.user_id = "alice"``.
- ``add_message(..., user_id="bob")`` overrides the constructor-set
  user_id for that one call; passing ``user_id=None`` explicitly drops
  the tag for the call.
- ``retrieve()`` filters to the constructor's user_id by default,
  composed with any caller-supplied ``where`` via AND. Pass
  ``user_id=None`` to see across all users (admin drill-down).
- ``supersede()`` refuses to mutate a fact owned by a different user
  (raises :class:`PermissionError`).
- ``clear_session()`` filters to this user_id when set.
- When ``user_id`` is unset everywhere (constructor + per-call), no
  ``user_id`` key appears in stored metadata — pre-Phase-12 bundles
  stay byte-identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic per-text vector."""
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


@dataclass
class _ScriptedLLM:
    """Minimal scripted LLM — covers extract / reconcile / summary prompts."""

    extract_replies: list[str] = field(default_factory=list)
    reconcile_replies: list[str] = field(default_factory=list)
    summary_reply: str = "rolling summary line"
    name: str = "scripted"
    extract_calls: int = 0
    reconcile_calls: int = 0
    summary_calls: int = 0

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            i = self.extract_calls
            self.extract_calls += 1
            if i < len(self.extract_replies):
                return self.extract_replies[i]
            return "[]"
        if "You reconcile a new fact" in prompt:
            i = self.reconcile_calls
            self.reconcile_calls += 1
            if i < len(self.reconcile_replies):
                return self.reconcile_replies[i]
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        if "summarizing a short segment" in prompt:
            self.summary_calls += 1
            return self.summary_reply
        return "{}"


def _fact(category: str, text: str) -> dict[str, Any]:
    return {"category": category, "text": text}


# ----------------------------------------------------------------------
# 1. user_id is stamped on every write (turn, fact, summary)
# ----------------------------------------------------------------------
def test_user_id_stored_on_every_write() -> None:
    """Turn + fact + summary metadata each carry user_id=alice."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[
            json.dumps([_fact("identity", "User's name is Alice")]),
        ],
        summary_reply="Alice introduced herself.",
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="chat-1", user_id="alice",
        summary_every=1,
    )
    cm.add_message("user", "I'm Alice")

    entries = [mem.get(nid) for nid in mem._ids]
    turns = [e for e in entries if e and e.metadata.get("type") == "turn"]
    facts = [e for e in entries if e and e.metadata.get("type") == "fact"]
    summaries = [e for e in entries if e and e.metadata.get("type") == "summary"]

    assert turns, "expected at least one stored turn"
    assert facts, "expected at least one stored fact"
    assert summaries, "expected at least one stored summary"

    for e in turns + facts + summaries:
        assert e is not None
        assert e.metadata.get("user_id") == "alice", (
            f"{e.metadata.get('type')} entry missing user_id: {e.metadata!r}"
        )


# ----------------------------------------------------------------------
# 2. retrieve() scopes to constructor user_id by default
# ----------------------------------------------------------------------
def test_retrieve_default_scopes_to_user_id() -> None:
    """cm(user_id='a').retrieve() returns only Alice's entries."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()

    # Two Alice facts + two Bob facts, all for the SAME session_id so the
    # only discriminator is user_id. The test fails unless retrieve()
    # threads user_id into the where-filter.
    mem.store(
        "Alice likes hiking",
        metadata={"type": "fact", "session_id": "shared", "user_id": "alice"},
    )
    mem.store(
        "Alice lives in Boston",
        metadata={"type": "fact", "session_id": "shared", "user_id": "alice"},
    )
    mem.store(
        "Bob likes hiking",
        metadata={"type": "fact", "session_id": "shared", "user_id": "bob"},
    )
    mem.store(
        "Bob lives in Seattle",
        metadata={"type": "fact", "session_id": "shared", "user_id": "bob"},
    )

    cm_a = ConversationalMemory(
        memory=mem, llm=llm, session_id="shared", user_id="alice",
    )
    hits = cm_a.retrieve("hiking", k=10)
    assert hits, "expected Alice to have at least one hit"
    for h in hits:
        assert h.metadata.get("user_id") == "alice", (
            f"leak: non-alice hit in alice-scoped retrieve: {h.metadata!r}"
        )


# ----------------------------------------------------------------------
# 3. retrieve(user_id=None) is the admin drill-down
# ----------------------------------------------------------------------
def test_retrieve_user_id_none_sees_all() -> None:
    """Explicit user_id=None overrides the constructor scope."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    mem.store(
        "Alice likes hiking",
        metadata={"type": "fact", "session_id": "shared", "user_id": "alice"},
    )
    mem.store(
        "Bob likes hiking",
        metadata={"type": "fact", "session_id": "shared", "user_id": "bob"},
    )
    cm_a = ConversationalMemory(
        memory=mem, llm=llm, session_id="shared", user_id="alice",
    )

    all_hits = cm_a.retrieve("hiking", k=10, user_id=None)
    users = {h.metadata.get("user_id") for h in all_hits}
    assert "alice" in users and "bob" in users, (
        f"admin drill-down must see all users, got: {users!r}"
    )


# ----------------------------------------------------------------------
# 4. add_message(user_id=...) overrides the constructor value
# ----------------------------------------------------------------------
def test_per_call_user_id_overrides_constructor() -> None:
    """add_message(..., user_id='b') stamps 'b' even when constructor said 'a'."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="chat-1", user_id="alice",
    )
    cm.add_message("user", "hi there", user_id="bob")

    entries = [mem.get(nid) for nid in mem._ids]
    turns = [e for e in entries if e and e.metadata.get("type") == "turn"]
    assert len(turns) == 1
    assert turns[0].metadata.get("user_id") == "bob", (
        f"expected user_id=bob on overridden turn, got {turns[0].metadata!r}"
    )


# ----------------------------------------------------------------------
# 5. Pre-Phase-12 behaviour preserved when user_id never set
# ----------------------------------------------------------------------
def test_user_id_unset_preserves_pre_phase12_behaviour() -> None:
    """No user_id kwarg anywhere -> no user_id key in any stored metadata."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[
            json.dumps([_fact("identity", "User's name is Charlie")]),
        ],
        summary_reply="Charlie introduced themselves.",
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="chat-1", summary_every=1,
    )
    cm.add_message("user", "I'm Charlie")

    entries = [mem.get(nid) for nid in mem._ids]
    assert entries, "expected some entries"
    for e in entries:
        assert e is not None
        assert "user_id" not in e.metadata, (
            f"pre-Phase-12 callers must see NO user_id key: {e.metadata!r}"
        )


# ----------------------------------------------------------------------
# 6. supersede() refuses cross-user mutation
# ----------------------------------------------------------------------
def test_supersede_respects_user_id() -> None:
    """cm(user=alice).supersede(bob's fact) raises PermissionError."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()

    bob_fact_id = mem.store(
        "Bob lives in Seattle",
        metadata={"type": "fact", "session_id": "shared", "user_id": "bob"},
    )

    cm_alice = ConversationalMemory(
        memory=mem, llm=llm, session_id="shared", user_id="alice",
    )
    with pytest.raises(PermissionError):
        cm_alice.supersede(bob_fact_id, "Alice tries to edit Bob's fact")

    # Bob's fact must be untouched.
    bob_hit = mem.get(bob_fact_id)
    assert bob_hit is not None
    assert bob_hit.metadata.get("superseded_by") is None, (
        "failed supersede must not mutate the target fact"
    )

    # Alice CAN supersede her own fact.
    alice_fact_id = mem.store(
        "Alice used to live in Portland",
        metadata={"type": "fact", "session_id": "shared", "user_id": "alice"},
    )
    new_id = cm_alice.supersede(alice_fact_id, "Alice now lives in Boston")
    assert mem.get(alice_fact_id).metadata.get("superseded_by") == new_id


# ----------------------------------------------------------------------
# 7. clear_session() is filtered by user_id when set
# ----------------------------------------------------------------------
def test_clear_session_filtered_by_user_id_when_set() -> None:
    """cm(user=alice).clear_session() leaves Bob's entries untouched."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()

    alice_turn_id = mem.store(
        "alice says hi",
        metadata={
            "type": "turn", "session_id": "shared", "user_id": "alice",
            "role": "user",
        },
    )
    alice_fact_id = mem.store(
        "Alice likes hiking",
        metadata={
            "type": "fact", "session_id": "shared", "user_id": "alice",
        },
    )
    bob_turn_id = mem.store(
        "bob says hi",
        metadata={
            "type": "turn", "session_id": "shared", "user_id": "bob",
            "role": "user",
        },
    )
    bob_fact_id = mem.store(
        "Bob likes hiking",
        metadata={
            "type": "fact", "session_id": "shared", "user_id": "bob",
        },
    )

    cm_alice = ConversationalMemory(
        memory=mem, llm=llm, session_id="shared", user_id="alice",
    )
    removed = cm_alice.clear_session()
    assert removed == 2, f"expected 2 removed (alice turn + fact), got {removed}"

    # Alice entries gone.
    assert mem.get(alice_turn_id) is None
    assert mem.get(alice_fact_id) is None
    # Bob entries still there.
    assert mem.get(bob_turn_id) is not None
    assert mem.get(bob_fact_id) is not None
