"""Tests for Phase 34: ``ConversationalMemory.forget(...)`` inventory API.

Contract (per ``docs/plans/2026-04-16-phase-34-forget-inventory.md``):

- ``forget(..., dry_run=True)`` returns a :class:`ForgetPreview`
  dataclass summarising what a deletion would touch — raw turns,
  derived facts, and overlapping summaries — without mutating the
  underlying :class:`MemoryLayer`.
- Three matcher modes, intersected when combined:
    - ``text_matches=pattern``: substring match over raw turn text
      (case-insensitive by default, ``case_sensitive=True`` to opt in).
    - ``subject=name``: equality on a fact's ``metadata["subject"]``.
    - ``user_id=uid``: equality on ``metadata["user_id"]`` across every
      entry; delegates to the Phase 12 multi-user scoping contract.
- Zero criteria → :class:`ValueError` (don't accidentally wipe).
- Derived-fact detection uses the Phase 25 ``source_turn_id`` back-
  pointer. Overlapping-summary detection uses the
  ``summarized_turn_start`` / ``summarized_turn_end`` range metadata
  that Phase 17 stamps on every summary write.

Phase 35 flipped ``dry_run`` to default to ``False`` and wired the
actual deletion path; the preview tests here always pass
``dry_run=True`` explicitly so they exercise only the inventory
behaviour. The delete-path contract lives in
``tests/test_memory/test_conversational_forget_delete.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory, ForgetPreview


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic per-text vector."""
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


@dataclass
class _ScriptedLLM:
    """Minimal scripted LLM covering extract / reconcile / summary prompts."""

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


def _make_cm(
    *,
    user_id: str | None = None,
    extract_replies: list[str] | None = None,
    summary_every: int = 1000,  # default off — specific tests turn it on
) -> tuple[ConversationalMemory, MemoryLayer]:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=extract_replies or [],
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        user_id=user_id,
        summary_every=summary_every,
    )
    return cm, mem


# ----------------------------------------------------------------------
# Sanity: import surface
# ----------------------------------------------------------------------
def test_forget_preview_is_exposed_from_conversational() -> None:
    """``ForgetPreview`` must be importable from ``soma.memory.conversational``."""
    # import side of this is the module-level import at the top.
    preview = ForgetPreview(
        raw_turns=[], derived_facts=[], summaries=[], total_vectors=0,
    )
    assert preview.is_empty() is True


# ----------------------------------------------------------------------
# 1. Criterion: text_matches — case-insensitive default
# ----------------------------------------------------------------------
def test_forget_preview_text_match_returns_matching_turns() -> None:
    """Substring match over raw turn text; facts from those turns ride along."""
    cm, _ = _make_cm(
        extract_replies=[
            json.dumps([_fact("preference", "User loves gardening")]),
            json.dumps([]),
            json.dumps([_fact("preference", "User gardens on weekends")]),
        ],
    )
    cm.add_message("user", "I love gardening")
    cm.add_message("user", "Coffee is good")
    cm.add_message("user", "Gardening on weekends is my hobby")

    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert isinstance(preview, ForgetPreview)
    assert len(preview.raw_turns) == 2, (
        f"expected 2 turns matching 'gardening', got {len(preview.raw_turns)}"
    )
    # Some facts were extracted from those turns — at least one should
    # trace back via source_turn_id.
    assert preview.derived_facts, (
        "expected at least one derived fact with source_turn_id in the match set"
    )
    assert preview.is_empty() is False


def test_forget_preview_case_insensitive_by_default() -> None:
    """text_matches ignores case unless case_sensitive=True."""
    cm, _ = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "I LOVE Gardening")
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert len(preview.raw_turns) == 1


def test_forget_preview_case_sensitive_optional() -> None:
    """case_sensitive=True tightens the match and can miss when case differs."""
    cm, _ = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "I LOVE Gardening")
    preview = cm.forget(text_matches="gardening", case_sensitive=True, dry_run=True)
    assert preview.is_empty()
    # Sanity: the capital-G variant still matches in case-sensitive mode.
    preview2 = cm.forget(text_matches="Gardening", case_sensitive=True, dry_run=True)
    assert len(preview2.raw_turns) == 1


# ----------------------------------------------------------------------
# 2. Criterion: subject — equality on metadata["subject"]
# ----------------------------------------------------------------------
def test_forget_preview_subject_match() -> None:
    """subject= matches facts whose metadata carries that subject field."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    # Hand-stamp a fact with a subject field so the matcher has something
    # to hit. In production a future extractor phase fills this; for now
    # callers can stamp it via add_message metadata or direct store().
    mem.store(
        "Alice likes mushrooms",
        metadata={
            "type": "fact",
            "session_id": "s",
            "subject": "alice",
            "category": "preference",
        },
    )
    mem.store(
        "Bob likes hiking",
        metadata={
            "type": "fact",
            "session_id": "s",
            "subject": "bob",
            "category": "preference",
        },
    )
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")

    preview = cm.forget(subject="alice", dry_run=True)
    assert len(preview.derived_facts) == 1
    # Alice's entry came back; Bob's did not.
    alice_hit = mem.get(preview.derived_facts[0])
    assert alice_hit is not None
    assert alice_hit.metadata.get("subject") == "alice"


# ----------------------------------------------------------------------
# 3. Criterion: user_id — delegates to Phase 12 scoping
# ----------------------------------------------------------------------
def test_forget_preview_user_id_match() -> None:
    """user_id= lists every entry owned by that user across categories."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")

    alice_turn = mem.store(
        "alice hello",
        metadata={"type": "turn", "session_id": "s", "user_id": "alice", "role": "user"},
    )
    alice_fact = mem.store(
        "Alice likes hiking",
        metadata={
            "type": "fact", "session_id": "s", "user_id": "alice",
            "source_turn_id": alice_turn,
        },
    )
    bob_turn = mem.store(
        "bob hello",
        metadata={"type": "turn", "session_id": "s", "user_id": "bob", "role": "user"},
    )
    bob_fact = mem.store(
        "Bob likes hiking",
        metadata={
            "type": "fact", "session_id": "s", "user_id": "bob",
            "source_turn_id": bob_turn,
        },
    )

    preview = cm.forget(user_id="alice", dry_run=True)
    assert alice_turn in preview.raw_turns
    assert bob_turn not in preview.raw_turns
    assert alice_fact in preview.derived_facts
    assert bob_fact not in preview.derived_facts


# ----------------------------------------------------------------------
# 4. Intersection: multiple criteria are AND-composed
# ----------------------------------------------------------------------
def test_forget_preview_intersection_of_criteria() -> None:
    """text_matches + user_id narrows to turns satisfying both."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s")

    alice_turn = mem.store(
        "gardening",
        metadata={"type": "turn", "session_id": "s", "user_id": "alice", "role": "user"},
    )
    bob_turn = mem.store(
        "gardening",
        metadata={"type": "turn", "session_id": "s", "user_id": "bob", "role": "user"},
    )
    alice_other = mem.store(
        "coffee",
        metadata={"type": "turn", "session_id": "s", "user_id": "alice", "role": "user"},
    )

    preview = cm.forget(text_matches="gardening", user_id="alice", dry_run=True)
    assert preview.raw_turns == [alice_turn]
    assert bob_turn not in preview.raw_turns
    assert alice_other not in preview.raw_turns


# ----------------------------------------------------------------------
# 5. Safety: zero criteria raises
# ----------------------------------------------------------------------
def test_forget_preview_no_criteria_raises() -> None:
    """forget() with no criteria must refuse — don't accidentally wipe everything."""
    cm, _ = _make_cm()
    with pytest.raises(ValueError, match="at least one criterion"):
        cm.forget()


# ----------------------------------------------------------------------
# 6. No match → empty preview (not an error)
# ----------------------------------------------------------------------
def test_forget_preview_empty_result_returns_empty_preview() -> None:
    """A miss returns an empty ForgetPreview rather than raising."""
    cm, _ = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "hello world")
    preview = cm.forget(text_matches="nonexistent", dry_run=True)
    assert preview.is_empty()
    assert preview.total_vectors == 0


# ----------------------------------------------------------------------
# 7. Explicit dry_run=True: no writes, no mutations
# ----------------------------------------------------------------------
def test_forget_dry_run_true_no_writes() -> None:
    """dry_run=True must not delete anything from the MemoryLayer.

    Phase 34 defaulted ``dry_run`` to True; Phase 35 flipped it to
    False, so the preview path now requires an explicit opt-in. The
    no-writes guarantee still holds for that explicit path.
    """
    cm, mem = _make_cm(
        extract_replies=[
            json.dumps([_fact("preference", "User loves gardening")]),
        ],
    )
    cm.add_message("user", "I love gardening")
    baseline = len(mem)
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert not preview.is_empty(), "sanity: preview found something"
    assert len(mem) == baseline, (
        f"dry_run must not mutate; len went {baseline} -> {len(mem)}"
    )


# ----------------------------------------------------------------------
# 9. Summaries that overlap the matched turn range are included
# ----------------------------------------------------------------------
def test_forget_preview_includes_overlapping_summaries() -> None:
    """Summaries whose [start, end] turn range intersects matched turns come along."""
    cm, mem = _make_cm(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_every=3,
    )
    # Turns 0-2 about gardening, turns 3-5 about coffee, turns 6-8 about travel.
    cm.add_message("user", "gardening is great")
    cm.add_message("user", "gardening on weekends")
    cm.add_message("user", "gardening tools")       # rolls summary covering 0..2
    cm.add_message("user", "coffee is good")
    cm.add_message("user", "coffee in the morning")
    cm.add_message("user", "coffee shops nearby")   # rolls summary covering 3..5
    cm.add_message("user", "travelling to Japan")
    cm.add_message("user", "travelling is fun")
    cm.add_message("user", "travelling abroad")     # rolls summary covering 6..8

    # All three summaries should be stored.
    summaries_all = [
        mem.get(nid)
        for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "summary"
    ]
    assert len(summaries_all) == 3

    preview = cm.forget(text_matches="gardening", dry_run=True)
    # All three gardening turns matched.
    assert len(preview.raw_turns) == 3
    # Exactly one summary overlaps — the one covering turn indices 0..2.
    assert len(preview.summaries) == 1
    matched_summary = mem.get(preview.summaries[0])
    assert matched_summary is not None
    assert matched_summary.metadata.get("summarized_turn_start") == 0
    assert matched_summary.metadata.get("summarized_turn_end") == 2


# ----------------------------------------------------------------------
# 10. total_vectors accounting
# ----------------------------------------------------------------------
def test_forget_preview_counts_total_vectors() -> None:
    """total_vectors equals raw_turns + derived_facts + summaries."""
    cm, _ = _make_cm(
        extract_replies=[
            json.dumps([_fact("preference", "User loves gardening")]),
        ],
        summary_every=1000,
    )
    cm.add_message("user", "I love gardening")
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert preview.total_vectors == (
        len(preview.raw_turns)
        + len(preview.derived_facts)
        + len(preview.summaries)
    )
    assert preview.total_vectors >= 2  # 1 turn + >=1 fact


# ----------------------------------------------------------------------
# 11. user_id respects Phase 12 scoping — no cross-user leak
# ----------------------------------------------------------------------
def test_forget_preview_user_id_scopes_text_matches() -> None:
    """text_matches + user_id must never return another user's turns."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM()
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s", user_id="alice",
    )
    cm.add_message("user", "alice says gardening")
    cm.add_message("user", "bob says gardening", user_id="bob")

    # Constructor user_id=alice; a text-only forget still sees both
    # because the scoping is on the user_id criterion, not the wrapper.
    all_preview = cm.forget(text_matches="gardening", dry_run=True)
    assert len(all_preview.raw_turns) == 2

    # But adding user_id=alice as a criterion excludes bob.
    scoped = cm.forget(
        text_matches="gardening", user_id="alice", dry_run=True,
    )
    assert len(scoped.raw_turns) == 1
    only_hit = mem.get(scoped.raw_turns[0])
    assert only_hit is not None
    assert only_hit.metadata.get("user_id") == "alice"


# ----------------------------------------------------------------------
# 12. ForgetPreview.is_empty semantics
# ----------------------------------------------------------------------
def test_forget_preview_is_empty_true_when_all_lists_empty() -> None:
    preview = ForgetPreview(
        raw_turns=[], derived_facts=[], summaries=[], total_vectors=0,
    )
    assert preview.is_empty() is True


def test_forget_preview_is_empty_false_when_anything_present() -> None:
    preview = ForgetPreview(
        raw_turns=["x"], derived_facts=[], summaries=[], total_vectors=1,
    )
    assert preview.is_empty() is False
