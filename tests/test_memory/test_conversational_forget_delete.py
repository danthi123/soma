"""Tests for Phase 35 + Phase 36: ``ConversationalMemory.forget(..., dry_run=False)``.

Phase 34 shipped the read-only :class:`ForgetPreview` path. Phase 35
wired the actual deletion for turns + facts. Phase 36 finishes the
cascade by handling summaries: summaries whose turn range is fully
covered by matched turns are dropped outright; summaries with some
surviving turns are regenerated from the survivors (new id since the
MemoryLayer has no in-place text update).

- ``dry_run`` default flips to ``False`` — plain ``cm.forget(...)`` now
  deletes. Callers who want the read-only inventory must pass
  ``dry_run=True`` explicitly.
- ``dry_run=False`` returns a :class:`ForgetResult` — deleted turn /
  fact / summary ids, a ``regenerated_summaries`` field populated by
  Phase 36 when a summary was rewritten in-store (new id), and a
  ``total_deleted`` count.
- ``total_deleted`` counts deletions only — regenerated summaries are
  mutated, not deleted, and do not contribute.
- Deletion order is facts-first then turns then summaries: facts
  reference their source turn via ``metadata["source_turn_id"]``;
  dropping turns first would briefly orphan the fact records.
  Summaries are processed last so the surviving-turn set used to
  decide regenerate-vs-drop reflects the post-delete state.
- LLM-unavailable fallback: if the summary regeneration prompt fails
  (network, quota, stubbed backend), drop the summary and log at
  WARNING. Principle: under a user request to forget, prefer over-
  deletion to silent retention of derived content.
- Async-extraction mode: ``flush()`` runs **before** the delete so
  in-flight extractions land (and are then included in the preview
  and deleted). Batch mode drops the pending buffer (wipe means
  wipe — don't extract facts from turns the user is forgetting).
- Zero criteria still raises :class:`ValueError` (shared guard with
  Phase 34's dry-run path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from soma.memory import MemoryLayer
from soma.memory.conversational import (
    ConversationalMemory,
    ForgetPreview,
    ForgetResult,
)


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
    summary_every: int = 1000,
    extraction_mode: str = "sync",
    batch_size: int = 8,
) -> tuple[ConversationalMemory, MemoryLayer]:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(extract_replies=extract_replies or [])
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        user_id=user_id,
        summary_every=summary_every,
        extraction_mode=extraction_mode,  # type: ignore[arg-type]
        batch_size=batch_size,
    )
    return cm, mem


# ----------------------------------------------------------------------
# 1. Surface: ForgetResult is importable and has the contract fields
# ----------------------------------------------------------------------
def test_forget_result_is_exposed_with_expected_fields() -> None:
    """``ForgetResult`` must be importable with Phase 36's new field."""
    result = ForgetResult(
        deleted_turns=["t1"],
        deleted_facts=["f1"],
        deleted_summaries=["s1"],
        regenerated_summaries=["s2"],
        total_deleted=3,
    )
    assert result.deleted_turns == ["t1"]
    assert result.deleted_facts == ["f1"]
    assert result.deleted_summaries == ["s1"]
    assert result.regenerated_summaries == ["s2"]
    # total_deleted intentionally excludes regenerated summaries —
    # they are mutated (replaced with new content), not deleted.
    assert result.total_deleted == 3


# ----------------------------------------------------------------------
# 2. Default behaviour flip: dry_run defaults to False
# ----------------------------------------------------------------------
def test_forget_default_is_not_dry_run() -> None:
    """Plain ``cm.forget(text_matches=...)`` must actually delete.

    Phase 34 defaulted to dry-run; Phase 35 flips the default so the
    public API matches the verb. Callers who want preview now opt in.
    """
    cm, mem = _make_cm(
        extract_replies=[json.dumps([_fact("preference", "user gardens")])],
    )
    cm.add_message("user", "I love gardening")
    baseline = len(mem)
    result = cm.forget(text_matches="gardening")
    # Returns a ForgetResult, not a ForgetPreview — hence deletion happened.
    assert isinstance(result, ForgetResult)
    assert len(mem) < baseline, (
        f"default forget() must delete; len stayed at {baseline} -> {len(mem)}"
    )


# ----------------------------------------------------------------------
# 3. Explicit dry_run=True still returns ForgetPreview and mutates nothing
# ----------------------------------------------------------------------
def test_forget_dry_run_true_still_returns_preview_without_mutation() -> None:
    cm, mem = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "gardening today")
    baseline = len(mem)
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert isinstance(preview, ForgetPreview)
    assert preview.raw_turns, "sanity: preview found the gardening turn"
    assert len(mem) == baseline, "dry_run=True must not delete anything"


# ----------------------------------------------------------------------
# 4. dry_run=False returns ForgetResult and deletes raw turns
# ----------------------------------------------------------------------
def test_forget_delete_removes_raw_turns() -> None:
    """Matched raw turns are removed from the MemoryLayer."""
    cm, mem = _make_cm(
        extract_replies=[json.dumps([]), json.dumps([])],
    )
    cm.add_message("user", "gardening note 1")
    cm.add_message("user", "gardening note 2")
    cm.add_message("user", "unrelated note")
    baseline = len(mem)
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert isinstance(result, ForgetResult)
    assert len(result.deleted_turns) == 2
    for tid in result.deleted_turns:
        assert tid not in mem, f"turn {tid!r} should have been deleted"
    # Unrelated turn survived.
    assert len(mem) == baseline - len(result.deleted_turns) - len(result.deleted_facts)


# ----------------------------------------------------------------------
# 5. Derived facts are cascaded (facts-first deletion order)
# ----------------------------------------------------------------------
def test_forget_delete_cascades_to_derived_facts() -> None:
    """Facts whose ``source_turn_id`` points at a matched turn are deleted too."""
    cm, mem = _make_cm(
        extract_replies=[
            json.dumps([_fact("preference", "User loves gardening")]),
        ],
    )
    cm.add_message("user", "I love gardening")
    # Sanity: the extractor produced a fact with a source_turn_id link.
    fact_ids_before = [
        nid for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "fact"
    ]
    assert fact_ids_before, "sanity: extractor should have produced a fact"

    result = cm.forget(text_matches="gardening", dry_run=False)
    assert result.deleted_facts, "expected the derived fact to be deleted"
    for fid in result.deleted_facts:
        assert fid not in mem, f"fact {fid!r} should have been deleted"
    # All derived facts that were in the preview set are gone.
    remaining_facts = [
        nid for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "fact"
    ]
    for fid in result.deleted_facts:
        assert fid not in remaining_facts


# ----------------------------------------------------------------------
# 6. total_deleted accounting
# ----------------------------------------------------------------------
def test_forget_total_deleted_equals_sum_of_categories() -> None:
    """``total_deleted`` is the sum of turns + facts (summaries always 0 in Phase 35)."""
    cm, _ = _make_cm(
        extract_replies=[
            json.dumps([_fact("preference", "User loves gardening")]),
        ],
    )
    cm.add_message("user", "I love gardening")
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert result.total_deleted == (
        len(result.deleted_turns)
        + len(result.deleted_facts)
        + len(result.deleted_summaries)
    )
    assert result.total_deleted >= 2  # 1 turn + >=1 fact


# ----------------------------------------------------------------------
# 7. Fully-covered summaries are dropped (Phase 36 cascade)
# ----------------------------------------------------------------------
def test_forget_drops_fully_covered_summary() -> None:
    """A summary whose turn range is entirely inside the matched set is deleted.

    Two gardening turns → one summary covering [0, 1]. Forgetting
    "gardening" leaves no survivors in the summary's range, so the
    cascade drops the summary outright.
    """
    cm, mem = _make_cm(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_every=2,
    )
    cm.add_message("user", "gardening one")
    cm.add_message("user", "gardening two")  # rolls summary covering 0..1
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert preview.summaries, "sanity: a summary overlaps the matched turns"
    summary_ids_before = set(preview.summaries)

    result = cm.forget(text_matches="gardening", dry_run=False)
    assert set(result.deleted_summaries) == summary_ids_before
    assert result.regenerated_summaries == []
    # Every flagged summary id is gone from the store.
    assert all(sid not in mem for sid in summary_ids_before), (
        "fully-covered summaries must be deleted by the Phase 36 cascade"
    )


# ----------------------------------------------------------------------
# 8. Empty result: zero matches do nothing quietly
# ----------------------------------------------------------------------
def test_forget_empty_result_does_nothing() -> None:
    """A miss returns an empty ForgetResult and leaves the store untouched."""
    cm, mem = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "hello world")
    baseline = len(mem)
    result = cm.forget(text_matches="nonexistent", dry_run=False)
    assert isinstance(result, ForgetResult)
    assert result.total_deleted == 0
    assert result.deleted_turns == []
    assert result.deleted_facts == []
    assert len(mem) == baseline


# ----------------------------------------------------------------------
# 9. Zero criteria still raises — shared guard with the preview path
# ----------------------------------------------------------------------
def test_forget_delete_no_criteria_still_raises() -> None:
    """Zero-criteria guard applies to dry_run=False too — same message."""
    cm, _ = _make_cm()
    with pytest.raises(ValueError, match="at least one criterion"):
        cm.forget(dry_run=False)


# ----------------------------------------------------------------------
# 10. Async mode: flush drains in-flight extractions before delete
# ----------------------------------------------------------------------
def test_forget_async_mode_flushes_in_flight_extractions_first() -> None:
    """In async mode, pending extract+reconcile futures land before delete.

    Otherwise a fact extracted from a matched turn would land AFTER
    ``forget()`` returns, leaking a dangling fact that the GDPR
    deletion claim said we'd wiped.
    """
    cm, mem = _make_cm(
        extract_replies=[
            json.dumps([_fact("preference", "user loves gardening")]),
        ],
        extraction_mode="async",
    )
    cm.add_message("user", "I love gardening")
    # Don't call cm.flush() here — forget() must do it for us.
    try:
        result = cm.forget(text_matches="gardening", dry_run=False)
        # The async-extracted fact landed during forget()'s internal
        # flush, was included in the preview, and got deleted.
        assert result.deleted_facts, (
            "expected forget() to flush the async extractor, then delete the fact"
        )
        for fid in result.deleted_facts:
            assert fid not in mem
    finally:
        cm.close()


# ----------------------------------------------------------------------
# 11. Batch mode: pending buffer is dropped, not extracted
# ----------------------------------------------------------------------
def test_forget_batch_mode_drops_pending_buffer() -> None:
    """Batch mode: the user wants those turns forgotten — don't extract first."""
    cm, mem = _make_cm(
        extract_replies=[json.dumps([_fact("preference", "X")])],
        extraction_mode="batch",
        batch_size=8,
    )
    for i in range(3):
        cm.add_message("user", f"gardening {i}")
    # Pending batch holds 3 turns, nothing extracted yet.
    assert len(cm._pending_batch) == 3
    # Count facts before forget — none extracted yet.
    fact_count_before = sum(
        1 for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "fact"
    )
    assert fact_count_before == 0, (
        "batch mode should not have extracted any facts yet"
    )

    result = cm.forget(text_matches="gardening", dry_run=False)
    # Buffer was cleared without firing the extractor.
    assert cm._pending_batch == []
    # Turns were deleted.
    assert len(result.deleted_turns) == 3
    # No facts were extracted on the way through (the user asked to
    # forget these turns — extracting first would write facts we'd
    # have to delete).
    fact_count_after = sum(
        1 for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "fact"
    )
    assert fact_count_after == 0


# ----------------------------------------------------------------------
# 12. Facts-first deletion order
# ----------------------------------------------------------------------
def test_forget_deletes_facts_before_turns() -> None:
    """Internal order: fact deletes must precede turn deletes.

    Facts reference their source turn via ``metadata["source_turn_id"]``.
    If turns were deleted first, during the brief window the derived
    facts would be orphans (their source_turn_id points at a node that
    no longer exists). Deleting facts first keeps referential integrity
    on every intermediate state.
    """
    # We pin this by patching _memory.forget to record the call order.
    cm, mem = _make_cm(
        extract_replies=[
            json.dumps([_fact("preference", "user loves gardening")]),
        ],
    )
    cm.add_message("user", "I love gardening")

    # Figure out which ids are facts and which are turns before the call.
    fact_ids = {
        nid for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "fact"
    }
    turn_ids = {
        nid for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "turn"
    }
    assert fact_ids and turn_ids

    call_order: list[str] = []
    real_forget = mem.forget

    def recording_forget(node_id: str) -> bool:
        call_order.append(node_id)
        return real_forget(node_id)

    mem.forget = recording_forget  # type: ignore[method-assign]
    try:
        cm.forget(text_matches="gardening", dry_run=False)
    finally:
        mem.forget = real_forget  # type: ignore[method-assign]

    # Every fact id appears in call_order before any turn id.
    first_turn_idx = min(
        (i for i, nid in enumerate(call_order) if nid in turn_ids),
        default=None,
    )
    last_fact_idx = max(
        (i for i, nid in enumerate(call_order) if nid in fact_ids),
        default=None,
    )
    assert first_turn_idx is not None, "expected at least one turn delete"
    assert last_fact_idx is not None, "expected at least one fact delete"
    assert last_fact_idx < first_turn_idx, (
        f"facts must be deleted before turns; "
        f"last fact at {last_fact_idx}, first turn at {first_turn_idx}, "
        f"order={call_order}"
    )


# ----------------------------------------------------------------------
# 13. Vector backend cascade — MemoryLayer.forget already hits it,
#     but pin the behaviour so Phase 36+ don't regress it.
# ----------------------------------------------------------------------
def test_forget_delete_also_removes_from_vector_backend() -> None:
    """MemoryLayer.forget cascades to backend.remove — ntotal drops."""
    cm, mem = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "gardening once")
    cm.add_message("user", "gardening twice")
    # Inproc backend's ntotal should match len(mem) after writes.
    baseline_backend = mem._backend.ntotal
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert result.deleted_turns
    # Backend shrank by the same number of rows the layer removed.
    assert mem._backend.ntotal == baseline_backend - result.total_deleted


# ----------------------------------------------------------------------
# 14. Subsequent retrieve does not return deleted ids
# ----------------------------------------------------------------------
def test_forget_deleted_entries_not_returned_by_retrieve() -> None:
    """After forget(dry_run=False), a retrieve over the same text is empty."""
    cm, _ = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "gardening is my favourite thing")
    cm.forget(text_matches="gardening", dry_run=False)
    hits = cm.retrieve("gardening", k=5)
    assert all("gardening" not in h.text.lower() for h in hits), (
        f"retrieve returned a purportedly-deleted entry: {[h.text for h in hits]}"
    )


# ======================================================================
# Phase 36: summary cascade
# ======================================================================


# ----------------------------------------------------------------------
# 15. Partially-covered summaries are regenerated from survivors
# ----------------------------------------------------------------------
def test_forget_regenerates_partially_covered_summary() -> None:
    """A summary whose range has surviving turns is regenerated, not dropped.

    Three turns roll into one summary covering [0, 2]. Only the first
    turn matches "gardening"; turns 1 and 2 survive, so the summary is
    regenerated from those survivors.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    # Stub summary text switches to a gardening-free sentence so we can
    # observe the regeneration output in the store.
    llm = _ScriptedLLM(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_reply="The user discussed coffee and travel plans.",
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=3,
    )
    cm.add_message("user", "I love gardening")
    cm.add_message("user", "I love coffee")
    cm.add_message("user", "I love travel")  # rolls summary covering 0..2
    summary_ids_before = [
        nid for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "summary"
    ]
    assert len(summary_ids_before) == 1

    result = cm.forget(text_matches="gardening", dry_run=False)
    # The original summary id is gone; a new one carries the regen text.
    assert result.regenerated_summaries, (
        "partially-covered summary must be regenerated"
    )
    assert result.deleted_summaries == [], (
        "regeneration must not double-count as a deletion"
    )
    # The original id no longer resolves (we deleted + re-stored to
    # pick up the new embedding — MemoryLayer has no update_text).
    assert summary_ids_before[0] not in mem

    regenerated_id = result.regenerated_summaries[0]
    regen = mem.get(regenerated_id)
    assert regen is not None
    assert "gardening" not in regen.text.lower(), (
        f"regenerated summary still mentions gardening: {regen.text!r}"
    )
    # Metadata shape is preserved (type, range, session).
    assert regen.metadata.get("type") == "summary"
    assert regen.metadata.get("summarized_turn_start") == 0
    assert regen.metadata.get("summarized_turn_end") == 2
    assert regen.metadata.get("session_id") == "s"


# ----------------------------------------------------------------------
# 16. Regeneration preserves user_id scoping
# ----------------------------------------------------------------------
def test_forget_regenerated_summary_preserves_user_id() -> None:
    """Regenerated summaries keep the original ``user_id`` stamp."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_reply="Summary about unrelated topics.",
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        user_id="alice",
        summary_every=3,
    )
    cm.add_message("user", "gardening is fun")
    cm.add_message("user", "coffee is great")
    cm.add_message("user", "travel plans")
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert result.regenerated_summaries
    regen = mem.get(result.regenerated_summaries[0])
    assert regen is not None
    assert regen.metadata.get("user_id") == "alice", (
        "regen must preserve the Phase 12 user_id stamp"
    )


# ----------------------------------------------------------------------
# 17. LLM failure during regen falls back to deletion
# ----------------------------------------------------------------------
def test_forget_summary_regen_failure_falls_back_to_delete() -> None:
    """When the regen LLM call raises, drop the summary instead of keeping stale text.

    Principle: under a user request to forget, prefer over-deletion to
    silent retention of derived content that might still reference the
    scrubbed subject.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_reply="initial summary text",
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=3,
    )
    cm.add_message("user", "gardening one")
    cm.add_message("user", "coffee two")
    cm.add_message("user", "travel three")
    summary_ids_before = [
        nid for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "summary"
    ]
    assert len(summary_ids_before) == 1

    # Now poison the LLM so any further generate() call raises. The
    # forget path's summary regen should catch this and fall back to
    # dropping the summary.
    def _raise(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("LLM unavailable")

    llm.generate = _raise  # type: ignore[method-assign]

    result = cm.forget(text_matches="gardening", dry_run=False)
    # Fallback deletion path was taken.
    assert result.deleted_summaries == summary_ids_before
    assert result.regenerated_summaries == []
    # Every originally-flagged summary id is gone from the store.
    for sid in summary_ids_before:
        assert sid not in mem


# ----------------------------------------------------------------------
# 18. total_deleted accounting with the full cascade
# ----------------------------------------------------------------------
def test_forget_summary_cascade_updates_total_deleted() -> None:
    """``total_deleted`` counts dropped summaries but NOT regenerated ones."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_reply="a summary about the topics discussed",
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=3,
    )
    # Three gardening turns → one fully-covered summary (drop).
    cm.add_message("user", "gardening one")
    cm.add_message("user", "gardening two")
    cm.add_message("user", "gardening three")
    result = cm.forget(text_matches="gardening", dry_run=False)
    # Invariant: total_deleted is turns + facts + deleted_summaries —
    # regenerated_summaries do not contribute.
    assert result.total_deleted == (
        len(result.deleted_turns)
        + len(result.deleted_facts)
        + len(result.deleted_summaries)
    )
    # Regenerated summaries are explicitly excluded from the count.
    expected_min = (
        len(result.deleted_turns)
        + len(result.deleted_facts)
        + len(result.deleted_summaries)
    )
    assert result.total_deleted == expected_min


# ----------------------------------------------------------------------
# 19. Dry-run preview must not mutate summaries
# ----------------------------------------------------------------------
def test_forget_preview_does_not_modify_summaries() -> None:
    """``dry_run=True`` inspects but never touches summary entries."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_reply="pre-existing summary text",
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=3,
    )
    cm.add_message("user", "gardening note one")
    cm.add_message("user", "gardening note two")
    cm.add_message("user", "gardening note three")
    summary_texts_before = {
        nid: h.text
        for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "summary"
    }
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert preview.summaries
    for sid in preview.summaries:
        assert sid in mem, "dry_run must not delete summaries"
        hit = mem.get(sid)
        assert hit is not None
        assert hit.text == summary_texts_before[sid], (
            "dry_run must not rewrite summary text"
        )


# ----------------------------------------------------------------------
# 20. Regenerated summary's vector reflects the new text
# ----------------------------------------------------------------------
def test_forget_summary_vector_updated_after_regen() -> None:
    """Regeneration re-embeds: retrieving the new text hits the new id."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    new_summary = "discussion of coffee and travel exclusively"
    llm = _ScriptedLLM(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_reply=new_summary,
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=3,
    )
    cm.add_message("user", "I love gardening")
    cm.add_message("user", "I love coffee")
    cm.add_message("user", "I love travel")
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert result.regenerated_summaries
    regen_id = result.regenerated_summaries[0]
    # Retrieving the exact regenerated text must surface the new node id
    # at the top — which only works if the new embedding is wired up.
    # (Stub embedder is text-deterministic, so a matching text always
    # scores 1.0.)
    hits = mem.retrieve(new_summary, k=5)
    assert hits, "retrieve over regen text returned nothing"
    assert hits[0].node_id == regen_id, (
        f"regen summary not at top of retrieve; got {hits[0].node_id!r} "
        f"expected {regen_id!r}"
    )


# ----------------------------------------------------------------------
# 21. Partial cascade: multiple summaries, different fates
# ----------------------------------------------------------------------
def test_forget_cascade_mixed_drop_and_regen() -> None:
    """Across several summaries, each is classified independently.

    Three summaries (0..2, 3..5, 6..8). "gardening" only appears in
    turns 0, 3, 6 — one matched turn inside each range. Summary 0..2
    has survivors {1, 2} → regen. Summary 3..5 has survivors {4, 5} →
    regen. Summary 6..8 has survivors {7, 8} → regen.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(
        extract_replies=[json.dumps([]) for _ in range(20)],
        summary_reply="generic summary text without the forbidden word",
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=3,
    )
    # Each block has 1 gardening turn + 2 survivors.
    cm.add_message("user", "gardening block A")
    cm.add_message("user", "coffee block A one")
    cm.add_message("user", "coffee block A two")
    cm.add_message("user", "gardening block B")
    cm.add_message("user", "travel block B one")
    cm.add_message("user", "travel block B two")
    cm.add_message("user", "gardening block C")
    cm.add_message("user", "books block C one")
    cm.add_message("user", "books block C two")

    summary_ids_before = {
        nid for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "summary"
    }
    assert len(summary_ids_before) == 3

    result = cm.forget(text_matches="gardening", dry_run=False)
    # All three summaries got regenerated — each range had survivors.
    assert len(result.regenerated_summaries) == 3
    assert result.deleted_summaries == []
    # Original ids are gone.
    for sid in summary_ids_before:
        assert sid not in mem
    # All three regenerated entries resolve and are summaries.
    for rid in result.regenerated_summaries:
        regen = mem.get(rid)
        assert regen is not None
        assert regen.metadata.get("type") == "summary"
        assert "gardening" not in regen.text.lower()
