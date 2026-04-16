"""Tests for Phase 35: ``ConversationalMemory.forget(..., dry_run=False)``.

Phase 34 shipped the read-only :class:`ForgetPreview` path. Phase 35
wires the actual deletion:

- ``dry_run`` default flips to ``False`` — plain ``cm.forget(...)`` now
  deletes. Callers who want the read-only inventory must pass
  ``dry_run=True`` explicitly.
- ``dry_run=False`` returns a :class:`ForgetResult` — deleted turn /
  fact / (Phase 36) summary ids and a ``total_deleted`` count. The
  ``deleted_summaries`` field is always ``[]`` in Phase 35 but the
  field is declared now so Phase 36 doesn't churn the dataclass.
- Deletion order is facts-first then turns: facts reference their
  source turn via ``metadata["source_turn_id"]``; dropping turns
  first would briefly orphan the fact records.
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
    """``ForgetResult`` must be importable; Phase 36 populates summaries."""
    result = ForgetResult(
        deleted_turns=["t1"],
        deleted_facts=["f1"],
        deleted_summaries=[],
        total_deleted=2,
    )
    assert result.deleted_turns == ["t1"]
    assert result.deleted_facts == ["f1"]
    assert result.deleted_summaries == []
    assert result.total_deleted == 2


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
# 7. deleted_summaries is always [] in Phase 35 (cascade arrives in 36)
# ----------------------------------------------------------------------
def test_forget_summaries_field_empty_in_phase_35() -> None:
    """Summaries field is declared but always empty until Phase 36 wires rewrite."""
    cm, mem = _make_cm(
        extract_replies=[json.dumps([]) for _ in range(10)],
        summary_every=2,
    )
    cm.add_message("user", "gardening one")
    cm.add_message("user", "gardening two")  # rolls a summary
    # The preview path would include the summary; the delete path
    # explicitly does NOT drop summaries in Phase 35.
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert preview.summaries, "sanity: a summary overlaps the matched turns"
    # Snapshot the summary ids the preview flagged. Phase 35 deletes
    # turns + facts only; the summary entries themselves must survive.
    summary_ids_before = set(preview.summaries)
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert result.deleted_summaries == []
    # Every flagged summary id is still resolvable in the store —
    # Phase 35 deliberately leaves them alone. Phase 36 will rewrite
    # them in place (source turns gone → summary shrinks accordingly).
    assert all(sid in mem for sid in summary_ids_before), (
        "Phase 35 must not delete summaries; they survive for Phase 36 to rewrite"
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
