"""Tests for Phase 37: ``ForgetAuditSink`` + JSONL audit trail.

Phase 37 closes the GDPR-forgetting track by logging every
``forget()`` call (dry-run or live) as one JSON line so operators can
answer "who forgot what, when?" without re-instrumenting their
bundles. ``ForgetAuditSink`` is the append-only JSONL writer; the
Conversational memory wrapper calls it from the ``forget()`` path.

Covered here:

- Path-driven writes: ``SOMA_FORGET_AUDIT_PATH`` activates the sink.
- Opt-out: ``SOMA_FORGET_AUDIT_DISABLE=1`` returns a no-op sink even
  when the path is set, so operators who log via an external pipeline
  can keep the path plumbing without double-writing.
- No-op when the path is unset (default behaviour).
- Shape: records carry ``ts`` (ISO-8601 UTC with trailing ``Z``),
  ``user_id``, ``target_user_id``, ``criteria`` (dict), ``dry_run``
  flag, and a ``result`` block mirroring either a
  :class:`ForgetPreview` (dry-run) or :class:`ForgetResult` (live).
- Failure isolation: an unwritable path logs at WARNING but never
  raises — the audit failing is a lesser harm than blocking the
  user's delete.
- Integration: :meth:`ConversationalMemory.forget` emits exactly one
  record per call, including on dry-run, and on zero-match deletions.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch

from soma.forget_audit import ForgetAuditSink
from soma.memory import MemoryLayer
from soma.memory.conversational import (
    ConversationalMemory,
    ForgetPreview,
    ForgetResult,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _stub_embed(text: str) -> torch.Tensor:
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


@dataclass
class _ScriptedLLM:
    extract_replies: list[str] = field(default_factory=list)
    reconcile_replies: list[str] = field(default_factory=list)
    summary_reply: str = "rolling summary"
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


def _sample_result() -> ForgetResult:
    return ForgetResult(
        deleted_turns=["t1", "t2"],
        deleted_facts=["f1"],
        deleted_summaries=["s1"],
        regenerated_summaries=["s2"],
        total_deleted=4,
    )


def _sample_preview() -> ForgetPreview:
    return ForgetPreview(
        raw_turns=["t1", "t2"],
        derived_facts=["f1"],
        summaries=["s1"],
        total_vectors=4,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _make_cm(
    *,
    user_id: str | None = None,
    extract_replies: list[str] | None = None,
    summary_every: int = 1000,
    audit_sink: ForgetAuditSink | None = None,
) -> tuple[ConversationalMemory, MemoryLayer]:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = _ScriptedLLM(extract_replies=extract_replies or [])
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        user_id=user_id,
        summary_every=summary_every,
        audit_sink=audit_sink,
    )
    return cm, mem


# ----------------------------------------------------------------------
# 1. Env-driven construction
# ----------------------------------------------------------------------
def test_audit_from_env_writes_jsonl_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SOMA_FORGET_AUDIT_PATH", str(path))
    monkeypatch.delenv("SOMA_FORGET_AUDIT_DISABLE", raising=False)
    sink = ForgetAuditSink.from_env()
    sink.emit(
        user_id="alice",
        target_user_id=None,
        criteria={"text_matches": "gardening"},
        dry_run=False,
        result=_sample_result(),
    )
    records = _read_jsonl(path)
    assert len(records) == 1
    rec = records[0]
    assert rec["user_id"] == "alice"
    assert rec["target_user_id"] is None
    assert rec["criteria"] == {"text_matches": "gardening"}
    assert rec["dry_run"] is False


def test_audit_disabled_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SOMA_FORGET_AUDIT_PATH", str(path))
    monkeypatch.setenv("SOMA_FORGET_AUDIT_DISABLE", "1")
    sink = ForgetAuditSink.from_env()
    sink.emit(
        user_id="alice",
        target_user_id=None,
        criteria={},
        dry_run=False,
        result=_sample_result(),
    )
    assert not path.exists(), "disabled sink must not write anything"


def test_audit_none_path_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOMA_FORGET_AUDIT_PATH", raising=False)
    monkeypatch.delenv("SOMA_FORGET_AUDIT_DISABLE", raising=False)
    sink = ForgetAuditSink.from_env()
    # No-op; no file to check. Just ensure emit() returns without raising.
    sink.emit(
        user_id="alice",
        target_user_id=None,
        criteria={},
        dry_run=True,
        result=_sample_preview(),
    )


def test_audit_empty_path_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOMA_FORGET_AUDIT_PATH", "   ")
    sink = ForgetAuditSink.from_env()
    sink.emit(
        user_id="a",
        target_user_id=None,
        criteria={},
        dry_run=True,
        result=_sample_preview(),
    )


# ----------------------------------------------------------------------
# 2. Record shape
# ----------------------------------------------------------------------
def test_audit_record_has_iso8601_utc_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    sink.emit(
        user_id="alice",
        target_user_id=None,
        criteria={"text_matches": "x"},
        dry_run=False,
        result=_sample_result(),
    )
    rec = _read_jsonl(path)[0]
    ts = rec["ts"]
    # Ends with Z so downstream tooling can parse as UTC without a
    # separate tzinfo hint.
    assert ts.endswith("Z"), ts
    # Round-trip parseable.
    parsed = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_audit_result_shape_for_forget_result(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    sink.emit(
        user_id="ops",
        target_user_id="alice",
        criteria={"user_id": "alice"},
        dry_run=False,
        result=_sample_result(),
    )
    rec = _read_jsonl(path)[0]
    assert rec["target_user_id"] == "alice"
    res = rec["result"]
    assert res["deleted_turns"] == 2
    assert res["deleted_facts"] == 1
    assert res["deleted_summaries"] == 1
    assert res["regenerated_summaries"] == 1
    assert res["total_deleted"] == 4


def test_audit_result_shape_for_forget_preview(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    sink.emit(
        user_id="ops",
        target_user_id=None,
        criteria={"text_matches": "x"},
        dry_run=True,
        result=_sample_preview(),
    )
    rec = _read_jsonl(path)[0]
    # Preview surfaces *counts* of what would go — not ids. Audit is a
    # log, not a discovery tool; ops can replay the dry-run if they
    # need the id list.
    res = rec["result"]
    assert res["raw_turns"] == 2
    assert res["derived_facts"] == 1
    assert res["summaries"] == 1
    assert res["total_vectors"] == 4


def test_audit_dry_run_flag_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    sink.emit(
        user_id="alice",
        target_user_id=None,
        criteria={"text_matches": "x"},
        dry_run=True,
        result=_sample_preview(),
    )
    rec = _read_jsonl(path)[0]
    assert rec["dry_run"] is True


# ----------------------------------------------------------------------
# 3. Append-only behaviour
# ----------------------------------------------------------------------
def test_audit_appends_multiple_records(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    for i in range(3):
        sink.emit(
            user_id=f"user-{i}",
            target_user_id=None,
            criteria={"n": i},
            dry_run=False,
            result=_sample_result(),
        )
    records = _read_jsonl(path)
    assert len(records) == 3
    assert [r["user_id"] for r in records] == ["user-0", "user-1", "user-2"]


# ----------------------------------------------------------------------
# 4. Failure isolation
# ----------------------------------------------------------------------
def test_audit_path_errors_warn_but_dont_raise(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Point at a directory that doesn't exist and can't be created
    # because a file with the same name already blocks it. This makes
    # open() raise FileNotFoundError / OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    bad_path = blocker / "nested" / "audit.jsonl"
    sink = ForgetAuditSink(path=bad_path)
    with caplog.at_level(logging.WARNING, logger="soma.forget_audit"):
        # Must not raise — the audit path failing shouldn't block the
        # user's forget.
        sink.emit(
            user_id="alice",
            target_user_id=None,
            criteria={},
            dry_run=False,
            result=_sample_result(),
        )
    assert any(
        "audit" in rec.message.lower() or "forget_audit" in rec.name
        for rec in caplog.records
    )


# ----------------------------------------------------------------------
# 5. Integration: ConversationalMemory.forget() emits one record
# ----------------------------------------------------------------------
def test_cm_forget_emits_audit_record(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    cm, _ = _make_cm(audit_sink=sink, extract_replies=[json.dumps([])])
    cm.add_message("user", "gardening today")
    cm.forget(text_matches="gardening")
    records = _read_jsonl(path)
    assert len(records) == 1
    rec = records[0]
    assert rec["dry_run"] is False
    assert rec["criteria"]["text_matches"] == "gardening"


def test_cm_forget_dry_run_also_emits_audit(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    cm, _ = _make_cm(audit_sink=sink, extract_replies=[json.dumps([])])
    cm.add_message("user", "gardening today")
    cm.forget(text_matches="gardening", dry_run=True)
    records = _read_jsonl(path)
    assert len(records) == 1
    assert records[0]["dry_run"] is True


def test_cm_forget_no_match_still_emits_audit(tmp_path: Path) -> None:
    """Zero-deletion forget is still an audit event (someone *tried*)."""
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    cm, _ = _make_cm(audit_sink=sink, extract_replies=[json.dumps([])])
    cm.add_message("user", "hello world")
    cm.forget(text_matches="nonexistent-pattern")
    records = _read_jsonl(path)
    assert len(records) == 1
    assert records[0]["result"]["total_deleted"] == 0


def test_cm_without_audit_sink_works_unchanged(tmp_path: Path) -> None:
    """Default construction (no sink) preserves Phase 36 behaviour."""
    cm, mem = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "gardening today")
    result = cm.forget(text_matches="gardening")
    assert isinstance(result, ForgetResult)
    assert result.total_deleted >= 1


# ----------------------------------------------------------------------
# 6. target_user_id: caller != subject
# ----------------------------------------------------------------------
def test_audit_records_target_user_id_when_differs(tmp_path: Path) -> None:
    """Admin forgetting another user's data stamps both ids."""
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    sink.emit(
        user_id="ops-root",  # caller (principal.sub)
        target_user_id="alice",  # subject of the delete
        criteria={"user_id": "alice"},
        dry_run=False,
        result=_sample_result(),
    )
    rec = _read_jsonl(path)[0]
    assert rec["user_id"] == "ops-root"
    assert rec["target_user_id"] == "alice"


# ----------------------------------------------------------------------
# 7. Record compactness — records should stay small
# ----------------------------------------------------------------------
def test_forget_summary_strategy_drop_forces_drop(tmp_path: Path) -> None:
    """``summary_strategy="drop"`` deletes the summary even when survivors exist.

    Phase 36 default ("regen"): partial coverage → summary is
    rewritten from surviving turns. Phase 37 Task 3 adds an opt-in
    override for conservative callers who'd rather over-delete than
    rely on an LLM to avoid leaking the scrubbed subject into the
    regenerated text.

    This test stages: two turns 0..1, a manual summary covering
    [0, 1], then ``forget(text_matches="gardening")`` where only turn
    0 matches. Under the default the Phase 36 cascade would
    regenerate the summary from turn 1. Under ``summary_strategy=
    "drop"`` the summary is deleted outright and
    ``regenerated_summaries`` stays empty.
    """
    cm, mem = _make_cm(extract_replies=[json.dumps([]), json.dumps([])])
    cm.add_message("user", "gardening turn 0")
    cm.add_message("user", "unrelated turn 1")
    # Manual summary covering [0, 1] so the partial-coverage branch
    # would fire on default. Forgetting turn 0 leaves turn 1 as a
    # survivor; without the opt-in, Phase 36 regenerates.
    mem.store(
        "original summary text",
        metadata={
            "session_id": "s",
            "type": "summary",
            "summarized_turn_start": 0,
            "summarized_turn_end": 1,
        },
    )
    result = cm.forget(text_matches="gardening", summary_strategy="drop")
    # Drop strategy: summary is deleted, not regenerated.
    assert result.deleted_summaries, (
        "drop strategy must delete the overlapping summary"
    )
    assert result.regenerated_summaries == [], (
        "drop strategy must skip regen entirely"
    )


def test_forget_summary_strategy_drop_no_llm_call(tmp_path: Path) -> None:
    """``summary_strategy="drop"`` never calls the LLM for summary regen.

    Under the default Phase 36 cascade a partially-covered summary
    triggers an LLM regen call. The drop strategy must skip the LLM
    path entirely — crucial for the LLM-unavailable deploy story
    where the wrapping caller wants a guaranteed no-regen forget.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)

    @dataclass
    class _RecordingLLM:
        summary_calls: int = 0
        extract_calls: int = 0
        name: str = "recording"

        def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
            if "summarizing a short segment" in prompt:
                self.summary_calls += 1
                raise RuntimeError("LLM should not be called for drop strategy")
            if "You extract atomic facts" in prompt:
                self.extract_calls += 1
                return "[]"
            return "{}"

    llm = _RecordingLLM()
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="s",
        summary_every=1000,
    )
    cm.add_message("user", "gardening turn 0")
    cm.add_message("user", "unrelated turn 1")
    mem.store(
        "original summary text",
        metadata={
            "session_id": "s",
            "type": "summary",
            "summarized_turn_start": 0,
            "summarized_turn_end": 1,
        },
    )
    # Must not raise — drop strategy skips the LLM, so the
    # RuntimeError guard in the fake backend never fires.
    result = cm.forget(text_matches="gardening", summary_strategy="drop")
    assert llm.summary_calls == 0, "drop strategy must not call the summary LLM"
    assert result.deleted_summaries


def test_forget_summary_strategy_default_is_regen(tmp_path: Path) -> None:
    """Default behaviour unchanged: regen when there are survivors."""
    cm, mem = _make_cm(extract_replies=[json.dumps([]), json.dumps([])])
    cm.add_message("user", "gardening turn 0")
    cm.add_message("user", "unrelated turn 1")
    mem.store(
        "original summary text",
        metadata={
            "session_id": "s",
            "type": "summary",
            "summarized_turn_start": 0,
            "summarized_turn_end": 1,
        },
    )
    result = cm.forget(text_matches="gardening")
    # Phase 36 regen: survivors exist, summary is rewritten.
    assert result.regenerated_summaries, (
        "default (regen) must rewrite partially-covered summaries"
    )
    assert result.deleted_summaries == [], (
        "regen is a replacement, not a deletion"
    )


def test_forget_summary_strategy_invalid_raises(tmp_path: Path) -> None:
    cm, _mem = _make_cm(extract_replies=[json.dumps([])])
    cm.add_message("user", "gardening")
    with pytest.raises(ValueError, match="summary_strategy"):
        cm.forget(text_matches="gardening", summary_strategy="bogus")  # type: ignore[arg-type]


def test_audit_record_is_one_line_per_emit(tmp_path: Path) -> None:
    """Each emit is one JSONL line; embedded newlines would break tail -f."""
    path = tmp_path / "a.jsonl"
    sink = ForgetAuditSink(path=path)
    sink.emit(
        user_id="alice",
        target_user_id=None,
        criteria={"text_matches": "line\nwith\nnewlines"},
        dry_run=False,
        result=_sample_result(),
    )
    content = path.read_text()
    # Exactly one trailing newline; no embedded raw newlines inside
    # the JSON body (json.dumps escapes them to \n literals).
    assert content.count("\n") == 1
    assert not re.search(r"\n.", content[:-1])
