"""Tests for scripts/approve.py and scripts/reject.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.approve import (
    format_list_table,
    load_queue,
    mark_approved,
    mark_rejected,
    mark_stale,
    write_queue,
)


def _sample_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "queue_id": "abc123def",
        "change_log_id": "uuid-1",
        "base_commit_sha": "a" * 40,
        "class": "config",
        "description": "raise base_lr from 0.001 to 0.0015",
        "rationale": "loss plateaued",
        "queued_at": "2026-04-12T10:00:00+00:00",
        "queue_status": "pending",
        "approved_at": None,
        "approved_by": None,
        "rejected_at": None,
        "rejected_reason": None,
        "applied_at": None,
        "applied_tick_id": None,
        "diff_hash": "deadbeef",
    }
    entry.update(overrides)
    return entry


# ---- transitions ---------------------------------------------------------


def test_mark_approved_sets_fields() -> None:
    entry = _sample_entry()
    updated = mark_approved(entry, username="alice", ts="2026-04-12T11:00:00+00:00")
    assert updated["queue_status"] == "approved"
    assert updated["approved_by"] == "alice"
    assert updated["approved_at"] == "2026-04-12T11:00:00+00:00"


def test_mark_stale_sets_reason() -> None:
    entry = _sample_entry()
    updated = mark_stale(entry, reason="base_commit_sha drifted", ts="2026-04-12T11:00:00+00:00")
    assert updated["queue_status"] == "stale"
    assert updated["rejected_reason"] == "base_commit_sha drifted"
    assert updated["rejected_at"] == "2026-04-12T11:00:00+00:00"


def test_mark_rejected_sets_reason() -> None:
    entry = _sample_entry()
    updated = mark_rejected(entry, reason="user doesn't want this", ts="2026-04-12T11:00:00+00:00")
    assert updated["queue_status"] == "rejected"
    assert updated["rejected_reason"] == "user doesn't want this"
    assert updated["rejected_at"] == "2026-04-12T11:00:00+00:00"


# ---- I/O -----------------------------------------------------------------


def test_load_write_roundtrip(tmp_path: Path) -> None:
    q = tmp_path / "approval_queue.jsonl"
    entries = [_sample_entry(queue_id="aaa"), _sample_entry(queue_id="bbb")]
    write_queue(q, entries)
    loaded = load_queue(q)
    assert len(loaded) == 2
    assert loaded[0]["queue_id"] == "aaa"
    assert loaded[1]["queue_id"] == "bbb"


def test_load_queue_missing(tmp_path: Path) -> None:
    assert load_queue(tmp_path / "missing.jsonl") == []


def test_load_queue_skips_bad_lines(tmp_path: Path) -> None:
    q = tmp_path / "q.jsonl"
    q.write_text(
        json.dumps(_sample_entry(queue_id="x"))
        + "\nnot-json\n"
        + json.dumps(_sample_entry(queue_id="y"))
        + "\n"
    )
    loaded = load_queue(q)
    assert [e["queue_id"] for e in loaded] == ["x", "y"]


# ---- list formatting -----------------------------------------------------


def test_format_list_table_headers() -> None:
    entries = [_sample_entry()]
    table = format_list_table(entries)
    assert "ID" in table
    assert "CLASS" in table
    assert "QUEUE_STATUS" in table
    assert "BASE_SHA" in table
    assert "DESCRIPTION" in table


def test_format_list_table_truncates_description() -> None:
    long_desc = "x" * 200
    entries = [_sample_entry(description=long_desc)]
    table = format_list_table(entries)
    # The full 200-char description should be truncated.
    assert long_desc not in table


def test_format_list_table_empty_message() -> None:
    assert "empty" in format_list_table([]).lower()


# ---- approve.py CLI integration -----------------------------------------


def _repo_path() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_approve(args: list[str], queue_path: Path) -> subprocess.CompletedProcess[str]:
    """Invoke approve.py from the repo root against ``queue_path``."""
    return subprocess.run(
        [
            sys.executable,
            str(_repo_path() / "scripts/approve.py"),
            *args,
            "--queue",
            str(queue_path),
        ],
        cwd=str(_repo_path()),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_reject(args: list[str], queue_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_repo_path() / "scripts/reject.py"),
            *args,
            "--queue",
            str(queue_path),
        ],
        cwd=str(_repo_path()),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write_queue_file(cwd: Path, entries: list[dict[str, object]]) -> Path:
    q = cwd / "approval_queue.jsonl"
    q.parent.mkdir(parents=True, exist_ok=True)
    with q.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    return q


def _current_head_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_repo_path()),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip()


def test_approve_list_shows_entries(tmp_path: Path) -> None:
    entries = [_sample_entry(queue_id="abc123", description="raise lr slightly")]
    q = _write_queue_file(tmp_path, entries)
    result = _run_approve(["--list"], queue_path=q)
    assert result.returncode == 0
    assert "abc123" in result.stdout
    assert "raise lr" in result.stdout


def test_approve_fresh_entry_goes_to_approved(tmp_path: Path) -> None:
    head = _current_head_sha()
    entries = [_sample_entry(queue_id="qid111", base_commit_sha=head)]
    q = _write_queue_file(tmp_path, entries)
    result = _run_approve(["qid111"], queue_path=q)
    assert result.returncode == 0, result.stderr
    loaded = [json.loads(ln) for ln in q.read_text().splitlines() if ln.strip()]
    assert loaded[0]["queue_status"] == "approved"
    assert loaded[0]["approved_by"]  # non-empty
    assert loaded[0]["approved_at"]


def test_approve_stale_when_head_drifted(tmp_path: Path) -> None:
    entries = [_sample_entry(queue_id="qid222", base_commit_sha="0" * 40)]
    q = _write_queue_file(tmp_path, entries)
    result = _run_approve(["qid222"], queue_path=q)
    assert result.returncode == 0
    loaded = [json.loads(ln) for ln in q.read_text().splitlines() if ln.strip()]
    assert loaded[0]["queue_status"] == "stale"
    assert "drift" in (loaded[0]["rejected_reason"] or "").lower()


def test_approve_refuses_already_approved(tmp_path: Path) -> None:
    entries = [_sample_entry(queue_id="qid333", queue_status="approved")]
    q = _write_queue_file(tmp_path, entries)
    result = _run_approve(["qid333"], queue_path=q)
    assert result.returncode == 1
    assert "approved" in result.stderr.lower() or "approved" in result.stdout.lower()


def test_approve_unknown_id_exits_nonzero(tmp_path: Path) -> None:
    q = _write_queue_file(tmp_path, [_sample_entry(queue_id="qid444")])
    result = _run_approve(["nonexistent"], queue_path=q)
    assert result.returncode != 0


def test_reject_pending_entry_marks_rejected(tmp_path: Path) -> None:
    entries = [_sample_entry(queue_id="qid555")]
    q = _write_queue_file(tmp_path, entries)
    result = _run_reject(["qid555", "--reason", "too risky"], queue_path=q)
    assert result.returncode == 0, result.stderr
    loaded = [json.loads(ln) for ln in q.read_text().splitlines() if ln.strip()]
    assert loaded[0]["queue_status"] == "rejected"
    assert loaded[0]["rejected_reason"] == "too risky"


def test_reject_refuses_already_applied(tmp_path: Path) -> None:
    entries = [_sample_entry(queue_id="qid666", queue_status="applied")]
    q = _write_queue_file(tmp_path, entries)
    result = _run_reject(["qid666"], queue_path=q)
    assert result.returncode == 1
    assert "applied" in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("status", ["approved", "rejected", "stale", "applied"])
def test_approve_refuses_terminal_statuses(tmp_path: Path, status: str) -> None:
    entries = [_sample_entry(queue_id=f"q_{status}", queue_status=status)]
    q = _write_queue_file(tmp_path, entries)
    result = _run_approve([f"q_{status}"], queue_path=q)
    assert result.returncode == 1
