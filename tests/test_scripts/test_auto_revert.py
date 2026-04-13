"""Tests for scripts/auto_revert.py — watchdog confirm/revert logic."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from scripts.auto_revert import (
    _summarize_metrics,
    append_change_log_entry,
    bucket_by_minute,
    check_confirmation,
    is_stale_tick,
    read_recent_metrics,
    service_crashed_after_apply,
    update_change_log_entry,
)

# ---- staleness ------------------------------------------------------------


def test_service_crashed_after_apply_missing_file(tmp_path: Path) -> None:
    assert service_crashed_after_apply(tmp_path / "missing.json", ts_applied=100.0) is False


def test_service_crashed_after_apply_post_dates(tmp_path: Path) -> None:
    path = tmp_path / "train_permanent_failure.json"
    path.write_text(json.dumps({"ts": 200.0, "step": 1}), encoding="utf-8")
    assert service_crashed_after_apply(path, ts_applied=100.0) is True


def test_service_crashed_after_apply_pre_dates(tmp_path: Path) -> None:
    path = tmp_path / "train_permanent_failure.json"
    path.write_text(json.dumps({"ts": 50.0, "step": 1}), encoding="utf-8")
    # A stale pre-apply permanent_failure (from before this change) must
    # NOT trigger revert - the change could still be fine.
    assert service_crashed_after_apply(path, ts_applied=100.0) is False


def test_service_crashed_after_apply_malformed(tmp_path: Path) -> None:
    path = tmp_path / "train_permanent_failure.json"
    path.write_text("not json", encoding="utf-8")
    assert service_crashed_after_apply(path, ts_applied=100.0) is False


def test_is_stale_tick_recent() -> None:
    assert is_stale_tick({"ts": time.time() - 10}) is False


def test_is_stale_tick_old() -> None:
    assert is_stale_tick({"ts": time.time() - 3601}) is True


def test_is_stale_tick_missing_ts() -> None:
    assert is_stale_tick({}) is True


# ---- metrics I/O ---------------------------------------------------------


def test_read_recent_metrics_empty_file(tmp_path: Path) -> None:
    (tmp_path / "m.jsonl").touch()
    assert read_recent_metrics(tmp_path / "m.jsonl", since_ts=0) == []


def test_read_recent_metrics_filters_old(tmp_path: Path) -> None:
    m = tmp_path / "m.jsonl"
    now = time.time()
    records = [
        {"ts": now - 1000, "step": 0, "loss": 1.0},
        {"ts": now - 100, "step": 10, "loss": 0.9},
        {"ts": now - 10, "step": 20, "loss": 0.8},
    ]
    m.write_text("\n".join(json.dumps(r) for r in records))
    recent = read_recent_metrics(m, since_ts=now - 200)
    assert len(recent) == 2


def test_read_recent_metrics_ignores_garbage(tmp_path: Path) -> None:
    m = tmp_path / "m.jsonl"
    m.write_text(
        json.dumps({"ts": time.time(), "step": 1, "loss": 0.1})
        + "\n"
        + "not json\n"
        + json.dumps({"ts": time.time(), "step": 2, "loss": 0.2})
        + "\n"
    )
    recent = read_recent_metrics(m, since_ts=0)
    assert len(recent) == 2


# ---- bucketing -----------------------------------------------------------


def test_bucket_by_minute_groups_correctly() -> None:
    records = [
        {"ts": 0, "step": 1},
        {"ts": 30, "step": 2},
        {"ts": 60, "step": 3},
        {"ts": 125, "step": 4},
    ]
    buckets = bucket_by_minute(records)
    assert len(buckets) == 3
    assert len(buckets[0]) == 2  # 0, 30 → minute 0
    assert len(buckets[1]) == 1  # 60 → minute 1
    assert len(buckets[2]) == 1  # 125 → minute 2


def test_bucket_by_minute_empty() -> None:
    assert bucket_by_minute([]) == []


# ---- check_confirmation --------------------------------------------------


def _make_records(
    start_ts: float,
    start_step: int,
    *,
    num_minutes: int,
    loss: float = 1.0,
    curiosity: float = 0.5,
    num_nodes: int = 50,
    num_edges: int = 200,
    per_bucket: int = 3,
) -> list[dict[str, object]]:
    """Build ``num_minutes * per_bucket`` synthetic metric records."""
    records: list[dict[str, object]] = []
    step = start_step
    for minute in range(num_minutes):
        for i in range(per_bucket):
            records.append(
                {
                    "ts": start_ts + minute * 60 + i * 10,
                    "step": step,
                    "loss": loss,
                    "curiosity": curiosity,
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                }
            )
            step += 50  # advance 50 steps between records
    return records


def test_check_confirmation_insufficient_buckets() -> None:
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 50}
    recent = _make_records(0, 0, num_minutes=1, per_bucket=3)
    verdict, _reason, _crits = check_confirmation(pre, recent, window_min=15, min_step_advance=500)
    assert verdict == "pending"


def test_check_confirmation_insufficient_step_advance() -> None:
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 50}
    recent = _make_records(0, 0, num_minutes=2, per_bucket=3)
    # 2 minutes × 3 per bucket × 50 step/record = 300 step advance < 500
    recent_trimmed = [r for r in recent if r.get("step", 0) < 300]
    verdict, _reason, _crits = check_confirmation(
        pre, recent_trimmed, window_min=15, min_step_advance=500
    )
    assert verdict == "pending"


def test_check_confirmation_all_pass() -> None:
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 50}
    recent = _make_records(
        0, 0, num_minutes=15, per_bucket=3, loss=1.05, curiosity=0.52, num_nodes=52
    )
    verdict, _reason, crits = check_confirmation(pre, recent, window_min=15, min_step_advance=500)
    assert verdict == "confirmed"
    assert all(crits.values())


def test_check_confirmation_revert_on_loss_spike() -> None:
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 50}
    # Make the LAST three buckets spike — those should trigger revert.
    recent = _make_records(0, 0, num_minutes=12, per_bucket=3, loss=1.0)
    recent += _make_records(12 * 60, 12 * 3 * 50, num_minutes=3, per_bucket=3, loss=5.0)
    verdict, _reason, _crits = check_confirmation(pre, recent, window_min=15, min_step_advance=500)
    assert verdict == "revert"


def test_check_confirmation_revert_on_nan_loss() -> None:
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 50}
    recent = _make_records(0, 0, num_minutes=10, per_bucket=3)
    # Append 3 buckets full of NaN loss
    recent += _make_records(10 * 60, 10 * 3 * 50, num_minutes=3, per_bucket=3, loss=float("nan"))
    verdict, _reason, _crits = check_confirmation(pre, recent, window_min=15, min_step_advance=500)
    assert verdict == "revert"


def test_check_confirmation_revert_on_graph_collapse() -> None:
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 100}
    recent = _make_records(0, 0, num_minutes=10, per_bucket=3, num_nodes=100)
    # Drop nodes below 50% of pre-change for 3 consecutive buckets
    recent += _make_records(10 * 60, 10 * 3 * 50, num_minutes=3, per_bucket=3, num_nodes=20)
    verdict, _reason, _crits = check_confirmation(pre, recent, window_min=15, min_step_advance=500)
    assert verdict == "revert"


def test_check_confirmation_transient_bad_doesnt_revert() -> None:
    """A single bad bucket doesn't trip revert (needs 3 consecutive).

    It DOES reset the good-streak counter, so a window with a transient
    bad bucket in the middle stays in ``pending`` until 15 more good
    buckets accumulate on the tail.
    """
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 50}
    good = _make_records(0, 0, num_minutes=7, per_bucket=3)
    bad = _make_records(7 * 60, 7 * 3 * 50, num_minutes=1, per_bucket=3, loss=5.0)
    more_good = _make_records(8 * 60, 8 * 3 * 50, num_minutes=7, per_bucket=3, loss=1.0)
    recent = good + bad + more_good
    verdict, _reason, _crits = check_confirmation(pre, recent, window_min=15, min_step_advance=500)
    # 1 bad < 3 consecutive → no revert. But good streak was reset, so we're pending.
    assert verdict == "pending"


# ---- change-log I/O ------------------------------------------------------


def test_append_and_update_change_log_entry(tmp_path: Path) -> None:
    log = tmp_path / "change_log.jsonl"
    append_change_log_entry(
        log,
        {"change_log_id": "abc123", "status": "in_progress", "commit_sha": "deadbeef"},
    )
    append_change_log_entry(
        log,
        {"change_log_id": "def456", "status": "in_progress", "commit_sha": None},
    )
    update_change_log_entry(log, change_log_id="abc123", updates={"status": "confirmed", "foo": 1})
    lines = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    assert lines[0]["status"] == "confirmed"
    assert lines[0]["foo"] == 1
    assert lines[1]["status"] == "in_progress"


def test_update_change_log_entry_no_match(tmp_path: Path) -> None:
    log = tmp_path / "change_log.jsonl"
    append_change_log_entry(log, {"change_log_id": "abc", "status": "done"})
    update_change_log_entry(log, change_log_id="missing", updates={"status": "x"})
    lines = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    assert lines[0]["status"] == "done"


def test_summarize_metrics_basic() -> None:
    records = [
        {"loss": 1.0},
        {"loss": 0.5},
        {"loss": 2.0},
    ]
    summary = _summarize_metrics(records)
    assert summary["count"] == 3
    assert summary["last_loss"] == pytest.approx(2.0)
    assert summary["mean_loss"] == pytest.approx((1.0 + 0.5 + 2.0) / 3)


def test_summarize_metrics_empty() -> None:
    summary = _summarize_metrics([])
    assert summary["count"] == 0
    assert summary["last_loss"] is None or (
        isinstance(summary["last_loss"], float) and math.isnan(summary["last_loss"])
    )
