"""Tests for scripts/audit_loop.py — Phase 1 validation oracle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scripts.audit_loop import (
    CARVEOUT_PATTERN,
    check_auto_commits_have_change_log_id,
    check_consecutive_failures_zero,
    check_device_cuda_consistent,
    check_disk_footprint,
    check_gate_passed_on_pristine,
    check_no_carveout_touch,
    parse_commit_log_entries,
)

# ---- carveout pattern ----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "src/soma/core/config.py",
        "src/soma/memory/working.py",
        "src/soma/growth/neurogenesis.py",
        "src/soma/system.py",
        "data/heldout.txt",
        "data/fixed_prompts.txt",
        "scripts/train_service.py",
        "tests/test_core/test_graph.py",
        "pyproject.toml",
        "CLAUDE.md",
        "docs/plans/2026-04-12-autonomous-loop-design.md",
    ],
)
def test_carveout_pattern_matches_restricted_paths(path: str) -> None:
    assert CARVEOUT_PATTERN.search(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "configs/current.yaml",
        "configs/default.yaml",
        "src/soma/io/text_encoder.py",
        "src/soma/ui/app.py",
        "README.md",
    ],
)
def test_carveout_pattern_misses_allowed_paths(path: str) -> None:
    assert CARVEOUT_PATTERN.search(path) is None


# ---- parse_commit_log_entries ----------------------------------------------


def test_parse_commit_log_entries_extracts_auto_only() -> None:
    raw = (
        "commit deadbeef\n"
        "Author: ai@soma\n"
        "\n"
        "    auto: tune lr\n"
        "    \n"
        "    Change-log-id: abc123\n"
        "\n"
        "commit feedface\n"
        "Author: human@soma\n"
        "\n"
        "    feat(loop): something\n"
    )
    entries = parse_commit_log_entries(raw)
    assert len(entries) == 2
    assert entries[0]["sha"] == "deadbeef"
    assert entries[0]["is_auto"] is True
    assert entries[0]["change_log_id"] == "abc123"
    assert entries[1]["is_auto"] is False


def test_parse_commit_log_entries_missing_trailer() -> None:
    raw = (
        "commit deadbeef\n"
        "Author: ai@soma\n"
        "\n"
        "    auto: tune lr\n"
    )
    entries = parse_commit_log_entries(raw)
    assert entries[0]["change_log_id"] is None


# ---- check_auto_commits_have_change_log_id -------------------------------


def test_check_auto_commits_all_have_trailer() -> None:
    commits = [
        {"sha": "a1", "is_auto": True, "change_log_id": "x"},
        {"sha": "a2", "is_auto": True, "change_log_id": "y"},
        {"sha": "a3", "is_auto": False, "change_log_id": None},
    ]
    ok, bad = check_auto_commits_have_change_log_id(commits, change_log_entries=[])
    assert ok is True
    assert bad == []


def test_check_auto_commits_fallback_to_change_log_sha() -> None:
    commits = [{"sha": "a1", "is_auto": True, "change_log_id": None}]
    change_log = [{"change_log_id": "uuid-1", "commit_sha": "a1"}]
    ok, _bad = check_auto_commits_have_change_log_id(commits, change_log_entries=change_log)
    assert ok is True


def test_check_auto_commits_flags_untracked() -> None:
    commits = [{"sha": "rogue", "is_auto": True, "change_log_id": None}]
    ok, bad = check_auto_commits_have_change_log_id(commits, change_log_entries=[])
    assert ok is False
    assert bad == ["rogue"]


# ---- check_no_carveout_touch ---------------------------------------------


def test_check_no_carveout_touch_ok() -> None:
    touches = [
        ("a1", True, {"configs/current.yaml"}),
        ("a2", False, {"src/soma/core/graph.py"}),  # human commit — allowed
    ]
    ok, bad = check_no_carveout_touch(touches)
    assert ok is True
    assert bad == []


def test_check_no_carveout_touch_flags_auto_on_carveout() -> None:
    touches = [("a1", True, {"src/soma/core/graph.py"})]
    ok, bad = check_no_carveout_touch(touches)
    assert ok is False
    assert bad[0][0] == "a1"


def test_check_no_carveout_touch_ignores_non_auto() -> None:
    touches = [("a1", False, {"src/soma/core/graph.py"})]
    ok, _bad = check_no_carveout_touch(touches)
    assert ok is True


# ---- check_device_cuda_consistent ---------------------------------------


def test_check_device_cuda_consistent_happy() -> None:
    heartbeats = [
        {"device": "cuda", "ts": time.time() - 100},
        {"device": "cuda:0", "ts": time.time()},
    ]
    ok, reason = check_device_cuda_consistent(heartbeats)
    assert ok is True
    assert reason is None


def test_check_device_cuda_consistent_flags_cpu() -> None:
    heartbeats = [
        {"device": "cuda", "ts": time.time() - 100},
        {"device": "cpu", "ts": time.time()},
    ]
    ok, reason = check_device_cuda_consistent(heartbeats)
    assert ok is False
    assert reason and "cpu" in reason.lower()


def test_check_device_cuda_consistent_empty_ok() -> None:
    # No data → can't complain
    ok, _reason = check_device_cuda_consistent([])
    assert ok is True


# ---- check_disk_footprint ------------------------------------------------


def test_check_disk_footprint_within_bounds(tmp_path: Path) -> None:
    (tmp_path / "small.bin").write_bytes(b"x" * 1024)
    ok, size_gb = check_disk_footprint(tmp_path, max_gb=1.0)
    assert ok is True
    assert size_gb < 1.0


def test_check_disk_footprint_over_bounds(tmp_path: Path) -> None:
    (tmp_path / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
    ok, size_gb = check_disk_footprint(tmp_path, max_gb=0.0000001)
    assert ok is False
    assert size_gb > 0.0000001


def test_check_disk_footprint_missing_dir(tmp_path: Path) -> None:
    ok, size_gb = check_disk_footprint(tmp_path / "missing", max_gb=10.0)
    assert ok is True
    assert size_gb == 0.0


# ---- check_consecutive_failures_zero ------------------------------------


def test_check_consecutive_failures_zero_ok(tmp_path: Path) -> None:
    path = tmp_path / "cf.json"
    path.write_text(json.dumps({"count": 0}))
    ok, _reason = check_consecutive_failures_zero(path)
    assert ok is True


def test_check_consecutive_failures_non_zero_fails(tmp_path: Path) -> None:
    path = tmp_path / "cf.json"
    path.write_text(json.dumps({"count": 2}))
    ok, _reason = check_consecutive_failures_zero(path)
    assert ok is False


def test_check_consecutive_failures_missing_file(tmp_path: Path) -> None:
    # Missing file = no known failures = OK
    ok, _reason = check_consecutive_failures_zero(tmp_path / "missing.json")
    assert ok is True


# ---- check_gate_passed_on_pristine --------------------------------------


def test_check_gate_passed_on_pristine_no_failure_file(tmp_path: Path) -> None:
    # No gate_failure.json present → gate has passed (or never run)
    ok, _reason = check_gate_passed_on_pristine(
        gate_failure=tmp_path / "missing.json", change_log_entries=[]
    )
    assert ok is True


def test_check_gate_passed_on_pristine_with_failure(tmp_path: Path) -> None:
    path = tmp_path / "gate_failure.json"
    path.write_text(json.dumps({"stage": "check", "ts": time.time()}))
    ok, _reason = check_gate_passed_on_pristine(gate_failure=path, change_log_entries=[])
    assert ok is False


def test_check_gate_passed_on_pristine_overridden_by_confirmed_entry(tmp_path: Path) -> None:
    """A gate_failure file from an old run is tolerated if the loop has since
    confirmed a change (proving the gate is usable)."""
    path = tmp_path / "gate_failure.json"
    path.write_text(json.dumps({"stage": "check", "ts": time.time() - 7200}))
    change_log = [{"status": "confirmed", "confirmed_at": time.time() - 100}]
    ok, _reason = check_gate_passed_on_pristine(
        gate_failure=path, change_log_entries=change_log
    )
    assert ok is True
