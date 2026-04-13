"""Tests for scripts/safety_gate.py — gate orchestrator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.safety_gate import _tail, write_failure


def test_tail_truncates_long_string() -> None:
    s = "a" * 5000
    assert len(_tail(s, n=100)) == 100
    assert _tail(s, n=100) == "a" * 100


def test_tail_returns_short_string_unchanged() -> None:
    assert _tail("hello", n=100) == "hello"


def test_write_failure_creates_state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_failure("check", "some stderr tail", 1.5)
    out = tmp_path / ".soma-loop/state/gate_failure.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["stage"] == "check"
    assert data["stderr_tail"] == "some stderr tail"
    assert data["duration_s"] == 1.5
    assert "ts" in data


@pytest.mark.slow
def test_safety_gate_skip_smoke_on_clean_tree() -> None:
    """On a clean tree, the full gate (minus smoke) should pass."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts/safety_gate.py"), "--skip-smoke"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"gate failed: {result.stdout}\n{result.stderr}"


@pytest.mark.slow
def test_safety_gate_single_stage_check() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/safety_gate.py"),
            "--stage",
            "check",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"check stage failed: {result.stdout}\n{result.stderr}"
    assert "check OK" in result.stdout
