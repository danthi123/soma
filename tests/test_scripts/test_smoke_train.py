"""Tests for scripts/smoke_train.py — safety-gate sanity check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_smoke_train_completes() -> None:
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/smoke_train.py"),
            "--steps",
            "10",
            "--device",
            "cpu",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"smoke_train failed: {result.stderr}"
    assert "smoke_train: OK" in result.stdout
