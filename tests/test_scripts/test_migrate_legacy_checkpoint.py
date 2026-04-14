"""Tests for scripts/migrate_legacy_checkpoint.py — pre-v1 → v1 migrator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from soma.core.brain_bundle import wrap_payload

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "migrate_legacy_checkpoint.py"


def test_migrate_legacy_checkpoint_converts_to_v1(tmp_path: Path) -> None:
    # Build a legacy-style checkpoint (no format envelope).
    legacy_path = tmp_path / "legacy.pt"
    legacy_state = {"global_step": 42, "config": {"seed": 0}, "graph": {}}
    torch.save(legacy_state, str(legacy_path))

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(legacy_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr

    # Backup file exists next to the migrated path.
    bak = legacy_path.with_suffix(legacy_path.suffix + ".legacy-bak")
    assert bak.exists()

    # Migrated file is a v1 envelope with the original payload preserved.
    raw = torch.load(str(legacy_path), map_location="cpu", weights_only=False)
    assert raw["format"] == "soma-brain"
    assert raw["schema_version"] == 1
    assert raw["payload"]["global_step"] == 42


def test_migrate_idempotent_on_v1(tmp_path: Path) -> None:
    # Running on an already-v1 file is a no-op (no backup, no rewrite).
    v1_path = tmp_path / "v1.pt"
    wrapped = wrap_payload({"global_step": 1}, soma_version="test")
    torch.save(wrapped, str(v1_path))

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(v1_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "Already v1" in result.stdout
    # No .legacy-bak created for idempotent runs.
    assert not v1_path.with_suffix(v1_path.suffix + ".legacy-bak").exists()


def test_migrate_missing_file_exits_nonzero(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(tmp_path / "nope.pt")],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 2
