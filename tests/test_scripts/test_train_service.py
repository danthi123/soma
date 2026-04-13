"""Tests for scripts/train_service.py — long-lived training daemon."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.train_service import (
    PidFile,
    atomic_write_json,
    check_pid_collision,
)


def test_atomic_write_json(tmp_path: Path) -> None:
    path = tmp_path / "h.json"
    atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    # Second write replaces atomically
    atomic_write_json(path, {"b": 2})
    assert json.loads(path.read_text()) == {"b": 2}


def test_pidfile_writes_current_pid(tmp_path: Path) -> None:
    pidfile = PidFile(tmp_path / "svc.pid")
    pidfile.write()
    assert pidfile.path.exists()
    assert int(pidfile.path.read_text()) == os.getpid()
    pidfile.remove()
    assert not pidfile.path.exists()


def test_check_pid_collision_no_file(tmp_path: Path) -> None:
    assert check_pid_collision(tmp_path / "no.pid") is False


def test_check_pid_collision_stale_pid(tmp_path: Path) -> None:
    path = tmp_path / "stale.pid"
    path.write_text("999999999")
    assert check_pid_collision(path) is False


def test_check_pid_collision_live_pid(tmp_path: Path) -> None:
    path = tmp_path / "live.pid"
    path.write_text(str(os.getpid()))
    assert check_pid_collision(path) is True
