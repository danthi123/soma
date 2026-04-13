"""Tests for scripts/lock.py — cross-platform PID-aware file lock."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from scripts.lock import FileLock, is_pid_alive


def test_is_pid_alive_current_process() -> None:
    assert is_pid_alive(os.getpid()) is True


def test_is_pid_alive_dead_process() -> None:
    # A PID that is extremely unlikely to exist.
    fake_pid = 999_999_999
    assert is_pid_alive(fake_pid) is False


def test_lock_acquire_and_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    lock = FileLock(lock_path, timeout=5.0, purpose="unit-test")
    assert lock.acquire() is True
    assert lock_path.exists()
    payload = json.loads(lock_path.read_text())
    assert payload["pid"] == os.getpid()
    assert payload["purpose"] == "unit-test"
    lock.release()
    assert not lock_path.exists()


def test_lock_context_manager(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    with FileLock(lock_path, timeout=5.0, purpose="ctx-test") as acquired:
        assert acquired is True
        assert lock_path.exists()
    assert not lock_path.exists()


def test_lock_blocked_by_live_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    # Seed the lock file with the current (alive) PID.
    lock_path.write_text(json.dumps({"pid": os.getpid(), "ts": time.time(), "purpose": "seed"}))
    lock = FileLock(lock_path, timeout=1.0, purpose="blocked")
    assert lock.acquire() is False  # our own PID is alive; timeout hits


def test_lock_reclaims_stale_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    lock_path.write_text(json.dumps({"pid": 999_999_999, "ts": time.time(), "purpose": "dead"}))
    lock = FileLock(lock_path, timeout=1.0, purpose="reclaim")
    assert lock.acquire() is True
    lock.release()


def test_lock_reclaims_old_timestamp(tmp_path: Path) -> None:
    lock_path = tmp_path / "test.lock"
    # 30+ min old timestamp — considered stale even with a (possibly) live PID.
    very_old = time.time() - 31 * 60
    lock_path.write_text(json.dumps({"pid": os.getpid(), "ts": very_old, "purpose": "old"}))
    lock = FileLock(lock_path, timeout=1.0, purpose="reclaim-old")
    assert lock.acquire() is True
    lock.release()
