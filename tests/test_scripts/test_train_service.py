"""Tests for scripts/train_service.py — long-lived training daemon."""

from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.train_service import (
    HEARTBEAT_STATUS_PAUSED,
    HEARTBEAT_STATUS_RUNNING,
    HEARTBEAT_STATUS_SHUTDOWN,
    HEARTBEAT_STATUS_WARMING_UP,
    Heartbeat,
    PidFile,
    SignalPoller,
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


def test_heartbeat_writes_atomic_json(tmp_path: Path) -> None:
    hb = Heartbeat(tmp_path / "hb.json", device="cpu")
    hb.update(step=5, status=HEARTBEAT_STATUS_RUNNING)
    data = json.loads((tmp_path / "hb.json").read_text())
    assert data["step"] == 5
    assert data["status"] == HEARTBEAT_STATUS_RUNNING
    assert data["device"] == "cpu"
    assert data["pid"] == os.getpid()
    assert "ts" in data


def test_heartbeat_status_constants_are_strings() -> None:
    for status in (
        HEARTBEAT_STATUS_WARMING_UP,
        HEARTBEAT_STATUS_RUNNING,
        HEARTBEAT_STATUS_PAUSED,
        HEARTBEAT_STATUS_SHUTDOWN,
    ):
        assert isinstance(status, str)


def test_signal_poller_detects_and_consumes(tmp_path: Path) -> None:
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    poller = SignalPoller(signals_dir)
    # No signals
    assert poller.read_and_consume() == []
    # Drop a shutdown + pause signal
    (signals_dir / "shutdown").touch()
    (signals_dir / "pause").touch()
    seen = poller.read_and_consume()
    assert set(seen) == {"shutdown", "pause"}
    # Signals consumed — next read is empty
    assert poller.read_and_consume() == []


def test_signal_poller_ignores_unknown(tmp_path: Path) -> None:
    signals_dir = tmp_path / "signals"
    signals_dir.mkdir()
    (signals_dir / "bogus").touch()
    poller = SignalPoller(signals_dir)
    assert poller.read_and_consume() == []
    # unknown file NOT consumed
    assert (signals_dir / "bogus").exists()


def test_signal_poller_missing_dir(tmp_path: Path) -> None:
    poller = SignalPoller(tmp_path / "does-not-exist")
    assert poller.read_and_consume() == []
