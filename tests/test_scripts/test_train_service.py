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
    LoadResult,
    PidFile,
    SignalPoller,
    atomic_write_json,
    check_pid_collision,
    load_checkpoint_chain,
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


def test_load_chain_fresh_when_nothing_exists(tmp_path: Path) -> None:
    result = load_checkpoint_chain(
        current=tmp_path / "current.pt",
        last_good=tmp_path / "last_good.pt",
        fresh_flag=tmp_path / "fresh_init.flag",
    )
    assert isinstance(result, LoadResult)
    assert result.source == "fresh"
    assert result.payload is None
    assert (tmp_path / "fresh_init.flag").exists()


def test_load_chain_picks_current_when_present(tmp_path: Path) -> None:
    (tmp_path / "current.pt").write_bytes(b"dummy")
    sentinel = object()

    def fake_loader(p: Path) -> object:
        return sentinel

    result = load_checkpoint_chain(
        current=tmp_path / "current.pt",
        last_good=tmp_path / "last_good.pt",
        fresh_flag=tmp_path / "fresh_init.flag",
        loader=fake_loader,
    )
    assert result.source == "current"
    assert result.payload is sentinel
    assert not (tmp_path / "fresh_init.flag").exists()


def test_load_chain_falls_back_to_last_good(tmp_path: Path) -> None:
    (tmp_path / "current.pt").write_bytes(b"corrupt")
    (tmp_path / "last_good.pt").write_bytes(b"also-dummy")
    attempts: list[Path] = []

    def fake_loader(p: Path) -> object:
        attempts.append(p)
        if p.name == "current.pt":
            raise RuntimeError("simulated corruption")
        return object()

    result = load_checkpoint_chain(
        current=tmp_path / "current.pt",
        last_good=tmp_path / "last_good.pt",
        fresh_flag=tmp_path / "fresh_init.flag",
        loader=fake_loader,
    )
    assert result.source == "last_good"
    assert attempts == [tmp_path / "current.pt", tmp_path / "last_good.pt"]
    assert not (tmp_path / "fresh_init.flag").exists()


def test_load_chain_all_corrupt_falls_through_to_fresh(tmp_path: Path) -> None:
    (tmp_path / "current.pt").write_bytes(b"bad")
    (tmp_path / "last_good.pt").write_bytes(b"also-bad")

    def bad_loader(p: Path) -> object:
        raise RuntimeError("corrupt")

    result = load_checkpoint_chain(
        current=tmp_path / "current.pt",
        last_good=tmp_path / "last_good.pt",
        fresh_flag=tmp_path / "fresh_init.flag",
        loader=bad_loader,
    )
    assert result.source == "fresh"
    assert (tmp_path / "fresh_init.flag").exists()
