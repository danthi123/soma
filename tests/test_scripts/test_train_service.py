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
    CrashBackoff,
    Heartbeat,
    LoadResult,
    PidFile,
    SignalPoller,
    append_metric_record,
    atomic_write_json,
    check_pid_collision,
    load_checkpoint_chain,
    prune_step_checkpoints,
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


def test_crash_backoff_triggers_at_threshold() -> None:
    cb = CrashBackoff(window_seconds=300.0, max_crashes=3)
    assert cb.record_crash(now=100.0) is False
    assert cb.record_crash(now=101.0) is False
    assert cb.record_crash(now=102.0) is True  # 3rd crash in window -> permanent


def test_crash_backoff_forgets_old_crashes() -> None:
    cb = CrashBackoff(window_seconds=300.0, max_crashes=3)
    cb.record_crash(now=0.0)
    cb.record_crash(now=100.0)
    # Third crash 400s later — both prior crashes are outside the 300s window
    # (400-0=400 and 400-100=300 are both >= window), so only the new one stays.
    assert cb.record_crash(now=400.0) is False
    assert cb.crashes == [400.0]
    # But if we'd added a crash just within window, it would count.
    assert cb.record_crash(now=500.0) is False  # window has [400, 500]
    assert cb.record_crash(now=600.0) is True  # window has [400, 500, 600]


def test_prune_step_checkpoints_keeps_last_n(tmp_path: Path) -> None:
    for step in (100, 200, 300, 400, 500, 600):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    deleted = prune_step_checkpoints(
        tmp_path, keep_last=3, permanent_every_steps=10_000
    )
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    # 400, 500, 600 kept
    assert remaining == ["step_00000400.pt", "step_00000500.pt", "step_00000600.pt"]
    assert {p.name for p in deleted} == {
        "step_00000100.pt",
        "step_00000200.pt",
        "step_00000300.pt",
    }


def test_prune_step_checkpoints_preserves_permanent(tmp_path: Path) -> None:
    # Permanent cadence 100 — so steps 100, 200, 300 are all permanent.
    for step in (50, 75, 100, 125, 150, 200, 250, 300):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    prune_step_checkpoints(tmp_path, keep_last=2, permanent_every_steps=100)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    # Permanent: 100, 200, 300. Last-2: 250, 300. Union: 100, 200, 250, 300.
    assert remaining == [
        "step_00000100.pt",
        "step_00000200.pt",
        "step_00000250.pt",
        "step_00000300.pt",
    ]


def test_prune_step_checkpoints_empty_dir(tmp_path: Path) -> None:
    assert prune_step_checkpoints(tmp_path, keep_last=5, permanent_every_steps=1000) == []


def test_prune_step_checkpoints_ignores_non_step_files(tmp_path: Path) -> None:
    (tmp_path / "current.pt").write_bytes(b"x")
    (tmp_path / "last_good.pt").write_bytes(b"x")
    (tmp_path / "step_00000100.pt").write_bytes(b"x")
    (tmp_path / "step_00000200.pt").write_bytes(b"x")
    (tmp_path / "step_00000300.pt").write_bytes(b"x")
    prune_step_checkpoints(tmp_path, keep_last=1, permanent_every_steps=10_000)
    # current.pt + last_good.pt never touched, last step kept
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert "current.pt" in remaining
    assert "last_good.pt" in remaining
    assert "step_00000300.pt" in remaining
    assert "step_00000100.pt" not in remaining


def test_append_metric_record_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    append_metric_record(path, {"step": 1, "loss": 0.5})
    append_metric_record(path, {"step": 2, "loss": 0.4})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 1, "loss": 0.5}
    assert json.loads(lines[1]) == {"step": 2, "loss": 0.4}
