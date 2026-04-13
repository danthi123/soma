"""Long-lived SOMA training daemon for the autonomous loop.

Responsibilities (across Tasks 5-10):
- Load config from configs/current.yaml (fallback: configs/default.yaml)
- Checkpoint-load fallback chain: current.pt -> last_good.pt -> fresh SOMA
- Write heartbeat to .soma-loop/state/train_heartbeat.json every 10s
- Poll .soma-loop/signals/ for pause/resume/shutdown/reload_config
- Append metrics every 10 steps
- Atomic checkpoint every config.checkpoint_interval steps
- Retention: last 20 step checkpoints + every-100th-checkpoint permanently
- Crash backoff: 3 exceptions within 5 min -> write train_permanent_failure.json + exit

Ctrl+C / SIGINT also triggers graceful shutdown.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lock import is_pid_alive


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to path atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


class PidFile:
    """Manages a PID file; atomic write, caller removes on exit."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        os.replace(tmp_name, self.path)

    def remove(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def check_pid_collision(pid_path: Path) -> bool:
    """Return True if a live process already owns the pid file."""
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return False
    return is_pid_alive(pid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOMA training service daemon.")
    parser.add_argument("--config", type=Path, default=Path("configs/current.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default="auto")
    parser.parse_args(argv)

    pidfile = PidFile(Path(".soma-loop/pid/train_service.pid"))
    if check_pid_collision(pidfile.path):
        print(
            f"ERROR: another train_service is already running (pid file: {pidfile.path})",
            file=sys.stderr,
        )
        return 1
    pidfile.write()
    print(f"train_service: pid {os.getpid()} started (stub)", flush=True)
    # Full main loop wired up in Tasks 6-8.
    pidfile.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
