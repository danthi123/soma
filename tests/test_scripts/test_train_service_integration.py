"""Integration test: does train_service start, heartbeat, and shut down cleanly?"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.slow
def test_train_service_starts_and_shuts_down() -> None:
    pytest.importorskip("torch")
    import torch

    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "configs/current.yaml").exists():
        pytest.skip("configs/current.yaml missing — run bootstrap_loop first")
    if not (repo_root / "data/tinyshakespeare.txt").exists():
        pytest.skip("tinyshakespeare corpus missing")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    heartbeat = repo_root / ".soma-loop/state/train_heartbeat.json"
    signals_dir = repo_root / ".soma-loop/signals"
    pidfile = repo_root / ".soma-loop/pid/train_service.pid"

    # Clean slate — fail fast if another service is already running.
    if pidfile.exists():
        pytest.skip(f"another train_service instance appears to own {pidfile}")
    for stale in (heartbeat, signals_dir / "shutdown", signals_dir / "pause"):
        stale.unlink(missing_ok=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        [
            sys.executable,
            "scripts/train_service.py",
            "--config",
            "configs/current.yaml",
            "--device",
            device,
        ],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait up to 60s for first heartbeat.
        deadline = time.time() + 60
        while time.time() < deadline:
            if heartbeat.exists():
                try:
                    json.loads(heartbeat.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    pass
            if proc.poll() is not None:
                raise AssertionError(
                    f"train_service exited prematurely "
                    f"(code={proc.returncode}); stderr={proc.stderr.read().decode()}"  # type: ignore[union-attr]
                )
            time.sleep(0.5)
        else:
            raise AssertionError("heartbeat never appeared within 60s")

        signals_dir.mkdir(parents=True, exist_ok=True)
        (signals_dir / "shutdown").touch()
        proc.wait(timeout=60)
        assert proc.returncode == 0
        final_hb = json.loads(heartbeat.read_text(encoding="utf-8"))
        assert final_hb["status"] == "shutdown"
        assert not pidfile.exists()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
