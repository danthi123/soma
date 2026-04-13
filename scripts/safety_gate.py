"""Autonomous-loop safety gate orchestrator.

Runs, in order, stopping at first failure:
  1. ruff format src/ tests/   (auto-applies formatting fixes)
  2. ruff check src/ tests/
  3. mypy src/soma/
  4. pytest -q
  5. (unless --skip-smoke) pause training, smoke_train.py, resume training

On any failure: writes ``.soma-loop/state/gate_failure.json`` with
``{stage, stderr_tail, duration_s, ts}`` and exits 1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

STAGE_ORDER = ["format", "check", "mypy", "test", "smoke"]


def _run(cmd: list[str], timeout: float) -> tuple[int, str, str, float]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr, time.time() - start
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        err = err or f"TIMEOUT after {timeout}s"
        return 124, out, err, time.time() - start


def _tail(s: str, n: int = 4000) -> str:
    return s[-n:] if len(s) > n else s


def run_format() -> tuple[int, str]:
    code, _out, err, _dur = _run(
        [sys.executable, "-m", "ruff", "format", "src/", "tests/"], timeout=60
    )
    return code, _tail(err)


def run_check() -> tuple[int, str]:
    code, out, err, _dur = _run(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/"], timeout=60
    )
    return code, _tail(out + err)


def run_mypy() -> tuple[int, str]:
    code, out, err, _dur = _run([sys.executable, "-m", "mypy", "src/soma/"], timeout=180)
    return code, _tail(out + err)


def run_pytest() -> tuple[int, str]:
    code, out, err, _dur = _run(
        [sys.executable, "-m", "pytest", "-q", "-m", "not slow"], timeout=600
    )
    return code, _tail(out + err)


def run_smoke(
    *,
    signal_dir: Path,
    heartbeat_path: Path,
    device: str = "auto",
) -> tuple[int, str]:
    """Pause training service, run smoke, resume training."""
    signal_dir.mkdir(parents=True, exist_ok=True)
    pause = signal_dir / "pause"
    resume = signal_dir / "resume"

    pause.touch()
    # Poll heartbeat for ``status=paused`` up to 10s.
    deadline = time.time() + 10
    reached_paused = False
    while time.time() < deadline:
        if heartbeat_path.exists():
            try:
                data = json.loads(heartbeat_path.read_text(encoding="utf-8"))
                if data.get("status") == "paused":
                    reached_paused = True
                    break
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.5)

    if not reached_paused:
        with contextlib.suppress(OSError):
            pause.unlink(missing_ok=True)
        return 1, "smoke: training did not reach paused state within 10s"

    try:
        code, out, err, _dur = _run(
            [
                sys.executable,
                "scripts/smoke_train.py",
                "--steps",
                "30",
                "--device",
                device,
            ],
            timeout=180,
        )
        return code, _tail(out + err)
    finally:
        resume.touch()
        with contextlib.suppress(OSError):
            pause.unlink(missing_ok=True)


def write_failure(stage: str, stderr_tail: str, duration_s: float) -> None:
    out = Path(".soma-loop/state/gate_failure.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "stage": stage,
                "stderr_tail": stderr_tail,
                "duration_s": round(duration_s, 2),
                "ts": time.time(),
            }
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOMA safety gate orchestrator.")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--stage", choices=[*STAGE_ORDER, "all"], default="all")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    stages_to_run = [args.stage] if args.stage != "all" else list(STAGE_ORDER)
    if args.skip_smoke and "smoke" in stages_to_run:
        stages_to_run = [s for s in stages_to_run if s != "smoke"]

    signal_dir = Path(".soma-loop/signals")
    heartbeat = Path(".soma-loop/state/train_heartbeat.json")

    for stage in stages_to_run:
        start = time.time()
        if stage == "format":
            code, tail = run_format()
        elif stage == "check":
            code, tail = run_check()
        elif stage == "mypy":
            code, tail = run_mypy()
        elif stage == "test":
            code, tail = run_pytest()
        elif stage == "smoke":
            code, tail = run_smoke(
                signal_dir=signal_dir, heartbeat_path=heartbeat, device=args.device
            )
        else:  # pragma: no cover - argparse constrains choices
            code, tail = 1, f"unknown stage: {stage}"

        dur = time.time() - start
        if code != 0:
            print(f"safety_gate: {stage} FAILED in {dur:.1f}s", file=sys.stderr)
            print(tail, file=sys.stderr)
            write_failure(stage, tail, dur)
            return 1
        print(f"safety_gate: {stage} OK ({dur:.1f}s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
