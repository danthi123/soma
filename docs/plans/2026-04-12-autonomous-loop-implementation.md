# Autonomous SOMA Improvement Loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Phase 1 autonomous improvement loop described in `docs/plans/2026-04-12-autonomous-loop-design.md`. Delivers a Claude-in-the-middle loop that runs `train → test → diagnose → propose → gate → commit → restart → wiki-sync` on a 37-minute cadence, with Level 4 autonomy and multi-layer safety backstops.

**Architecture:** Two long-running processes + short-lived Claude sessions. `scripts/train_service.py` runs SOMA continuously. Task Scheduler (Windows) fires `scripts/run_tick.ps1` every 37 min → `claude --print` executes `docs/loop_tick_prompt.md`. A separate 2-min watchdog (`scripts/auto_revert.py`) confirms or reverts changes. Three new skills (`soma-diagnose`, `soma-propose-change`, `soma-bootstrap-kb`) encapsulate the Claude-judgment phases.

**Tech Stack:** Python 3.11+, PyTorch (existing), portalocker (new), Claude Code CLI (`claude --print`), Windows Task Scheduler (Phase 1), Gitea API via curl. Tests via pytest. Lint/type via ruff + mypy. Platform: Windows 11 + RTX 3090 (Linux cloud documented as future).

**Design doc:** `docs/plans/2026-04-12-autonomous-loop-design.md` — authoritative for all design decisions (G1-G90). Refer back when intent is unclear.

---

## Pre-flight for this plan

Before starting any task:
1. Confirm `main` branch, clean working tree: `git status`
2. Confirm existing tests pass: `python -m pytest tests/ -q`
3. Confirm ruff + mypy clean: `python -m ruff check src/ tests/ && python -m mypy src/soma/`
4. Read the design doc end-to-end (`docs/plans/2026-04-12-autonomous-loop-design.md`) — particularly §3 (architecture), §4 (tick flow), §5 (components), Appendix B (schemas).

All tasks below assume you are operating from the repo root `E:/Documents/Projects/SOMA/`. Windows path separators in the file system; forward slashes in git / Python paths.

**Commit convention during this build:** all commits use `feat(loop):`, `test(loop):`, `docs(loop):`, `chore(loop):` prefixes (NOT the autonomous `auto:` prefix — that's reserved for the running loop itself). Each task ends with a single commit that moves the plan forward.

**If any step fails unexpectedly:** stop, diagnose, fix root cause, do NOT skip to the next task. The design doc gates this level — broken mid-build state compromises Phase 1 validation.

---

## Phase 1A — Foundation

### Task 1: Add `portalocker` dependency

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_deps_smoke.py`

**Step 1: Write the failing test**

```python
# tests/test_deps_smoke.py
"""Smoke tests for new dependencies required by the autonomous loop."""

def test_portalocker_importable():
    import portalocker
    assert hasattr(portalocker, "Lock")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deps_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'portalocker'`

**Step 3: Add portalocker to pyproject.toml**

Open `pyproject.toml`, locate the `[project] dependencies = [...]` list, add `"portalocker>=2.8"`. Then install: `pip install -e ".[dev]"` (or whichever extras your dev env uses).

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_deps_smoke.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml tests/test_deps_smoke.py
git commit -m "chore(loop): add portalocker dep for cross-platform file locking"
```

---

### Task 2: `scripts/lock.py` — FileLock library

**Files:**
- Create: `scripts/lock.py`
- Create: `tests/test_scripts/test_lock.py`

Cross-platform file-lock helper using portalocker + PID/timestamp metadata. See design §5.2 `scripts/lock.py`.

**Step 1: Write failing tests**

```python
# tests/test_scripts/test_lock.py
"""Tests for scripts/lock.py — cross-platform PID-aware file lock."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# Import from scripts/ — add a conftest.py that adjusts sys.path if not already present
from scripts.lock import FileLock, is_pid_alive


def test_is_pid_alive_current_process():
    assert is_pid_alive(os.getpid()) is True


def test_is_pid_alive_dead_process():
    # PID 0 (Linux) / some very high unlikely PID works cross-platform;
    # we'll use a PID that definitely doesn't exist.
    fake_pid = 999_999_999
    assert is_pid_alive(fake_pid) is False


def test_lock_acquire_and_release(tmp_path: Path):
    lock_path = tmp_path / "test.lock"
    lock = FileLock(lock_path, timeout=5.0, purpose="unit-test")
    assert lock.acquire() is True
    assert lock_path.exists()
    payload = json.loads(lock_path.read_text())
    assert payload["pid"] == os.getpid()
    assert payload["purpose"] == "unit-test"
    lock.release()
    assert not lock_path.exists()


def test_lock_context_manager(tmp_path: Path):
    lock_path = tmp_path / "test.lock"
    with FileLock(lock_path, timeout=5.0, purpose="ctx-test") as acquired:
        assert acquired is True
        assert lock_path.exists()
    assert not lock_path.exists()


def test_lock_blocked_by_live_pid(tmp_path: Path):
    lock_path = tmp_path / "test.lock"
    # Seed the lock file with the current PID
    lock_path.write_text(json.dumps({"pid": os.getpid(), "ts": time.time(), "purpose": "seed"}))
    lock = FileLock(lock_path, timeout=1.0, purpose="blocked")
    assert lock.acquire() is False  # our own PID is alive; timeout hits


def test_lock_reclaims_stale_pid(tmp_path: Path):
    lock_path = tmp_path / "test.lock"
    lock_path.write_text(json.dumps({"pid": 999_999_999, "ts": time.time(), "purpose": "dead"}))
    lock = FileLock(lock_path, timeout=1.0, purpose="reclaim")
    assert lock.acquire() is True
    lock.release()


def test_lock_reclaims_old_timestamp(tmp_path: Path):
    lock_path = tmp_path / "test.lock"
    # 30+ min old timestamp — considered stale even with a (possibly) live PID
    very_old = time.time() - 31 * 60
    lock_path.write_text(json.dumps({"pid": os.getpid(), "ts": very_old, "purpose": "old"}))
    lock = FileLock(lock_path, timeout=1.0, purpose="reclaim-old")
    assert lock.acquire() is True
    lock.release()
```

Also create `tests/test_scripts/__init__.py` (empty) and `tests/conftest.py` (adds scripts/ to sys.path):

```python
# tests/conftest.py (create if not existing; else add the sys.path insertion)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scripts/test_lock.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts.lock'`).

**Step 3: Implement `scripts/lock.py`**

```python
# scripts/lock.py
"""Cross-platform PID-aware file lock using portalocker.

Lock file JSON format: {"pid": <int>, "ts": <epoch seconds>, "purpose": "<str>"}

Staleness: a lock is considered abandoned (reclaimable) when EITHER the owning
PID is no longer alive OR the timestamp is older than 30 minutes. This matches
the design doc's §5.2 lock.py contract.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import portalocker

STALE_AGE_SECONDS = 30 * 60  # 30 minutes


def is_pid_alive(pid: int) -> bool:
    """Return True iff a process with this PID exists and is visible."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # On Windows, OpenProcess with SYNCHRONIZE access tells us if the PID exists.
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        # Check exit code — STILL_ACTIVE means alive.
        exit_code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but we can't signal


def _is_lock_stale(lock_data: dict[str, Any]) -> bool:
    pid = int(lock_data.get("pid", 0))
    ts = float(lock_data.get("ts", 0))
    if not is_pid_alive(pid):
        return True
    if time.time() - ts > STALE_AGE_SECONDS:
        return True
    return False


class FileLock:
    """PID-aware exclusive lock on a file path.

    Lightweight implementation: writes a JSON payload. portalocker gives us the
    exclusive-access guarantee during the read-modify-write. Staleness reclaim
    lets us recover from crashed lock holders.
    """

    def __init__(self, path: Path | str, timeout: float = 30.0, purpose: str = "") -> None:
        self.path = Path(path)
        self.timeout = float(timeout)
        self.purpose = purpose
        self._held = False

    def acquire(self) -> bool:
        """Attempt to acquire the lock within ``self.timeout`` seconds.

        Returns True on success, False on timeout.
        """
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Use portalocker.RLock via a small guard file for atomic contention.
                guard = self.path.with_suffix(self.path.suffix + ".guard")
                with portalocker.Lock(
                    guard, mode="a", timeout=0.5, flags=portalocker.LOCK_EX | portalocker.LOCK_NB
                ):
                    if self.path.exists():
                        try:
                            existing = json.loads(self.path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError):
                            existing = None
                        if existing and not _is_lock_stale(existing):
                            # Live owner — wait and retry
                            time.sleep(0.5)
                            continue
                    payload = {"pid": os.getpid(), "ts": time.time(), "purpose": self.purpose}
                    self.path.write_text(json.dumps(payload), encoding="utf-8")
                    self._held = True
                    return True
            except portalocker.exceptions.LockException:
                time.sleep(0.2)
                continue
        return False

    def release(self) -> None:
        if self._held and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        self._held = False

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scripts/test_lock.py -v`
Expected: all 6 tests PASS.

**Step 5: Run ruff + mypy**

Run: `python -m ruff check scripts/ tests/test_scripts/ && python -m mypy scripts/lock.py`
Expected: clean.

**Step 6: Commit**

```bash
git add scripts/lock.py tests/test_scripts/ tests/conftest.py
git commit -m "feat(loop): add cross-platform PID-aware FileLock helper"
```

---

### Task 3: `.gitignore` additions + `.soma-loop/` directory scaffolding (doc only)

**Files:**
- Modify: `.gitignore`

**Step 1: Update `.gitignore`**

Add these lines at the end:

```gitignore

# Autonomous loop runtime state
.soma-loop/
checkpoints/
```

(`checkpoints/` may already be there — check first.)

**Step 2: Verify**

Run: `git status` — nothing in `.soma-loop/` or `checkpoints/` should appear as untracked (since those dirs don't exist yet, status is just clean).

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore(loop): gitignore .soma-loop/ runtime state + checkpoints/"
```

---

### Task 4: `scripts/bootstrap_loop.py` — one-time setup

**Files:**
- Create: `scripts/bootstrap_loop.py`
- Create: `tests/test_scripts/test_bootstrap_loop.py`

Per design §5.2 `scripts/bootstrap_loop.py`. Seeds `configs/current.yaml`, `data/heldout.txt`, `data/fixed_prompts.txt`, `data/corpus_token_freq.json`, and the `.soma-loop/` directory tree.

**Step 1: Write failing tests**

```python
# tests/test_scripts/test_bootstrap_loop.py
"""Tests for scripts/bootstrap_loop.py — idempotent directory + file scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.bootstrap_loop import (
    FIXED_PROMPTS,
    build_heldout_split,
    build_token_freq,
    ensure_state_dirs,
    seed_current_yaml,
    seed_fixed_prompts,
)


def test_fixed_prompts_count():
    assert len(FIXED_PROMPTS) == 20
    # prompt 16 (index 15) is empty per design Appendix C
    assert FIXED_PROMPTS[15] == ""
    # last prompt is long
    assert len(FIXED_PROMPTS[-1]) > 20


def test_ensure_state_dirs(tmp_path: Path):
    ensure_state_dirs(tmp_path)
    for rel in [
        ".soma-loop/signals",
        ".soma-loop/state/ticks",
        ".soma-loop/metrics",
        ".soma-loop/reports/chat",
        ".soma-loop/logs",
        ".soma-loop/pid",
    ]:
        assert (tmp_path / rel).is_dir(), f"missing: {rel}"
    # Seed files created
    for rel in [
        ".soma-loop/state/change_log.jsonl",
        ".soma-loop/state/approval_queue.jsonl",
        ".soma-loop/state/consecutive_failures.json",
    ]:
        assert (tmp_path / rel).is_file()
    cf = json.loads((tmp_path / ".soma-loop/state/consecutive_failures.json").read_text())
    assert cf == {"count": 0, "last_reset_ts": None, "last_failure_ts": None}


def test_seed_current_yaml_copies_default(tmp_path: Path):
    default = tmp_path / "configs/default.yaml"
    current = tmp_path / "configs/current.yaml"
    default.parent.mkdir(parents=True)
    default.write_text("vocab_size: 512\ntext_embed_dim: 64\n", encoding="utf-8")
    seed_current_yaml(default, current, force=False)
    assert current.exists()
    assert current.read_text() == default.read_text()
    # Never overwrite without force
    current.write_text("vocab_size: 999\n", encoding="utf-8")
    seed_current_yaml(default, current, force=False)
    assert "999" in current.read_text(), "must not be overwritten without --force"
    # --force still does NOT overwrite current.yaml per G64 safety
    seed_current_yaml(default, current, force=True)
    assert "999" in current.read_text(), "even --force preserves user-edited current.yaml"


def test_seed_fixed_prompts_preserves_user_edits(tmp_path: Path):
    path = tmp_path / "data/fixed_prompts.txt"
    path.parent.mkdir(parents=True)
    seed_fixed_prompts(path, force=False)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    # Operator edits — must be preserved on rerun (with OR without --force)
    path.write_text("custom prompt\n", encoding="utf-8")
    seed_fixed_prompts(path, force=True)  # G64: never overwritten
    assert path.read_text() == "custom prompt\n"


def test_build_heldout_split_deterministic(tmp_path: Path):
    corpus = tmp_path / "data/corpus.txt"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("\n".join(f"line {i}" for i in range(100)) + "\n", encoding="utf-8")
    out_a = tmp_path / "data/heldout_a.txt"
    out_b = tmp_path / "data/heldout_b.txt"
    build_heldout_split(corpus, out_a, ratio=0.1, seed=42)
    build_heldout_split(corpus, out_b, ratio=0.1, seed=42)
    assert out_a.read_text() == out_b.read_text()
    # Roughly 10% of 100 lines
    assert 5 <= out_a.read_text().count("\n") <= 15


def test_build_token_freq_json(tmp_path: Path):
    corpus = tmp_path / "data/corpus.txt"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("hello world\nhello there\n", encoding="utf-8")
    out = tmp_path / "data/corpus_token_freq.json"
    build_token_freq(corpus, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    # At minimum, the common words are present
    assert "hello" in data
    assert data["hello"] == 2
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scripts/test_bootstrap_loop.py -v`
Expected: FAIL (module not found).

**Step 3: Implement `scripts/bootstrap_loop.py`**

```python
# scripts/bootstrap_loop.py
"""One-time bootstrap for the autonomous SOMA improvement loop.

Creates .soma-loop/ directory tree, seeds configs/current.yaml from default,
writes data/fixed_prompts.txt, builds data/heldout.txt split and
data/corpus_token_freq.json KL reference.

Idempotent by default; pass --force to regenerate heldout + token_freq
(configs/current.yaml and data/fixed_prompts.txt are NEVER overwritten
once present — per design G64).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

FIXED_PROMPTS: list[str] = [
    "To be, or not to be,",
    "The king said,",
    "O Romeo, Romeo,",
    "What light through",
    "Shall I compare thee",
    "hello",
    "what is your name",
    "tell me about yourself",
    "how do you feel",
    "who are you",
    "The quick brown",
    "Once upon a",
    "In the beginning",
    "Long ago in",
    "The secret of",
    "",
    ".",
    "a",
    "a a a a a a a a a a a a a a a a",
    "Now is the winter of our discontent made glorious summer by this sun of York",
]


def ensure_state_dirs(repo_root: Path) -> None:
    """Create .soma-loop/ structure + initial state files.

    Idempotent — creates missing dirs/files, leaves existing alone.
    """
    dirs = [
        ".soma-loop/signals",
        ".soma-loop/state/ticks",
        ".soma-loop/metrics",
        ".soma-loop/reports/chat",
        ".soma-loop/logs",
        ".soma-loop/pid",
    ]
    for rel in dirs:
        (repo_root / rel).mkdir(parents=True, exist_ok=True)

    empty_files = [
        ".soma-loop/state/change_log.jsonl",
        ".soma-loop/state/approval_queue.jsonl",
    ]
    for rel in empty_files:
        path = repo_root / rel
        if not path.exists():
            path.touch()

    cf_path = repo_root / ".soma-loop/state/consecutive_failures.json"
    if not cf_path.exists():
        cf_path.write_text(
            json.dumps({"count": 0, "last_reset_ts": None, "last_failure_ts": None}),
            encoding="utf-8",
        )


def seed_current_yaml(default: Path, current: Path, *, force: bool) -> None:
    """Copy default.yaml to current.yaml if current doesn't exist.

    Per G64, NEVER overwrites current.yaml even with --force.
    """
    if current.exists():
        return
    current.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(default, current)


def seed_fixed_prompts(path: Path, *, force: bool) -> None:
    """Write the default 20 fixed prompts if file absent. Never overwrite."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(FIXED_PROMPTS) + "\n", encoding="utf-8")


def build_heldout_split(
    corpus: Path, out: Path, *, ratio: float = 0.1, seed: int = 42
) -> None:
    """Deterministic ratio% split of corpus lines into out."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(lines)), max(1, int(len(lines) * ratio))))
    out.write_text("\n".join(lines[i] for i in selected) + "\n", encoding="utf-8")


def build_token_freq(corpus: Path, out: Path) -> None:
    """Empirical word-frequency dict (JSON) over corpus — used as KL reference.

    Simple whitespace tokenization; the true BPE-tokenized distribution is
    computed by test_harness.py when available. This is the pre-BPE fallback.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    counter: Counter[str] = Counter()
    for line in corpus.read_text(encoding="utf-8").splitlines():
        counter.update(line.split())
    out.write_text(json.dumps(dict(counter)), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap SOMA autonomous loop.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate heldout and token_freq (config + fixed_prompts always preserved).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/tinyshakespeare.txt"),
        help="Corpus file for heldout split and token-freq generation.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root (default: current directory).",
    )
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    default_yaml = repo / "configs/default.yaml"
    current_yaml = repo / "configs/current.yaml"
    prompts = repo / "data/fixed_prompts.txt"
    heldout = repo / "data/heldout.txt"
    token_freq = repo / "data/corpus_token_freq.json"

    if not default_yaml.exists():
        print(f"ERROR: {default_yaml} missing", file=sys.stderr)
        return 1
    if not args.corpus.exists():
        print(f"ERROR: corpus {args.corpus} missing", file=sys.stderr)
        return 1

    ensure_state_dirs(repo)
    seed_current_yaml(default_yaml, current_yaml, force=args.force)
    seed_fixed_prompts(prompts, force=args.force)

    if args.force or not heldout.exists():
        build_heldout_split(args.corpus, heldout, ratio=0.1, seed=42)
    if args.force or not token_freq.exists():
        build_token_freq(args.corpus, token_freq)

    print("Bootstrap complete. Next steps:")
    print("  1. Review data/fixed_prompts.txt (edit if desired)")
    print("  2. Invoke soma-bootstrap-kb skill (once, manually)")
    print("  3. Start train_service: python scripts/train_service.py")
    print("  4. Schedule run_tick.ps1 and auto_revert.py in Task Scheduler")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scripts/test_bootstrap_loop.py -v`
Expected: all 6 PASS.

**Step 5: Run bootstrap once for real**

Run: `python scripts/bootstrap_loop.py`
Expected: completes, creates `.soma-loop/` tree, `configs/current.yaml`, `data/heldout.txt`, `data/corpus_token_freq.json`, `data/fixed_prompts.txt`. Prints next steps.

Verify:
- `ls -la .soma-loop/` shows signals/, state/, metrics/, reports/, logs/, pid/
- `cat configs/current.yaml` matches `configs/default.yaml`
- `wc -l data/heldout.txt` shows ~10% of tinyshakespeare lines

**Step 6: Commit**

```bash
git add scripts/bootstrap_loop.py tests/test_scripts/test_bootstrap_loop.py configs/current.yaml data/heldout.txt data/fixed_prompts.txt data/corpus_token_freq.json
git commit -m "feat(loop): bootstrap_loop.py — one-time setup + seed files"
```

---

## Phase 1B — Training service

### Task 5: `scripts/train_service.py` — skeleton + PID guard

**Files:**
- Create: `scripts/train_service.py`
- Create: `tests/test_scripts/test_train_service.py`

This is the largest component. Build it incrementally across Tasks 5-10. Task 5 = minimal CLI + PID-collision check.

**Step 1: Write failing tests for PID guard**

```python
# tests/test_scripts/test_train_service.py
"""Tests for scripts/train_service.py — long-lived training daemon."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from scripts.train_service import (
    PidFile,
    atomic_write_json,
    check_pid_collision,
)


def test_atomic_write_json(tmp_path: Path):
    path = tmp_path / "h.json"
    atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    # Second write replaces atomically
    atomic_write_json(path, {"b": 2})
    assert json.loads(path.read_text()) == {"b": 2}


def test_pidfile_writes_current_pid(tmp_path: Path):
    pidfile = PidFile(tmp_path / "svc.pid")
    pidfile.write()
    assert pidfile.path.exists()
    assert int(pidfile.path.read_text()) == os.getpid()
    pidfile.remove()
    assert not pidfile.path.exists()


def test_check_pid_collision_no_file(tmp_path: Path):
    assert check_pid_collision(tmp_path / "no.pid") is False


def test_check_pid_collision_stale_pid(tmp_path: Path):
    path = tmp_path / "stale.pid"
    path.write_text("999999999")
    assert check_pid_collision(path) is False


def test_check_pid_collision_live_pid(tmp_path: Path):
    path = tmp_path / "live.pid"
    path.write_text(str(os.getpid()))
    assert check_pid_collision(path) is True
```

**Step 2: Run to verify failure**

Run: `python -m pytest tests/test_scripts/test_train_service.py -v`
Expected: FAIL (module not found).

**Step 3: Create minimal `scripts/train_service.py`**

```python
# scripts/train_service.py
"""Long-lived SOMA training daemon for the autonomous loop.

Responsibilities:
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
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lock import is_pid_alive


def atomic_write_json(path: Path, data: dict[str, object]) -> None:
    """Write JSON to path atomically via tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
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
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


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
    args = parser.parse_args(argv)

    pidfile = PidFile(Path(".soma-loop/pid/train_service.pid"))
    if check_pid_collision(pidfile.path):
        print(
            f"ERROR: another train_service is already running (pid file: {pidfile.path})",
            file=sys.stderr,
        )
        return 1
    pidfile.write()
    print(f"train_service: pid {os.getpid()} started (stub)", flush=True)
    # TODO: main loop added in later tasks
    pidfile.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scripts/test_train_service.py -v`
Expected: all 5 PASS.

**Step 5: Commit**

```bash
git add scripts/train_service.py tests/test_scripts/test_train_service.py
git commit -m "feat(loop): train_service skeleton + PID guard + atomic JSON write"
```

---

### Task 6: `train_service` — heartbeat + signal polling

**Files:**
- Modify: `scripts/train_service.py`
- Modify: `tests/test_scripts/test_train_service.py`

Add the heartbeat writer and signal-polling logic (without the actual SOMA training yet).

**Step 1: Add failing tests**

Append to `tests/test_scripts/test_train_service.py`:

```python
from scripts.train_service import (
    Heartbeat,
    SignalPoller,
    HEARTBEAT_STATUS_WARMING_UP,
    HEARTBEAT_STATUS_RUNNING,
    HEARTBEAT_STATUS_PAUSED,
    HEARTBEAT_STATUS_SHUTDOWN,
)


def test_heartbeat_writes_atomic_json(tmp_path: Path):
    hb = Heartbeat(tmp_path / "hb.json", device="cpu")
    hb.update(step=5, status=HEARTBEAT_STATUS_RUNNING)
    data = json.loads((tmp_path / "hb.json").read_text())
    assert data["step"] == 5
    assert data["status"] == HEARTBEAT_STATUS_RUNNING
    assert data["device"] == "cpu"
    assert data["pid"] == os.getpid()
    assert "ts" in data


def test_signal_poller_detects_and_consumes(tmp_path: Path):
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
```

**Step 2: Run to verify failure**

Expected: ImportError on Heartbeat / SignalPoller.

**Step 3: Implement**

Add to `scripts/train_service.py`:

```python
import time
from typing import Literal

# ---- Heartbeat --------------------------------------------------------------

HEARTBEAT_STATUS_WARMING_UP = "warming_up"
HEARTBEAT_STATUS_RUNNING = "running"
HEARTBEAT_STATUS_PAUSED = "paused"
HEARTBEAT_STATUS_SHUTDOWN = "shutdown"

_HB_STATUS = Literal["warming_up", "running", "paused", "shutdown"]


class Heartbeat:
    def __init__(self, path: Path, *, device: str) -> None:
        self.path = path
        self.device = device

    def update(self, *, step: int, status: _HB_STATUS) -> None:
        atomic_write_json(
            self.path,
            {
                "ts": time.time(),
                "step": step,
                "status": status,
                "device": self.device,
                "pid": os.getpid(),
            },
        )


# ---- Signals ----------------------------------------------------------------


class SignalPoller:
    """Filesystem sentinel poller. Signals consumed by deleting the file."""

    KNOWN_SIGNALS = {"shutdown", "pause", "resume", "reload_config"}

    def __init__(self, signals_dir: Path) -> None:
        self.signals_dir = signals_dir

    def read_and_consume(self) -> list[str]:
        if not self.signals_dir.exists():
            return []
        seen: list[str] = []
        for name in self.KNOWN_SIGNALS:
            path = self.signals_dir / name
            if path.exists():
                seen.append(name)
                try:
                    path.unlink()
                except OSError:
                    pass
        return seen
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_scripts/test_train_service.py -v`
Expected: all PASS.

**Step 5: Commit**

```bash
git add scripts/train_service.py tests/test_scripts/test_train_service.py
git commit -m "feat(loop): train_service heartbeat + signal poller"
```

---

### Task 7: `train_service` — checkpoint load fallback chain

**Files:**
- Modify: `scripts/train_service.py`
- Modify: `tests/test_scripts/test_train_service.py`

Per design §5.2: try `current.pt` → `last_good.pt` → fresh SOMA. Writes `.soma-loop/state/fresh_init.flag` if fresh.

**Step 1: Add failing tests**

```python
from scripts.train_service import load_checkpoint_chain, LoadResult


def test_load_chain_fresh_when_nothing_exists(tmp_path: Path):
    result = load_checkpoint_chain(
        current=tmp_path / "current.pt",
        last_good=tmp_path / "last_good.pt",
        fresh_flag=tmp_path / "fresh_init.flag",
    )
    assert result.source == "fresh"
    assert (tmp_path / "fresh_init.flag").exists()


def test_load_chain_picks_current_when_present(tmp_path: Path):
    (tmp_path / "current.pt").write_bytes(b"dummy")
    result = load_checkpoint_chain(
        current=tmp_path / "current.pt",
        last_good=tmp_path / "last_good.pt",
        fresh_flag=tmp_path / "fresh_init.flag",
    )
    assert result.source == "current"
    assert not (tmp_path / "fresh_init.flag").exists()


def test_load_chain_falls_back_to_last_good(tmp_path: Path, monkeypatch):
    """If current.pt fails to load, fall back to last_good.pt."""
    (tmp_path / "current.pt").write_bytes(b"corrupt")
    (tmp_path / "last_good.pt").write_bytes(b"also-dummy")
    # Mock the loader to raise on current, succeed on last_good
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
```

**Step 2-3: Implement**

```python
# in scripts/train_service.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class LoadResult:
    source: str  # "current" | "last_good" | "fresh"
    payload: object | None


def load_checkpoint_chain(
    *,
    current: Path,
    last_good: Path,
    fresh_flag: Path,
    loader: Callable[[Path], object] | None = None,
) -> LoadResult:
    """Try current.pt, then last_good.pt, then fresh init.

    ``loader`` exists for testing; in production it calls into SOMA.load_state.
    Writes ``fresh_flag`` if falling through to fresh init.
    """
    def _default_loader(p: Path) -> object:
        # Lazy import: SOMA is heavy; tests mock this path.
        import torch

        return torch.load(p, map_location="cpu")

    loader = loader or _default_loader

    for path, source in [(current, "current"), (last_good, "last_good")]:
        if path.exists():
            try:
                payload = loader(path)
                return LoadResult(source=source, payload=payload)
            except Exception as exc:
                print(f"train_service: {source} load failed ({exc}); trying fallback", file=sys.stderr)
                continue

    # Fresh init
    fresh_flag.parent.mkdir(parents=True, exist_ok=True)
    fresh_flag.touch()
    return LoadResult(source="fresh", payload=None)
```

**Step 4: Run tests + commit**

```bash
python -m pytest tests/test_scripts/test_train_service.py -v
git add scripts/train_service.py tests/test_scripts/test_train_service.py
git commit -m "feat(loop): train_service checkpoint load fallback chain"
```

---

### Task 8: `train_service` — main training loop integration

**Files:**
- Modify: `scripts/train_service.py`

Integrate with SOMA. Main loop: load config → build SOMA + feeder → warm-up (100 steps) → running loop with signal polling, pause handling, metrics append, atomic checkpointing every `config.checkpoint_interval` steps, crash backoff (3 crashes in 5 min = permanent failure exit).

Given scope, this task is **larger** than 2-5 minutes and does not follow strict TDD (too many integration moving pieces to unit-test easily). Instead: implement against integration criteria, verify by running.

**Step 1: Add the main loop body to `scripts/train_service.py`**

Implement using existing `TrainingController` logic from `src/soma/ui/training_controller.py` as a reference for how SOMA is built and stepped. Key rules:
- `warming_up` state for first 100 steps (iff LoadResult.source == "fresh")
- Pause state blocks stepping but heartbeat keeps writing with `status=paused`
- Shutdown signal → save atomic checkpoint, write `status=shutdown` heartbeat, remove pidfile, `sys.exit(0)`
- Metrics append every 10 steps to `.soma-loop/metrics/metrics.current.jsonl` (one JSON object per line: `{ts, step, loss, curiosity, num_nodes, num_edges, ...}`)
- Every `config.checkpoint_interval` steps (default 5000): atomic save to `checkpoints/step_NNNNN.pt` + `os.replace` to `checkpoints/current.pt` + update `checkpoints/current.txt` pointer
- Retention: keep last 20 step checkpoints + permanent every-100th (i.e., steps divisible by `100 * checkpoint_interval = 500000`)
- Crash backoff via outer retry loop: if `SOMA.step` raises, log to `.soma-loop/state/train_crash.json`, sleep 10s, retry. If 3 crashes within 5 min, write `.soma-loop/state/train_permanent_failure.json` and exit 2.

**Step 2: Smoke test manually**

After implementing, run:
```
python scripts/train_service.py
```

Expected:
- Process runs; `.soma-loop/state/train_heartbeat.json` updates every ~10s
- `.soma-loop/metrics/metrics.current.jsonl` accumulates lines as training progresses
- `checkpoints/current.pt` appears after `checkpoint_interval` steps
- `touch .soma-loop/signals/pause` → heartbeat `status` becomes `paused`, step counter stops
- `touch .soma-loop/signals/resume` → resumes
- `touch .soma-loop/signals/shutdown` → saves, writes `status=shutdown`, exits

Run each of these in a second terminal.

**Step 3: Add a targeted integration test**

```python
# tests/test_scripts/test_train_service_integration.py
"""Light integration test for train_service: can it start, write heartbeat, shutdown cleanly?"""

import subprocess
import time
from pathlib import Path


def test_train_service_starts_and_shuts_down(tmp_path: Path):
    # Skip on CI without SOMA deps; this is an opt-in test
    pytest.importorskip("torch")
    # Run in a sandbox dir
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        ["python", "scripts/train_service.py", "--config", "configs/current.yaml"],
        cwd=str(Path.cwd()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(5)  # let service start + write first heartbeat
        hb = Path(".soma-loop/state/train_heartbeat.json")
        assert hb.exists()
        # Send shutdown
        Path(".soma-loop/signals/shutdown").touch()
        proc.wait(timeout=30)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
```

(Mark with `@pytest.mark.slow` or similar if CI will skip.)

**Step 4: Commit**

```bash
git add scripts/train_service.py tests/test_scripts/test_train_service_integration.py
git commit -m "feat(loop): train_service full main loop (SOMA step + checkpoints + crash backoff)"
```

---

### Task 9: `scripts/smoke_train.py`

**Files:**
- Create: `scripts/smoke_train.py`
- Create: `tests/test_scripts/test_smoke_train.py`

Per design §5.2 `smoke_train.py`: ephemeral SOMA with **hardcoded minimal config** (vocab_size=64, text_embed_dim=16, small dims), 30 steps on synthetic corpus, validates "does it crash?" not "does current.yaml work end-to-end."

**Step 1: Write failing test**

```python
# tests/test_scripts/test_smoke_train.py
import subprocess
from pathlib import Path


def test_smoke_train_completes():
    pytest.importorskip("torch")
    result = subprocess.run(
        ["python", "scripts/smoke_train.py", "--steps", "10", "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"smoke_train failed: {result.stderr}"
```

**Step 2: Implement**

```python
# scripts/smoke_train.py
"""Smoke-train guard for the safety gate.

Runs a tiny ephemeral SOMA for N steps to verify the training loop, gradient
flow, and graph ops still work. Uses a HARDCODED minimal config independent of
configs/current.yaml, so shape-mismatch between corpus and vocab size is not
an issue.

Exit 0 on success; non-zero on any exception or non-finite loss.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SMOKE_CONFIG = {
    "sensor_output_dim": 16,
    "associator_input_dim": 16,
    "associator_hidden_dim": 32,
    "associator_output_dim": 16,
    "integrator_input_dim": 32,
    "integrator_hidden_dim": 32,
    "integrator_output_dim": 32,
    "position_dim": 4,
    "wm_slots": 4,
    "wm_dim": 16,
    "episodic_capacity": 100,
    "key_dim": 16,
    "value_dim": 16,
    "vocab_size": 64,
    "text_embed_dim": 16,
    "max_nodes": 1000,
    "max_edges_per_node": 10.0,
    "initial_associator_count": 4,
    "initial_integrator_count": 2,
    "max_input_tokens": 16,
    "max_output_tokens": 8,
    "checkpoint_interval": 10000,  # never during smoke
}

SMOKE_CORPUS = [
    "hello world",
    "foo bar baz",
    "the quick brown fox",
    "lorem ipsum dolor",
    "ping pong",
    "abc xyz",
    "one two three",
    "quick test",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOMA smoke-train sanity check.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)

    start = time.time()

    import math

    import torch

    from soma.core.config import SOMAConfig
    from soma.io.dataset_feeders import TextDatasetFeeder
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.system import SOMA

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto" and device.type == "cpu":
        # Explicit: smoke on CPU if no CUDA
        pass

    config = SOMAConfig(**SMOKE_CONFIG)
    tokenizer = train_bpe_tokenizer(SMOKE_CORPUS, vocab_size=config.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=config.text_embed_dim,
        max_seq_len=config.max_input_tokens,
        device=device,
    )
    feeder = TextDatasetFeeder(encoder, SMOKE_CORPUS, chunk_size=4)
    soma = SOMA(config, device=device)

    steps_done = 0
    first_output = config.output_modalities[0]
    for sample in feeder:
        if time.time() - start > args.timeout:
            print(f"smoke_train: TIMEOUT after {args.timeout}s (steps={steps_done})", file=sys.stderr)
            return 2
        if steps_done >= args.steps:
            break
        if sample.target.numel() == 0:
            continue
        inputs = {
            m: t[0].detach() for m, t in sample.inputs.items() if t.numel() > 0
        }
        if not inputs:
            continue
        target = {first_output: sample.target[0].detach()}
        try:
            result = soma.step(inputs=inputs, targets=target)
        except Exception as exc:  # noqa: BLE001
            print(f"smoke_train: step raised: {exc}", file=sys.stderr)
            return 1
        loss = result.get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            print(f"smoke_train: non-finite loss at step {steps_done}: {loss}", file=sys.stderr)
            return 1
        steps_done += 1

    print(f"smoke_train: OK ({steps_done} steps, {time.time()-start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: Run tests + commit**

```bash
python -m pytest tests/test_scripts/test_smoke_train.py -v
python scripts/smoke_train.py --steps 5 --device cpu  # manual verification
git add scripts/smoke_train.py tests/test_scripts/test_smoke_train.py
git commit -m "feat(loop): smoke_train.py for safety gate"
```

---

### Task 10: `scripts/safety_gate.py`

**Files:**
- Create: `scripts/safety_gate.py`
- Create: `tests/test_scripts/test_safety_gate.py`

Per design §5.2. Orchestrates ruff format → ruff check → mypy → pytest → (pause-smoke-resume unless `--skip-smoke`).

**Step 1: Failing tests**

```python
# tests/test_scripts/test_safety_gate.py
import subprocess
from pathlib import Path


def test_safety_gate_skip_smoke_on_clean_tree():
    """On a clean tree, full gate (minus smoke) should pass."""
    result = subprocess.run(
        ["python", "scripts/safety_gate.py", "--skip-smoke"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"gate failed: {result.stdout}\n{result.stderr}"
```

(Extend with failure-case tests using `--stage format` with a deliberately malformed file, rolled back after.)

**Step 2: Implement**

```python
# scripts/safety_gate.py
"""Autonomous-loop safety gate orchestrator.

Runs, in order, stopping at first failure:
  1. ruff format src/ tests/
  2. ruff check src/ tests/
  3. mypy src/soma/
  4. pytest -q
  5. (unless --skip-smoke) pause training, smoke_train.py, resume training

On any failure: writes .soma-loop/state/gate_failure.json with
{stage, stderr_tail, duration_s, ts} and exits 1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGE_ORDER = ["format", "check", "mypy", "test", "smoke"]


def _run(cmd: list[str], timeout: float) -> tuple[int, str, str, float]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr, time.time() - start
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or f"TIMEOUT after {timeout}s", time.time() - start


def _tail(s: str, n: int = 4000) -> str:
    return s[-n:] if len(s) > n else s


def run_format() -> tuple[int, str]:
    code, _, err, _ = _run(["python", "-m", "ruff", "format", "src/", "tests/"], timeout=60)
    return code, _tail(err)


def run_check() -> tuple[int, str]:
    code, out, err, _ = _run(["python", "-m", "ruff", "check", "src/", "tests/"], timeout=60)
    return code, _tail(out + err)


def run_mypy() -> tuple[int, str]:
    code, out, err, _ = _run(["python", "-m", "mypy", "src/soma/"], timeout=120)
    return code, _tail(out + err)


def run_pytest() -> tuple[int, str]:
    code, out, err, _ = _run(["python", "-m", "pytest", "-q"], timeout=600)
    return code, _tail(out + err)


def run_smoke(*, signal_dir: Path, heartbeat_path: Path, device: str = "auto") -> tuple[int, str]:
    """Pause training service, run smoke, resume."""
    signal_dir.mkdir(parents=True, exist_ok=True)
    pause = signal_dir / "pause"
    resume = signal_dir / "resume"

    pause.touch()
    # Poll heartbeat for status=paused
    deadline = time.time() + 10
    while time.time() < deadline:
        if heartbeat_path.exists():
            try:
                data = json.loads(heartbeat_path.read_text())
                if data.get("status") == "paused":
                    break
            except (json.JSONDecodeError, OSError):
                pass
        time.sleep(0.5)
    else:
        if pause.exists():
            pause.unlink()
        return 1, "smoke: training did not reach paused state within 10s"

    try:
        code, out, err, _ = _run(
            ["python", "scripts/smoke_train.py", "--steps", "30", "--device", device],
            timeout=180,
        )
        return code, _tail(out + err)
    finally:
        resume.touch()
        if pause.exists():
            try:
                pause.unlink()
            except OSError:
                pass


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
    parser.add_argument(
        "--stage",
        choices=STAGE_ORDER + ["all"],
        default="all",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    stages_to_run = [args.stage] if args.stage != "all" else STAGE_ORDER
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
            code, tail = run_smoke(signal_dir=signal_dir, heartbeat_path=heartbeat, device=args.device)
        else:
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
```

**Step 3: Run + commit**

```bash
python scripts/safety_gate.py --skip-smoke
git add scripts/safety_gate.py tests/test_scripts/test_safety_gate.py
git commit -m "feat(loop): safety_gate.py orchestrator (ruff/mypy/pytest/smoke)"
```

---

## Phase 1C — Watchdog, approvals, audit

### Task 11: `scripts/test_harness.py` — B-tier evaluation

**Files:**
- Create: `scripts/test_harness.py`
- Create: `tests/test_scripts/test_test_harness.py`

Per design §5.2. Reads `checkpoints/current.pt`, computes held-out loss/perplexity/KL, runs 20 fixed prompts, writes `tick-<tick_id>.json` report + `chat/tick-<tick_id>.jsonl`.

Tokenizer rebuild from corpus (see design tokenizer consistency note).

**Step 1: Failing test**

```python
# tests/test_scripts/test_test_harness.py
import json
import subprocess
from pathlib import Path
import pytest


@pytest.mark.slow
def test_test_harness_produces_valid_report(tmp_path: Path):
    pytest.importorskip("torch")
    # Needs a live checkpoint; produce one via smoke_train
    # Skip if no checkpoint available
    ckpt = Path("checkpoints/current.pt")
    if not ckpt.exists():
        pytest.skip("no live checkpoint for harness test")
    out = tmp_path / "report.json"
    chat_out = tmp_path / "chat.jsonl"
    result = subprocess.run(
        [
            "python",
            "scripts/test_harness.py",
            "--out",
            str(out),
            "--chat-out",
            str(chat_out),
            "--corpus",
            "data/tinyshakespeare.txt",
            "--heldout",
            "data/heldout.txt",
            "--fixed-prompts",
            "data/fixed_prompts.txt",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert "global_step" in data
    assert "heldout_loss_mean" in data
    assert "graph" in data
    assert chat_out.exists()
    lines = chat_out.read_text().splitlines()
    assert len(lines) == 20  # one per fixed prompt
```

**Step 2: Implement**

Implement `scripts/test_harness.py` per design §5.2 contract. Details:
- Load checkpoint via `soma.load_state(path)`
- Build tokenizer from corpus (deterministic)
- Held-out loss: iterate heldout lines, tokenize, run `soma.step` with `loss_only=True` if possible; else compute loss and discard. Average.
- Perplexity: `exp(heldout_loss_mean)`
- KL divergence: compare output-token distribution (softmax of decoder logits) against `data/corpus_token_freq.json`
- Fixed-prompt generation: call `soma.interactive_session(prompt, text_encoder, text_decoder, max_output_tokens=32)` for each prompt, capture output
- Graph stats: `soma.graph.num_nodes`, `num_edges`, avg degree
- Memory stats: `soma.working_memory.occupancy()`, `soma.episodic_memory.num_valid`
- Training stats from training heartbeat: recent loss EMAs
- `health_flags`: list of observations (e.g., "wm_pinned_high", "graph_collapsed", "nan_detected")
- Output JSON per schema in design Appendix B.4

**Step 3: Commit**

```bash
git add scripts/test_harness.py tests/test_scripts/test_test_harness.py
git commit -m "feat(loop): test_harness.py B-tier evaluation"
```

---

### Task 12: `scripts/auto_revert.py` — watchdog

**Files:**
- Create: `scripts/auto_revert.py`
- Create: `tests/test_scripts/test_auto_revert.py`

Per design §5.2 `auto_revert.py`. Confirmation criteria per Appendix B.6 + G33.

**Step 1: Failing tests**

Test each of the 7 logic paths in `auto_revert.py` with mocked `change_log.jsonl` + `metrics.current.jsonl`:

```python
# tests/test_scripts/test_auto_revert.py
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scripts.auto_revert import (
    check_confirmation,
    read_recent_metrics,
    is_stale_tick,
)


def test_is_stale_tick_recent():
    assert is_stale_tick({"ts": time.time() - 10}) is False


def test_is_stale_tick_old():
    assert is_stale_tick({"ts": time.time() - 3601}) is True


def test_read_recent_metrics_empty_file(tmp_path: Path):
    (tmp_path / "m.jsonl").touch()
    assert read_recent_metrics(tmp_path / "m.jsonl", since_ts=0) == []


def test_read_recent_metrics_filters_old(tmp_path: Path):
    m = tmp_path / "m.jsonl"
    now = time.time()
    m.write_text(
        "\n".join(
            json.dumps({"ts": now - t, "step": i, "loss": 1.0}) for i, t in enumerate([1000, 100, 10])
        )
    )
    recent = read_recent_metrics(m, since_ts=now - 200)
    assert len(recent) == 2


def test_check_confirmation_all_pass():
    pre = {"loss": 1.0, "curiosity": 0.5, "num_nodes": 50, "num_edges": 200}
    recent = [{"ts": i, "step": i, "loss": 1.05, "curiosity": 0.52, "num_nodes": 52, "num_edges": 210} for i in range(20)]
    verdict, _, criteria = check_confirmation(pre, recent, window_min=15, min_step_advance=500)
    # Needs >500 step advance — our stub has only 20. So this should NOT confirm.
    assert verdict in {"pending", "fail"}  # short window
```

(Extend with tests for all five criteria and the 3-consecutive-buckets rule.)

**Step 2: Implement**

Implement per design contract. Watchdog structure:

```python
# scripts/auto_revert.py
"""Autonomous-loop watchdog — confirms or reverts in-progress changes."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lock import FileLock

WINDOW_MIN = 15
MIN_STEP_ADVANCE = 500
TOLERANCE_PCT = 0.20
CONFIRMATION_MINUTE_BUCKETS_REQUIRED = 15
REVERT_CONSECUTIVE_BAD_BUCKETS = 3


def is_stale_tick(last_tick: dict[str, object]) -> bool:
    ts = float(last_tick.get("ts", 0))
    return (time.time() - ts) > 3600


def read_recent_metrics(path: Path, *, since_ts: float) -> list[dict[str, object]]:
    if not path.exists():
        return []
    recent: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if float(rec.get("ts", 0)) >= since_ts:
                recent.append(rec)
    return recent


def bucket_by_minute(records: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Group into per-minute buckets (list of record lists), oldest first."""
    if not records:
        return []
    buckets: dict[int, list[dict[str, object]]] = {}
    for rec in records:
        minute_key = int(float(rec.get("ts", 0)) // 60)
        buckets.setdefault(minute_key, []).append(rec)
    return [buckets[k] for k in sorted(buckets.keys())]


def check_confirmation(
    pre: dict[str, object],
    recent: list[dict[str, object]],
    *,
    window_min: int,
    min_step_advance: int,
) -> tuple[str, str | None, dict[str, bool]]:
    """Return (verdict, reason, per-criterion pass/fail map).

    Verdicts:
      - "confirmed": all criteria pass
      - "revert": criterion failed in N consecutive minute-buckets
      - "pending": not enough data yet
    """
    buckets = bucket_by_minute(recent)
    if len(buckets) < 2:
        return "pending", "insufficient buckets", {}

    last_step = max(int(r.get("step", 0)) for r in recent)
    first_step = min(int(r.get("step", 0)) for r in recent)
    if last_step - first_step < min_step_advance:
        return "pending", f"step advance {last_step - first_step} < {min_step_advance}", {}

    pre_loss = float(pre["loss"]) if pre.get("loss") is not None else None
    pre_cur = float(pre.get("curiosity", 0.5))
    pre_nodes = int(pre.get("num_nodes", 0))

    # For per-bucket pass check
    def bucket_pass(bucket: list[dict[str, object]]) -> tuple[bool, dict[str, bool]]:
        losses = [float(r.get("loss", math.nan)) for r in bucket if r.get("loss") is not None]
        curs = [float(r.get("curiosity", 0.5)) for r in bucket]
        nodes = [int(r.get("num_nodes", 0)) for r in bucket]
        avg_loss = (sum(losses) / len(losses)) if losses else math.nan
        avg_cur = (sum(curs) / len(curs)) if curs else 0.0
        min_nodes = min(nodes) if nodes else 0

        crits = {
            "loss_within_20pct": pre_loss is None
            or (not math.isnan(avg_loss) and abs(avg_loss - pre_loss) <= TOLERANCE_PCT * abs(pre_loss)),
            "curiosity_within_20pct": abs(avg_cur - pre_cur) <= TOLERANCE_PCT * abs(pre_cur or 1.0),
            "no_nan_inf": all(not math.isnan(x) and not math.isinf(x) for x in losses),
            "nodes_not_collapsed": min_nodes >= int(0.5 * pre_nodes),
            "step_advancing": True,  # handled above
        }
        return all(crits.values()), crits

    # Count consecutive bad buckets
    bad_streak = 0
    latest_criteria: dict[str, bool] = {}
    for bucket in buckets:
        passed, crits = bucket_pass(bucket)
        latest_criteria = crits
        if passed:
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= REVERT_CONSECUTIVE_BAD_BUCKETS:
                return "revert", f"criteria failed for {bad_streak} consecutive minute-buckets", crits

    # All buckets good AND >= confirmation window
    if len(buckets) >= CONFIRMATION_MINUTE_BUCKETS_REQUIRED and bad_streak == 0:
        return "confirmed", "all criteria held over confirmation window", latest_criteria

    return "pending", "within confirmation window", latest_criteria


def append_change_log(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def update_last_entry_status(path: Path, change_log_id: str, updates: dict[str, object]) -> None:
    """Update the matching change_log entry in-place by rewriting the file.

    Small file (bounded by rate limit); rewriting is acceptable.
    """
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("change_log_id") == change_log_id:
            rec.update(updates)
            lines[i] = json.dumps(rec)
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo = Path(".")
    last_tick_path = repo / ".soma-loop/state/last_tick.json"
    change_log = repo / ".soma-loop/state/change_log.jsonl"
    metrics = repo / ".soma-loop/metrics/metrics.current.jsonl"
    consecutive_fail = repo / ".soma-loop/state/consecutive_failures.json"
    git_lock = repo / ".soma-loop/state/git.lock"
    revert_failure = repo / ".soma-loop/state/revert_failure.json"
    stop = repo / ".soma-loop/STOP"

    if not last_tick_path.exists():
        return 0
    last_tick = json.loads(last_tick_path.read_text())
    if is_stale_tick(last_tick):
        return 0

    # Find last in_progress entry with commit_sha != null
    if not change_log.exists():
        return 0
    lines = change_log.read_text().splitlines()
    target: dict[str, object] | None = None
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "in_progress" and rec.get("commit_sha"):
            target = rec
            break
    if target is None:
        return 0

    # Read metrics since ts_applied
    since_ts = float(target.get("ts_applied") or target.get("ts_proposed") or 0)
    if isinstance(since_ts, str):
        # ISO string; parse
        from datetime import datetime

        since_ts = datetime.fromisoformat(since_ts).timestamp()
    recent = read_recent_metrics(metrics, since_ts=since_ts)
    if not recent:
        return 0  # no data yet

    # Pause detection (G30): if step counter hasn't advanced, skip
    first_step = min(int(r.get("step", 0)) for r in recent)
    last_step = max(int(r.get("step", 0)) for r in recent)
    if last_step == first_step:
        return 0

    pre = target.get("pre_change_metrics", {})
    verdict, reason, criteria = check_confirmation(
        pre, recent, window_min=WINDOW_MIN, min_step_advance=MIN_STEP_ADVANCE
    )

    if verdict == "pending":
        return 0

    change_log_id = str(target["change_log_id"])
    commit_sha = str(target["commit_sha"])

    lock = FileLock(git_lock, timeout=60.0, purpose=f"auto_revert:{verdict}")
    if not lock.acquire():
        print(f"auto_revert: could not acquire git.lock ({verdict})", file=sys.stderr)
        return 0

    try:
        if verdict == "confirmed":
            update_last_entry_status(
                change_log,
                change_log_id,
                {
                    "status": "confirmed",
                    "confirmed_at": time.time(),
                    "post_change_metrics": _summarize_metrics(recent),
                },
            )
            # Copy current.pt -> last_good.pt (atomic)
            cp_cur = Path("checkpoints/current.pt")
            cp_lg = Path("checkpoints/last_good.pt")
            if cp_cur.exists():
                import shutil

                shutil.copy2(cp_cur, cp_lg.with_suffix(".tmp"))
                import os

                os.replace(cp_lg.with_suffix(".tmp"), cp_lg)
            # Reset consecutive_failures
            if consecutive_fail.exists():
                consecutive_fail.write_text(
                    json.dumps({"count": 0, "last_reset_ts": time.time(), "last_failure_ts": None})
                )
            print(f"auto_revert: confirmed {commit_sha[:8]}")
        elif verdict == "revert":
            # git revert
            rev_result = subprocess.run(
                ["git", "revert", "--no-edit", commit_sha],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if rev_result.returncode != 0:
                # Revert failed — G70 path
                revert_failure.parent.mkdir(parents=True, exist_ok=True)
                revert_failure.write_text(
                    json.dumps(
                        {
                            "change_log_id": change_log_id,
                            "commit_sha": commit_sha,
                            "error": rev_result.stderr[:4000],
                            "ts": time.time(),
                        }
                    )
                )
                stop.touch()
                print(f"auto_revert: revert FAILED, STOP set: {rev_result.stderr}", file=sys.stderr)
                return 1
            # Amend commit message to use revert(auto): prefix
            subprocess.run(
                [
                    "git",
                    "commit",
                    "--amend",
                    "-m",
                    f"revert(auto): revert {commit_sha[:8]} ({reason})",
                ],
                check=False,
                timeout=30,
            )
            update_last_entry_status(
                change_log,
                change_log_id,
                {
                    "status": "reverted",
                    "reverted_at": time.time(),
                    "reverted_by": "watchdog",
                    "failure_signal": str(criteria),
                    "post_change_metrics": _summarize_metrics(recent),
                },
            )
            # Increment consecutive_failures
            if consecutive_fail.exists():
                cf = json.loads(consecutive_fail.read_text())
                cf["count"] = int(cf.get("count", 0)) + 1
                cf["last_failure_ts"] = time.time()
                consecutive_fail.write_text(json.dumps(cf))
            # Restore last_good.pt if present
            cp_cur = Path("checkpoints/current.pt")
            cp_lg = Path("checkpoints/last_good.pt")
            if cp_lg.exists():
                import shutil

                shutil.copy2(cp_lg, cp_cur.with_suffix(".tmp"))
                import os

                os.replace(cp_cur.with_suffix(".tmp"), cp_cur)
            # Signal training to shutdown (tick will restart it)
            (Path(".soma-loop/signals") / "shutdown").touch()
            print(f"auto_revert: reverted {commit_sha[:8]} ({reason})")
    finally:
        lock.release()

    return 0


def _summarize_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    losses = [float(r["loss"]) for r in records if r.get("loss") is not None]
    return {
        "count": len(records),
        "last_loss": float(records[-1].get("loss", math.nan)) if records else None,
        "mean_loss": (sum(losses) / len(losses)) if losses else None,
    }


if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_scripts/test_auto_revert.py -v
python scripts/auto_revert.py  # manually; should exit 0 (no in-flight change)
```

**Step 4: Commit**

```bash
git add scripts/auto_revert.py tests/test_scripts/test_auto_revert.py
git commit -m "feat(loop): auto_revert.py watchdog (confirm/revert/revert-fail paths)"
```

---

### Task 13: `scripts/approve.py` + `scripts/reject.py`

**Files:**
- Create: `scripts/approve.py`
- Create: `scripts/reject.py`
- Create: `tests/test_scripts/test_approve.py`

Per design §5.2. Modifies `.soma-loop/state/approval_queue.jsonl` entries.

**Step 1-4: TDD each script**

Write tests first for `--list`, valid approve, stale approve, already-approved rejection, reject with reason. Implement. Verify.

```bash
git add scripts/approve.py scripts/reject.py tests/test_scripts/test_approve.py
git commit -m "feat(loop): approve.py + reject.py for arch-carveout approval flow"
```

---

### Task 14: `scripts/audit_loop.py`

**Files:**
- Create: `scripts/audit_loop.py`
- Create: `tests/test_scripts/test_audit_loop.py`

Per design §5.2. Validates the 7 Phase-1 validation checks.

Each check is a separate function; `main()` runs all and exits 1 if any fails.

```bash
git add scripts/audit_loop.py tests/test_scripts/test_audit_loop.py
git commit -m "feat(loop): audit_loop.py for Phase 1 validation checks"
```

---

## Phase 1D — Shell wrappers + tick prompt

### Task 15: `scripts/run_tick.sh` (Linux/Git Bash)

**Files:**
- Create: `scripts/run_tick.sh`

**Implementation:**

```bash
#!/usr/bin/env bash
# run_tick.sh — invoke claude --print with the tick prompt; derive exit code from state.

set -u

cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

LOG_DIR=".soma-loop/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/tick-$(date +%s).log"

# Clear last-tick failure markers
rm -f .soma-loop/state/tick_failure.json

ALLOWED="Bash(git:*) Bash(python:*) Bash(ruff:*) Bash(mypy:*) Bash(pytest:*) Bash(taskkill:*) Bash(kill:*) Bash(curl:*) Read Edit Write Grep Glob Skill"
DISALLOWED="Agent WebFetch WebSearch NotebookEdit"

# Prompt lives in docs/loop_tick_prompt.md
if [ ! -f docs/loop_tick_prompt.md ]; then
  echo "FATAL: docs/loop_tick_prompt.md missing" >&2
  exit 2
fi

claude --print --dangerously-skip-permissions \
  --allowedTools "$ALLOWED" \
  --disallowedTools "$DISALLOWED" \
  --prompt "$(cat docs/loop_tick_prompt.md)" \
  > "$LOG" 2>&1 || true

# Detect Claude CLI failure patterns (G71)
if grep -qiE "rate limit|authentication|expired|quota exceeded" "$LOG"; then
  mkdir -p .soma-loop/state
  cat > .soma-loop/state/claude_cli_error.json <<EOF
{"ts": $(date +%s), "log": "$LOG"}
EOF
fi

# Derive exit code
if [ -f .soma-loop/state/tick_failure.json ]; then
  exit 2
elif [ -f .soma-loop/STOP ] || [ -f .soma-loop/state/baseline_broken.json ]; then
  exit 1
else
  exit 0
fi
```

Set executable: `chmod +x scripts/run_tick.sh`.

```bash
git add scripts/run_tick.sh
git commit -m "feat(loop): run_tick.sh wrapper (Linux/Git Bash)"
```

---

### Task 16: `scripts/run_tick.ps1` (Windows PowerShell)

**Files:**
- Create: `scripts/run_tick.ps1`

Mirror `.sh` behavior:

```powershell
# run_tick.ps1 — invoke claude --print with the tick prompt; derive exit code from state.

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = ".soma-loop/logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "tick-$([int](Get-Date -UFormat %s)).log"

# Clear last-tick failure markers
Remove-Item -ErrorAction SilentlyContinue ".soma-loop/state/tick_failure.json"

$allowed = "Bash(git:*) Bash(python:*) Bash(ruff:*) Bash(mypy:*) Bash(pytest:*) Bash(taskkill:*) Bash(kill:*) Bash(curl:*) Read Edit Write Grep Glob Skill"
$disallowed = "Agent WebFetch WebSearch NotebookEdit"

if (-not (Test-Path "docs/loop_tick_prompt.md")) {
    Write-Error "FATAL: docs/loop_tick_prompt.md missing"
    exit 2
}

$prompt = Get-Content -Raw "docs/loop_tick_prompt.md"

claude --print --dangerously-skip-permissions `
    --allowedTools $allowed `
    --disallowedTools $disallowed `
    --prompt $prompt *> $log

# Detect Claude CLI failure patterns (G71)
if (Select-String -Path $log -Pattern "rate limit|authentication|expired|quota exceeded" -Quiet) {
    New-Item -ItemType Directory -Force -Path ".soma-loop/state" | Out-Null
    $err = @{ ts = [int](Get-Date -UFormat %s); log = $log } | ConvertTo-Json -Compress
    Set-Content -Path ".soma-loop/state/claude_cli_error.json" -Value $err
}

if (Test-Path ".soma-loop/state/tick_failure.json") { exit 2 }
if ((Test-Path ".soma-loop/STOP") -or (Test-Path ".soma-loop/state/baseline_broken.json")) { exit 1 }
exit 0
```

```bash
git add scripts/run_tick.ps1
git commit -m "feat(loop): run_tick.ps1 wrapper (Windows PowerShell)"
```

---

### Task 17: `docs/loop_tick_prompt.md`

**Files:**
- Create: `docs/loop_tick_prompt.md`

Per design §4 "Tick prompt skeleton". Full text with all 11 phases, hard rules, and exit semantics.

Write comprehensively — this is what Claude reads every tick. Key sections:
1. Preamble (who you are, what tools you have)
2. Hard rules (never bypass, never write carveout, always use git.lock)
3. Phase checklist (phases 0-11 with sub-steps and failure branches)
4. Common failure modes + handling
5. Exit codes

Reference design doc §4 as source of truth for the exact phase logic.

```bash
git add docs/loop_tick_prompt.md
git commit -m "docs(loop): loop_tick_prompt.md for each tick Claude session"
```

---

## Phase 1E — Skills

### Task 18: `soma-diagnose` skill

**Files:**
- Create: `~/.claude/skills-repo/skills/domain-specific/soma-diagnose/SKILL.md`
- Modify: `~/.claude/skills/REGISTRY.json`
- Copy registry back: `cp ~/.claude/skills-repo/REGISTRY.json ~/.claude/skills/REGISTRY.json`

Per design §5.1 `soma-diagnose` contract.

Author the SKILL.md per `skill-creator` conventions (check `skill-creator` SKILL.md). 80-150 line budget. Include:
- Frontmatter `name:` + `description:` with keyword triggers (`SOMA`, `diagnose`, `autonomous loop`, `tick report`)
- Process steps: read tick report, read change_log subset, query wiki-knowledge, write diagnosis.md
- Hard rules (cite metrics, check reverted entries, 30s wiki timeout, --max-time 10 on curl)
- Example diagnosis format template

Commit to the skills-repo, update REGISTRY.json, copy to local `~/.claude/skills/REGISTRY.json`.

```bash
# In E:/Documents/Projects/SOMA (this repo is unchanged)
# Workflow:
# 1. edit ~/.claude/skills-repo/skills/domain-specific/soma-diagnose/SKILL.md
# 2. edit ~/.claude/skills-repo/REGISTRY.json to add entry
# 3. commit + push the skills-repo
# 4. cp ~/.claude/skills-repo/REGISTRY.json ~/.claude/skills/REGISTRY.json
```

(No SOMA repo commit for this task — skills live in a separate repo.)

---

### Task 19: `soma-propose-change` skill

Same workflow as Task 18. 100-150 line SKILL.md per design §5.1. Triggers: `SOMA`, `propose change`, `change spec`.

---

### Task 20: `soma-bootstrap-kb` skill

Same workflow. 120-180 lines per design §5.1. Uses Gitea API directly (curl). Triggers: `SOMA`, `bootstrap wiki`, `SOMA knowledge base`.

This skill also needs `wiki-knowledge` read in its process (to inspect existing atom format for consistency).

---

## Phase 1F — UI spectator mode + runbook

### Task 21: UI spectator-mode patch

**Files:**
- Modify: `src/soma/ui/panels/controls_panel.py`
- Modify: `src/soma/ui/panels/config_panel.py`
- Modify: `src/soma/ui/panels/chat_panel.py`
- Modify: `src/soma/ui/panels/metrics_panel.py` + `graph_panel.py` (read from disk)
- Modify: `src/soma/ui/state.py` (add `autonomous_mode: bool` field)
- Create: `tests/test_ui/test_spectator_mode.py`

When `.soma-loop/state/autonomous_mode.flag` exists at UIState construction:
- `UIState.autonomous_mode = True`
- Controls panel disables Start/Pause/Stop/Step/Reset/Apply+Rebuild buttons (show a banner: "Autonomous mode — read-only")
- Config panel disables all inputs
- Metrics panel reads from `.soma-loop/metrics/metrics.current.jsonl` (tail N lines) instead of bus
- Graph panel reads last snapshot file from `.soma-loop/reports/` if available
- Chat panel loads latest checkpoint in a CPU-only copy for inference (no GPU contention)

Tests: construct UIState with the flag file present, verify `autonomous_mode` is True, verify panels respect the flag (you'll need headless DPG-mock tests per existing pattern).

```bash
git add src/soma/ui/ tests/test_ui/test_spectator_mode.py
git commit -m "feat(loop): UI spectator mode when autonomous_mode.flag present"
```

---

### Task 22: `docs/AUTONOMOUS_LOOP.md` — operator runbook

**Files:**
- Create: `docs/AUTONOMOUS_LOOP.md`

Sections:

1. Overview (1 para)
2. Prerequisites (Python 3.11+, CUDA, Claude Code authenticated, Gitea token in settings.local.json)
3. First-time setup (the §6.1 pre-launch checklist from the design doc)
4. Starting the loop
5. Monitoring (what to watch: tick-summaries.md, heartbeat, git log)
6. Common operations (start/stop/approve/reject, emergency rollback)
7. Troubleshooting
8. Phase 1 → Phase 2 transition runbook (per design §7.1)
9. Cloud migration (per design §7.4)

Reference the design doc's §6.1 for the checklist; do not duplicate.

```bash
git add docs/AUTONOMOUS_LOOP.md
git commit -m "docs(loop): operator runbook"
```

---

## Phase 1G — End-to-end validation

### Task 23: Full pre-launch checklist dry-run

Execute the pre-launch checklist from design §6.1 against the implemented system:

- [ ] `python -m pytest tests/ -q` passes 10 consecutive times
- [ ] `python scripts/smoke_train.py --steps 10 --device cuda` succeeds
- [ ] `python scripts/safety_gate.py --skip-smoke` passes
- [ ] `python scripts/bootstrap_loop.py` completes
- [ ] `data/fixed_prompts.txt` reviewed
- [ ] `claude --print --prompt "Use soma-bootstrap-kb skill..."` populates wiki
- [ ] Gitea wiki shows SOMA entity + ≥10 atoms
- [ ] `configs/current.yaml` present
- [ ] No STOP / baseline_broken files
- [ ] `python scripts/train_service.py` starts cleanly, heartbeat updates, ≥1 min of metrics recorded
- [ ] Single manual tick: `bash scripts/run_tick.sh` → exit 0
- [ ] `python scripts/auto_revert.py` → exit 0

Any failure: investigate and fix before proceeding.

**Commit:** Update `reports/tick-summaries.md` with Phase 1 launch-readiness entry, commit:

```bash
git add reports/tick-summaries.md
git commit -m "docs(loop): Phase 1 launch-readiness verified"
```

---

### Task 24: Create Task Scheduler entries (Windows)

Follow design Appendix D.1. Create two tasks, keep DISABLED.

Touch `.soma-loop/state/autonomous_mode.flag` to activate UI spectator mode.

Document the Task Scheduler settings in `docs/AUTONOMOUS_LOOP.md`.

No repo commit — these are system settings.

---

### Task 25: Inject the two validation tests (§6.3)

**Revert-path test:**
1. `touch .soma-loop/STOP` to halt autonomous ticks
2. Manually append to `approval_queue.jsonl` a change spec setting `base_lr: 0.5` in `configs/current.yaml`
3. `python scripts/approve.py <id>`
4. `rm .soma-loop/STOP`
5. Enable Task Scheduler entries
6. Wait for next tick to apply; within 15-20 min, watchdog reverts
7. Verify `change_log.jsonl` shows `status="reverted"` with `revert(auto):` commit in git log

**Gate-failure test:**
1. `touch .soma-loop/STOP`
2. Queue a synthetic change with an intentional syntax error in a non-carveout file
3. `python scripts/approve.py <id>`
4. `rm .soma-loop/STOP`
5. Next tick applies, Phase 8 gate fails, status=reverted_at_gate with `failure_signal="check"`

Document both test runs in `reports/tick-summaries.md`.

---

### Task 26: 48-hour production run + audit

Enable the scheduled tasks. Let loop run for 48 continuous hours.

At end:
1. Touch STOP
2. Wait for current tick to finish
3. Run `python scripts/audit_loop.py`
4. Verify all §6.2 criteria pass
5. If all pass: mark Phase 1 complete in `reports/tick-summaries.md`; proceed to plan Phase 2 (Approach 2 / Corpus scale-up)
6. If any fail: investigate; iterate; re-run

---

## Task summary

| # | Task | Deliverable |
|---|------|-------------|
| 1 | portalocker dep | `pyproject.toml` + smoke test |
| 2 | `scripts/lock.py` | Cross-platform FileLock |
| 3 | `.gitignore` | Runtime state ignored |
| 4 | `scripts/bootstrap_loop.py` | Setup + seed files |
| 5-8 | `scripts/train_service.py` | Long-lived daemon |
| 9 | `scripts/smoke_train.py` | Gate sanity check |
| 10 | `scripts/safety_gate.py` | Gate orchestrator |
| 11 | `scripts/test_harness.py` | B-tier eval |
| 12 | `scripts/auto_revert.py` | Watchdog |
| 13 | `scripts/approve.py` + `reject.py` | Arch-carveout approval |
| 14 | `scripts/audit_loop.py` | Validation oracle |
| 15 | `scripts/run_tick.sh` | Linux wrapper |
| 16 | `scripts/run_tick.ps1` | Windows wrapper |
| 17 | `docs/loop_tick_prompt.md` | Per-tick prompt |
| 18 | `soma-diagnose` skill | In user skills-repo |
| 19 | `soma-propose-change` skill | In user skills-repo |
| 20 | `soma-bootstrap-kb` skill | In user skills-repo |
| 21 | UI spectator-mode patch | Read-only UI |
| 22 | `docs/AUTONOMOUS_LOOP.md` | Operator runbook |
| 23 | Pre-launch checklist | All green |
| 24 | Task Scheduler entries | Scheduled (disabled) |
| 25 | Validation injections | Revert + gate-fail tests |
| 26 | 48-hour run + audit | Phase 1 complete |

**Total: 26 tasks across 7 phases (1A-1G).** Tasks 5-8 (train_service) are the largest component. Tasks 18-20 (skills) operate on an external repo.

**Design doc is authoritative for all behavior.** If any step in this plan contradicts the design doc, the design doc wins. File bugs in plan, not deviations in implementation.

---

*End of implementation plan.*
