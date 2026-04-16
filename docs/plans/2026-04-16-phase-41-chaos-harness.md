# Phase 41: Power-Loss / Ungraceful-Shutdown Chaos Harness

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Stress-test the crash-safety invariants SOMA already claims:
atomic bundle writes, WAL replay, compaction atomicity, GDPR
forget-cascade integrity. Use `SIGKILL` (not `SIGTERM`) to simulate
real power loss — no `atexit`, no `__exit__`, no flush-on-exit. After
each kill, a fresh process loads the bundle and verifies invariants
hold.

**Architecture:**
- `tests/chaos/` new package with chaos-specific infrastructure:
  * `runner.py` — spawns a subprocess running a target script, waits
    for a "ready" signal (process prints a sentinel to stdout), then
    `os.kill(pid, signal.SIGKILL)`. Cross-platform: Windows uses
    `signal.CTRL_BREAK_EVENT` as a SIGKILL-equivalent, but the
    filesystem atomicity guarantees differ — gate Windows with a
    `skipif` unless the operator explicitly opts in.
  * `scenarios/` — one file per kill-during-X test. Each scenario is
    a script-style target that: (a) sets up a temp bundle, (b) prints
    a "START_<scenario>" sentinel, (c) performs the risky operation,
    (d) prints "DONE_<scenario>" if it survives. Runner kills between
    START and DONE.
- Test matrix (v1):
  1. **Kill during `mem.save()`**: bundle is either fully old or
     fully new, never a half-written mix.
  2. **Kill during `mem.add()`** (WAL write): WAL replay on load
     recovers the entry.
  3. **Kill during compaction**: bundle loads cleanly and has the
     same entries as before compaction started.
  4. **Kill during `mem.forget()`**: no orphan facts (fact whose
     `source_turn_id` points at a deleted turn).
  5. **Kill during `ConversationalMemory.forget()` cascade** (Phase
     34-37): audit record either written or not written, never a
     torn JSONL line.
- Each scenario runs N=10 iterations; a single pass counts as
  success, a single fail counts as fail. Randomised kill timing
  within a window to shake out timing-dependent races.
- Gated by `SOMA_CHAOS=1` env. Default `pytest` run executes a single
  smoke case (scenario #1 at N=1) to catch obvious breakage. Full
  matrix runs on the nightly.
- Skipped on Windows unless `SOMA_CHAOS_WINDOWS=1` is set — POSIX
  `os.replace` has cross-file-system atomicity guarantees that
  Windows NTFS doesn't always match.

**Out-of-scope:**
- Actual disk power-loss (fsync ordering, disk-write reordering).
  Simulating those without real hardware is unreliable. We assume
  `os.replace` + OS-level crash consistency, which is the normal
  Linux/macOS guarantee.
- Network partitions. Different failure class.
- Corruption injection (bit-flips, truncated files). Separate harness
  if ever needed.

**Out-of-scope (central merge):** `CHANGELOG.md`, `deferred-items.md`.

---

### Task 1: Chaos runner infrastructure

**Files:**
- Create: `tests/chaos/__init__.py`
- Create: `tests/chaos/runner.py`
- Create: `tests/chaos/conftest.py` (shared fixtures)
- Create: `tests/chaos/test_runner_self_test.py`

**Runner API:**
```python
@dataclass
class ChaosResult:
    scenario: str
    kill_at_ms: int        # time within the risky op when SIGKILL fired
    exit_code: int         # the (killed) subprocess's exit code
    survived: bool         # did the invariant hold?
    details: str

def run_chaos_scenario(
    target: str,           # path to a script
    *, kill_window_ms: tuple[int, int],
    invariant: Callable[[Path], None],
    iterations: int = 10,
) -> list[ChaosResult]: ...
```

**Self-tests (make sure the runner itself works):**
```python
def test_runner_kills_subprocess(tmp_path):
    # Target script that sleeps forever. Runner should kill and
    # surface the expected exit-code (negative on POSIX SIGKILL).

def test_runner_reports_survived_when_invariant_passes(tmp_path):
    # Target writes "hello" to a file at start. Invariant asserts
    # the file exists + contains "hello". Kill after write → invariant
    # passes → survived=True.

def test_runner_reports_not_survived_when_invariant_fails(tmp_path):
    # Same but kill BEFORE the write. Invariant fails → survived=False.

def test_randomised_kill_timing_covers_window(tmp_path):
    # Over N=20 runs with kill_window_ms=(0, 100), the kill timestamps
    # should distribute across the window (not all clustered).
```

**Step 5:** `git commit -m "test(chaos): subprocess-kill runner + self-tests"`

---

### Task 2: Scenario 1 — kill during `mem.save()`

**Files:**
- Create: `tests/chaos/scenarios/kill_during_save.py` (target script)
- Create: `tests/chaos/test_save_atomicity.py`

**Target script:**
```python
# kill_during_save.py
import sys, os, json
from soma.memory import MemoryLayer

path = sys.argv[1]
mem = MemoryLayer(...)
mem.add(["hello", "world"], embeddings)
print("START_save", flush=True)
mem.save(path)
print("DONE_save", flush=True)
```

**Invariant:** after a kill during save, `MemoryLayer.load(path)` must
either:
(a) succeed and contain the pre-save state (bundle is unchanged), OR
(b) succeed and contain the new state (bundle is fully committed).
The failure mode is: load raises, or loads a half-written state.

**Tests:**
```python
@pytest.mark.chaos
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX replace semantics")
def test_save_survives_sigkill_mid_write(tmp_path):
    results = run_chaos_scenario(
        "tests/chaos/scenarios/kill_during_save.py",
        kill_window_ms=(5, 80),
        invariant=_load_bundle_and_check_consistency,
        iterations=10,
    )
    assert all(r.survived for r in results), _format_failures(results)
```

**Step 5:** `git commit -m "test(chaos): SIGKILL during save() preserves atomicity"`

---

### Task 3: Scenarios 2-5

**Files:**
- Create: `tests/chaos/scenarios/kill_during_add.py` (+ test)
- Create: `tests/chaos/scenarios/kill_during_compaction.py` (+ test)
- Create: `tests/chaos/scenarios/kill_during_forget.py` (+ test)
- Create: `tests/chaos/scenarios/kill_during_conv_forget.py` (+ test)

Each scenario follows the Task-2 pattern: target script prints START
+ sentinel, runner kills, invariant check runs in a fresh subprocess
to avoid any in-memory contamination.

Invariants per scenario:
- `kill_during_add`: WAL contains the intended entry OR the entry
  isn't observable. No partial WAL lines.
- `kill_during_compaction`: bundle still loads; entry count
  post-crash equals pre-crash; compaction can retry cleanly.
- `kill_during_forget`: no orphan facts post-crash; either the fact
  and its turn are both deleted, or both present.
- `kill_during_conv_forget`: audit JSONL has no torn lines; either
  the full record is present or absent.

**Step 5 (per scenario):**
- `git commit -m "test(chaos): SIGKILL during add() preserves WAL"`
- `git commit -m "test(chaos): SIGKILL during compaction is retry-safe"`
- `git commit -m "test(chaos): SIGKILL during forget() has no orphan facts"`
- `git commit -m "test(chaos): SIGKILL during conv forget preserves audit integrity"`

---

### Task 4: CI nightly hook

**Files:**
- Modify: `.gitea/workflows/bench-regression.yml` (or a separate
  `.gitea/workflows/chaos.yml` — agent's call; a separate workflow is
  cleaner if chaos runtime is substantial)

**Shape:**
```yaml
chaos:
  runs-on: ubuntu-latest
  if: ${{ github.event_name == 'schedule' }}
  env:
    SOMA_CHAOS: "1"
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: { python-version: '3.12' }
    - run: pip install -e ".[dev,sbert,ann]"
    - run: pytest tests/chaos -v
```

**Step 5:** `git commit -m "ci(chaos): weekly chaos harness run"`

---

### Task 5: Docs

**Files:**
- Modify: `docs/observability.md` — new "Crash safety" section
  explaining the invariants SOMA guarantees, the test matrix, how to
  re-run locally, and what a failure means.
- Update: `README.md` feature table — add "Crash-safe WAL + atomic
  bundle writes (chaos-tested)" or tighten the existing row's
  language now that we have real tests.

**Step 5:** `git commit -m "docs(observability): crash-safety guarantees + chaos matrix"`

---

### Final sanity

```bash
ruff check tests/chaos
SOMA_CHAOS=1 pytest tests/chaos -v    # full matrix, POSIX only
pytest tests/chaos -v                  # default: smoke case only
```

Target: +~15 new tests (5 scenarios × ~3 tests each), all gated.
Default pytest adds +1 smoke test. Zero regressions.

**Gotchas:**
- Subprocess startup is expensive (~1-2 seconds of Python interp +
  torch import). At N=10 per scenario × 5 scenarios = 50 spawns. The
  full matrix will take minutes, not seconds. Acceptable for a
  nightly; wouldn't run on every PR.
- Kill-timing flakiness: the "risky op" window may be shorter than
  the scheduler grants. Print the kill timestamp and the DONE
  timestamp (if reached) so flakes show up as "kill fired AFTER
  DONE, scenario trivially survived without testing the race."
- On Windows, `os.kill(pid, signal.CTRL_BREAK_EVENT)` isn't SIGKILL
  — the child gets a chance to cleanup. That defeats the point. Mark
  Windows as opt-in via `SOMA_CHAOS_WINDOWS=1` and document the
  caveat: Windows chaos is best-effort, not the real power-loss
  semantics we test on POSIX.
- Parallel test runners (`pytest-xdist`): chaos scenarios shouldn't
  run in parallel with themselves — some scenarios touch shared
  resources (file handles, network ports) in ways that can race with
  the runner. Pin `@pytest.mark.chaos` to serial execution.
- Subprocess inherits env — make sure `SOMA_EMBED_MODEL=stub` is
  passed so the kill happens during the actual `save()`/`add()` and
  not during sentence-transformer load.
