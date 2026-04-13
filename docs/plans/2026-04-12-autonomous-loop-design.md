# Autonomous SOMA Improvement Loop — Design Document

**Date:** 2026-04-12
**Status:** Approved — ready for implementation planning
**Phase:** 1 (local 3090, TinyShakespeare → light scale-up)
**Supersedes:** n/a (first autonomous-loop design for SOMA)
**Related:** `docs/plans/2026-04-12-autonomous-build-design.md` (prior build), `docs/whitepaper.md`, `docs/AUTONOMOUS_LOOP.md` (operator runbook — created alongside implementation)

---

## 0. Executive Summary

### What we're building

A Claude-in-the-middle autonomous improvement loop for SOMA. OS-level task scheduling fires short-lived `claude --print` sessions on a cadence (default 37 minutes). Each session runs a structured 11-phase tick: read state, run test harness, diagnose, propose change, apply, gate-check, commit, restart training, finalize, wiki-sync. A long-lived `train_service.py` daemon runs SOMA continuously between ticks. A 2-minute `auto_revert.py` watchdog confirms or reverts changes based on post-change metric behavior.

### Why

SOMA is a research prototype. Its path to conversational capability is unknown — may plateau at phrase-level, may reach token-frequency quality, may (with enough scale) approach GPT-2-small / SmolLM2-135M territory. The loop converts "train hard and hope" into "systematically iterate and discover what SOMA can do" — and gives us compounding gains across iterations via a knowledge wiki. The loop itself is the deliverable; whatever SOMA becomes is the emergent outcome.

### Primary deliverable

The loop, not the chat quality. Phase 1 exits when the loop has demonstrated stability, safety, effectiveness, and auditability over 48 continuous hours. Subsequent phases (corpus scale-up, curriculum, cloud) use the same loop against progressively harder substrates.

### Target milestone (north star, not Phase 1 exit)

Conversational capability on par with the smallest existing LLM that genuinely holds a conversation (SmolLM2-135M / Qwen2.5-0.5B-Instruct tier). Honest expectation: architecturally unvalidated for SOMA; the loop will tell us whether that's reachable or where the plateau is.

---

## 1. Context & Constraints

### Environment
- Local development: Windows 11, NVIDIA RTX 3090 (24 GB VRAM), bash shell, Python 3.11+
- Cloud escape hatch: Linux VM (A10G / L40S / A100 / H100) when local becomes compute-limited
- Authentication substrate: Claude Max subscription (not API); interactions via `claude --print` CLI
- Repo: git-managed, Gitea remote at `git.dant123.com/dant123/soma`
- Knowledge wiki: Gitea at `git.dant123.com/dant123/knowledge-wiki` (Karpathy-style atomic notes: `atoms/`, `molecules/`, `entities/`, `moc/`, `sources/`)

### Approach sequencing

| Approach | Description | Decision |
|---|---|---|
| **1** | Loop infrastructure first, scale corpus later | **Selected for Phase 1** |
| **2** | Loop + corpus scale-up in parallel | Documented (future, see §7) |
| **3** | Loop + developmental curriculum | Documented (future, see §7) |

---

## 2. Frozen Design Decisions

| # | Decision | Value |
|---|---|---|
| 1 | Primary deliverable | The loop is the deliverable |
| 2 | Autonomy level | **Level 4** (fully autonomous) with multi-layer safety backstop (see row 8) |
| 3 | Architecture carveout | **Brain modules:** `src/soma/core/**`, `src/soma/memory/**`, `src/soma/growth/**`, `src/soma/system.py`. **Checkpoint-shape keys** in `configs/current.yaml` (full list in Phase 6). **Data:** `data/*.txt`, `data/*.jsonl`. **Loop infrastructure:** `scripts/**`, `tests/**`, `pyproject.toml`, `CLAUDE.md`, and this design doc. All require human approval via approval queue. Metacognition + consolidation deliberately NOT in carveout for Phase 1 (see §8.1). |
| 4 | Wiki-knowledge integration | Bootstrap at start (SOMA entity + atoms); read wiki each diagnose tick; write summary each change-tick via `wiki-sync` |
| 5 | Custom skills | `soma-diagnose`, `soma-propose-change`, `soma-bootstrap-kb` |
| 6 | Test harness | **B-tier** now (held-out loss/perplexity/KL + fixed-prompt regression + graph/memory health). C-tier documented for future. |
| 7 | Loop orchestration | Windows Task Scheduler + `claude --print` (via `run_tick.sh` + `run_tick.ps1` wrappers). For Linux cloud: `systemd timer` preferred, cron `*/30` as fallback (cron can't express 37-min cleanly). Python + Anthropic SDK path ("Orchestration D") only if API access obtained. |
| 8a | **Pre-flight gates** (block tick before change attempted) | STOP sentinel; baseline_broken.json; rate limit (≤6 `^auto:` commits in last hour); baseline gate (ruff/mypy/pytest on pristine tree); tick.lock; heartbeat-alive check. Any fail → tick exits without making changes. |
| 8b | **Per-change gates** (all must pass before a change is committed) | 1. `ruff format src/ tests/`, 2. `ruff check src/ tests/`, 3. `mypy src/soma/`, 4. `pytest -q`, 5. Smoke train (30 steps, CUDA, finite loss via `scripts/smoke_train.py`). Any fail → stash + drop, change_log status=reverted_at_gate, consecutive_failures++. |
| 8c | **Post-change backstop** | Auto-revert watchdog runs every 2 min; observes 15-min window of post-change metrics; confirms (status=confirmed, reset consecutive_failures) or reverts (`revert(auto):` commit, consecutive_failures++). |
| 8d | **Circuit breaker** | `consecutive_failures.count >= 3` → tick Phase 0 auto-touches STOP; operator investigation required. |
| 9 | Schedule cadence | Tick every 37 min (off-minute), watchdog every 2 min |

---

## 3. Architecture & Data Flow

### High-level picture

```
  Windows Task Scheduler (or Linux cron) fires every 37 min
                            |
                            v
                  scripts/run_tick.sh (or .ps1 wrapper)
                            |
                            v
       claude --print --prompt "$(cat docs/loop_tick_prompt.md)"
                --dangerously-skip-permissions
                --allowedTools "<whitelist>"
                --disallowedTools "Agent WebFetch WebSearch NotebookEdit"
                            |
                            v
            Fresh Claude Code session, 1-5 min typical
            Executes 11 phases, writes state files,
            invokes skills, runs subprocess Python scripts
                            |
                            v
+---------------------------+  +------------------------------+
| train_service.py (daemon) |  | auto_revert.py (watchdog,    |
| long-lived, operator-     |  | separate scheduler, 2 min)   |
| started once; recovered   |  | - tails metrics              |
| by tick if heartbeat stale|  | - confirms OR reverts based  |
| - polls signals/          |  |   on 15-min criteria         |
| - atomic checkpoints      |  | - acquires git.lock for all  |
| - heartbeat every 10s     |  |   change_log writes (both    |
|                           |  |   confirm and revert paths)  |
+---------------------------+  +------------------------------+
```

### Directory layout

```
SOMA/
├── src/soma/…                              # existing — unchanged
├── configs/
│   ├── default.yaml                        # pristine reset (committed)
│   └── current.yaml                        # live mutable config (committed)
├── scripts/
│   ├── train_service.py                    # NEW: long-lived daemon
│   ├── test_harness.py                     # NEW: B-tier evaluation
│   ├── safety_gate.py                      # NEW: ruff/mypy/pytest/smoke orchestrator
│   ├── smoke_train.py                      # NEW: 30-step sanity check
│   ├── auto_revert.py                      # NEW: regression watchdog
│   ├── approve.py                          # NEW: releases arch change from queue
│   ├── reject.py                           # NEW: rejects queued change
│   ├── bootstrap_loop.py                   # NEW: one-time setup
│   ├── audit_loop.py                       # NEW: validation scripts
│   ├── lock.py                             # NEW: cross-platform file-lock helper (library)
│   ├── run_tick.sh                         # NEW: tick wrapper (Linux/Git Bash)
│   └── run_tick.ps1                        # NEW: tick wrapper (Windows PowerShell)
├── data/
│   ├── tinyshakespeare.txt                 # existing
│   ├── heldout.txt                         # NEW: deterministic 10% split (committed)
│   ├── fixed_prompts.txt                   # NEW: 20 canonical prompts (committed)
│   └── corpus_token_freq.json              # NEW: KL reference (committed)
├── docs/
│   ├── plans/2026-04-12-autonomous-loop-design.md  # this file
│   ├── AUTONOMOUS_LOOP.md                  # NEW: operator runbook
│   └── loop_tick_prompt.md                 # NEW: the prompt each tick receives
├── reports/
│   └── tick-summaries.md                   # NEW: human-readable running log (committed)
├── checkpoints/                            # gitignored
│   ├── current.txt                         # pointer: filename of current ckpt
│   ├── current.pt                          # atomic os.replace target
│   ├── last_good.pt                        # survived 15-min post-change window
│   └── step_NNNNN.pt                       # retained per G35 policy
├── .soma-loop/                             # gitignored (runtime state)
│   ├── STOP                                # presence halts loop ticks
│   ├── signals/
│   │   ├── shutdown                        # train_service: save + exit
│   │   ├── reload_config                   # train_service: swap current.yaml (future)
│   │   ├── pause                           # train_service: stop stepping
│   │   └── resume
│   ├── state/
│   │   ├── autonomous_mode.flag            # UI reads: spectator mode if present
│   │   ├── baseline.json                   # harness output at cold start
│   │   ├── baseline_broken.json            # gates failed on pristine tree
│   │   ├── change_log.jsonl                # every autonomous change + outcome
│   │   ├── approval_queue.jsonl            # arch changes awaiting /approve
│   │   ├── consecutive_failures.json       # circuit breaker counter
│   │   ├── train_heartbeat.json            # {ts, step, status, device}
│   │   ├── tick_failure.json               # last unhandled failure trace
│   │   ├── halt_reason.json                # written on class=halt
│   │   ├── revert_failure.json             # watchdog revert couldn't complete
│   │   ├── push_failure.json               # last push error
│   │   ├── claude_cli_error.json           # Max auth / rate-limit detection
│   │   ├── fresh_init.flag                 # train_service started without checkpoint
│   │   ├── train_crash.json                # service exception dump
│   │   ├── train_permanent_failure.json    # 3 crashes in 5 min
│   │   ├── gate_failure.json               # last safety_gate failure summary
│   │   ├── git.lock                        # {pid, ts, purpose}
│   │   ├── tick.lock                       # prevents overlapping ticks
│   │   ├── last_tick.json                  # {tick_id, commit_sha, change_log_id, ts}
│   │   └── ticks/tick-<tick_id>/                 # per-tick workspace (30-day retention)
│   │       ├── pending_change.json
│   │       ├── diagnosis.md
│   │       └── network_warnings.log
│   ├── pid/train_service.pid
│   ├── metrics/
│   │   ├── metrics.current.jsonl
│   │   └── metrics.YYYY-MM-DD.jsonl        # rotated daily
│   ├── reports/
│   │   ├── tick-<tick_id>.json                   # raw harness output
│   │   └── chat/tick-<tick_id>.jsonl             # fixed-prompt generations
│   └── logs/
│       ├── tick-<tick_id>.log                    # claude --print stdout/stderr
│       └── train_service.log
└── pyproject.toml                          # +1 dep: portalocker
```

### What's committed vs gitignored

| Committed | Gitignored |
|---|---|
| All code (`src/`, `scripts/`) | `checkpoints/` (bulk, transient) |
| All config (`configs/*.yaml`) | `.soma-loop/` (all runtime state, including `approval_queue.jsonl`) |
| All data (corpus, heldout, prompts, KL ref) | |
| All docs | |
| `reports/tick-summaries.md` | |

**Note on `approval_queue.jsonl`:** lives inside `.soma-loop/state/` and is gitignored like other runtime state. User approvals leave a commit trail via the arch-change commit itself when eventually applied; the queue entries are transient. If cross-machine visibility is needed, promote to committed file in Phase 2+.

### Authoritative state

The git commit log is the primary source of truth. Every autonomous change commit carries a `Change-log-id: <uuid>` trailer plus metadata (rationale, pre-change metrics, expected effect). `.soma-loop/state/change_log.jsonl` is a local cache; on a fresh machine, it's rehydrated by scanning:

- `git log --grep="Change-log-id:" --grep="^auto:"` — captures the apply/commit moment with full rationale
- `git log --grep="^revert(auto):"` — captures revert events (pair by referenced Change-log-id)
- Missing from commit trailers: `failure_signal` on gate failures (change was stashed, never committed) — these are local-cache-only and lost on fresh-machine rehydration. Accept this limitation; gate failures are tactical signals (fix-and-retry), not strategic (long-term record).

---

## 4. Tick Flow (11 phases)

### Full phase sequence

**Phase 0 — Pre-flight** (any failure → tick exits cleanly without changes)
- 0a. Check `.soma-loop/STOP` or `state/baseline_broken.json` — if present, exit 1
- 0b. **Circuit breaker:** read `state/consecutive_failures.json`; if `count >= 3`, touch `STOP` (for operator alerting) and exit 1
- 0c. Rate limit: `git log --since="1 hour ago" --grep="^auto:" -E | wc -l` must be ≤6. Subject prefix `^auto:` is rate-limited; `chore(auto):` and `revert(auto):` are not.
- 0d. Baseline gate (read-only): `python scripts/safety_gate.py --skip-smoke`. Failure → touch `STOP` + `state/baseline_broken.json`, exit 1.
- 0e. Acquire `.soma-loop/state/tick.lock` (PID + timestamp format, 30s timeout, stale = PID dead or >30 min old). Held until tick exits.
- 0f. Read `state/train_heartbeat.json`:
  - Missing or stale (>60s) → attempt restart via Phase 10 logic; if restart fails, exit 2
  - `status == "warming_up"` → exit 0 (try again next schedule)
  - `status == "running"` → proceed
- 0g. Compute `tick_id = int(time.time())` (unix timestamp seconds). Create tick workspace `.soma-loop/state/ticks/tick-<tick_id>/`. All per-tick files use this id (report, chat log, tick log). Monotonic across restarts and machines.

**Phase 1 — Harness**
```
try:
    write signals/pause
    poll heartbeat for status="paused" (10s timeout; abort if not reached)
    python scripts/test_harness.py \
      --out .soma-loop/reports/tick-<tick_id>.json \
      --chat-out .soma-loop/reports/chat/tick-<tick_id>.jsonl \
      --corpus data/tinyshakespeare.txt \
      --heldout data/heldout.txt \
      --fixed-prompts data/fixed_prompts.txt \
      [--checkpoint checkpoints/current.pt]
finally:
    write signals/resume  # unconditional
```

Harness output: see Appendix B.

**Phase 2 — Cold start check**
If `state/baseline.json` is absent → copy `tick-<tick_id>.json` to `baseline.json`, jump to Phase 11 (no change proposed on first tick; just capture baseline).

**Phase 3 — Approval queue**
Read `state/approval_queue.jsonl`. If any entry has `queue_status="approved"` and `base_commit_sha == git rev-parse HEAD` (stale check per G16):
- Acquire git.lock
- Update entry `queue_status="applied"`, `applied_at=now`, `applied_tick_id=<tick_id>` (prevents re-consumption next tick)
- Release git.lock
- Load entry as `pending_change.json` for this tick
- Skip Phases 4-6
- Go to Phase 7

Stale approvals marked `queue_status="stale"` and skipped (no git.lock needed — idempotent update). FIFO ordering for multiple approved entries.

**Phase 4 — Diagnose** *(Claude judgment)*
Invoke `soma-diagnose` skill. Skill reads:
- `tick-<tick_id>.json` and up to 5 prior tick reports
- Last 20 + all reverted entries from last 100 in `change_log.jsonl`
- Wiki via `wiki-knowledge` (timeout 30s total; proceed without wiki if exceeded; each individual curl call uses `--max-time 10`)
- `state/last_tick.json` for `base_commit_sha`

Writes `diagnosis.md` to tick workspace.

**Phase 5 — Propose change** *(Claude judgment)*
Capture `base_commit_sha = git rev-parse HEAD` *before* invoking skill.
Invoke `soma-propose-change` skill. Skill emits `pending_change.json` per schema (Appendix B).
- `class == null` → go to Phase 11 (no-op tick; normal outcome)
- `class == "halt"` → touch `STOP`, write `state/halt_reason.json`, append `change_log` entry `status="halted_by_diagnose"`, go to Phase 11
- Otherwise → continue to Phase 6

**Phase 6 — Architecture carveout**
Normalize each `diff[].file` to forward-slash relative path. If any match carveout list:
- `src/soma/core/**`
- `src/soma/memory/**`
- `src/soma/growth/**`
- `src/soma/system.py`
- `configs/current.yaml` with shape-key diff. **Shape keys** (changing any invalidates existing checkpoints — full list):
  - `vocab_size`, `text_embed_dim`
  - `sensor_output_dim`
  - `associator_input_dim`, `associator_hidden_dim`, `associator_output_dim`
  - `integrator_input_dim`, `integrator_hidden_dim`, `integrator_output_dim`
  - `position_dim`
  - `wm_slots`, `wm_dim`
  - `episodic_capacity`, `key_dim`, `value_dim`
  - `max_input_tokens`, `max_output_tokens`
  - `max_nodes` (affects growth trajectory and buffer pre-allocation)
- `data/*.txt`, `data/*.jsonl`

(`src/soma/metacognition/` and `src/soma/consolidation/` are deliberately NOT in the Phase 1 carveout — see §8.1 for rationale and when to expand.)

Action: append `pending_change` to `state/approval_queue.jsonl` (dedupe by `similarity_signature`), log to `reports/tick-summaries.md`, go to Phase 11.

**Phase 7 — Apply change**
```
acquire git.lock (PID-aware, 30s timeout)
verify git rev-parse HEAD == base_commit_sha (G4 staleness)
    if drifted: release lock, go to Phase 11 (stale, no change_log entry)
for each diff entry (ordered):
    op == "create" → Write file with content
    op == "delete" → Bash: rm file
    op == "edit" → Edit with before/after
    multi-edit → multiple "edit" entries same file, applied in order
append change_log entry status="in_progress", pre_change_metrics from tick-<tick_id>.json, commit_sha=null
release git.lock
```

**Phase 8 — Safety gates**
```
python scripts/safety_gate.py  # full: format, check, mypy, pytest, smoke (with pause/resume)
on failure:
    acquire git.lock  # for change_log write consistency with watchdog
    git stash && git stash drop
    update change_log entry → status="reverted_at_gate", failure_signal="<stage>"
    increment consecutive_failures.json
    release git.lock
    go to Phase 11
```

`safety_gate.py` writes `state/gate_failure.json` with `{stage, stderr_tail, duration_s}`.

**Phase 9 — Commit**
```
acquire git.lock
git add <diff files>
commit message (structured template):
    auto: <one-line summary>

    Class: <config|bugfix|feature|architecture>
    Rationale: <2-3 lines from diagnosis>
    Expected: <expected_effect JSON>
    Rollback hint: <rollback_hint>
    Pre-change metrics: loss=X curiosity=Y nodes=N edges=M

    Change-log-id: <uuid>

    Co-Authored-By: SOMA-autonomous-loop <noreply@soma.local>

on hook failure:
    git restore <diff files>
    update change_log → status="commit_hook_blocked"
    release git.lock
    go to Phase 11
update change_log entry commit_sha=<new sha>, ts_applied=now
release git.lock
```

**Phase 10 — Restart training**
```
write signals/shutdown
poll pid/train_service.pid for disappearance (30s timeout)
    timeout → taskkill /F (Windows) | kill -9 (Linux) + log
spawn new train_service.py detached:
    Windows: subprocess.Popen(..., creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
                              stdout=open(".soma-loop/logs/train_service.log","ab"),
                              stderr=subprocess.STDOUT)
    Linux: subprocess.Popen(..., start_new_session=True,
                             stdout=open(".soma-loop/logs/train_service.log","ab"),
                             stderr=subprocess.STDOUT)
    Prevents child from dying with tick; captures output to rotating log file.
poll state/train_heartbeat.json for fresh ts (60s timeout; CUDA init allowance)
    timeout → write tick_failure.json, release any locks, exit 2
    (commit NOT reverted; train startup bug may be orthogonal to change)
```

**Phase 11 — Finalize**
```
# Note: consecutive_failures is NEVER reset by the tick itself — only by the
# watchdog when a change reaches status=confirmed (see auto_revert.py step 5).
# This avoids premature reset before the change has proven stable.

update state/last_tick.json {tick_id, commit_sha, change_log_id, ts}
append entry to reports/tick-summaries.md

try:
    invoke wiki-sync skill (30s timeout)
except:
    log to tick workspace network_warnings.log

acquire git.lock
try:
    git add reports/tick-summaries.md configs/current.yaml <other state files>
    git commit -m "chore(auto): finalize tick <tick_id>"
except:
    # Phase 9's code commit is already durable; state-file commit is best-effort.
    log "state file commit skipped: <reason>"; continue
release git.lock

# Push with one pull-rebase fallback for simple divergence:
try:
    git push origin main (30s timeout)
except:
    try:
        git pull --rebase origin main
        git push origin main
    except:
        write state/push_failure.json (includes pull-rebase stderr if applicable)
        log and continue (operator resolves if persistent; next tick retries)

release tick.lock
exit 0
```

### Tick prompt — `docs/loop_tick_prompt.md` skeleton

Full text lives at `docs/loop_tick_prompt.md`. Invoked by wrapper:

```bash
# run_tick.sh
claude --print --dangerously-skip-permissions \
  --allowedTools "Bash(git:*) Bash(python:*) Bash(ruff:*) Bash(mypy:*) Bash(pytest:*) Bash(taskkill:*) Bash(kill:*) Read Edit Write Grep Glob Skill" \
  --disallowedTools "Agent WebFetch WebSearch NotebookEdit" \
  --prompt "$(cat docs/loop_tick_prompt.md)" \
  > .soma-loop/logs/tick-$(date +%s).log 2>&1 || true

# Derive exit code from state
if [ -f .soma-loop/state/tick_failure.json ]; then exit 2
elif [ -f .soma-loop/STOP ] || [ -f .soma-loop/state/baseline_broken.json ]; then exit 1
else exit 0
fi
```

Task Scheduler (Windows), systemd timer (preferred Linux), or cron (Linux fallback) all call this wrapper with env:
```
PYTHONIOENCODING=utf-8
PYTHONUTF8=1
```

### State transitions for a change

```
tick start
  ↓
[tree at commit X]
  ↓ Phase 7
[tree modified, change_log status=in_progress, commit_sha=null]
  ↓ Phase 9
[commit Y, change_log commit_sha=Y]
  ↓ Phase 10
[train_service running on Y's config/code]
  ↓ Phase 11
[tick finalized, exit]
  ↓ watchdog ticks over next 15 min
          ┌─ all confirmation criteria hold  →  status=confirmed, last_good.pt updated
          └─ 3 consecutive bad minute-buckets →  status=reverted, revert(auto): commit Z,
                                                 last_good.pt restored to current.pt,
                                                 signals/shutdown to training
```

---

## 5. Components

### 5.1 Custom skills

#### `soma-diagnose`
- **Purpose:** Analyze current tick report + history + wiki → structured diagnosis
- **Invoked by:** Phase 4
- **Inputs read:** `tick-<tick_id>.json`, up to 5 prior tick reports, `change_log.jsonl` (last 20 + reverted-from-last-100), `state/last_tick.json`, wiki (`hot.md`, SOMA entity, ≤5 relevant atoms via topic-keyword mapping)
- **Output:** `state/ticks/tick-<tick_id>/diagnosis.md` (structured markdown with cited metrics, ranked concerns, candidate interventions excluding reverted-hypothesis collisions)
- **Hard rules:** must quote ≥1 specific metric before any claim; must check `change_log` for overlapping reverted entries; wiki curl uses `--max-time 10`
- **Line budget:** 80-150 lines of SKILL.md

#### `soma-propose-change`
- **Purpose:** Given diagnosis, emit concrete change-spec JSON
- **Invoked by:** Phase 5
- **Inputs read:** `diagnosis.md`, `change_log.jsonl` reverted subset, relevant source files, `git rev-parse HEAD` via Bash
- **Output:** `state/ticks/tick-<tick_id>/pending_change.json` per Appendix B schema
- **Hard rules:** compute `similarity_signature` and emit `class=null` if matches any of last 5 reverted (no `force` override in autonomous path — G40); `class="halt"` reserved for catastrophic state (NaN, graph collapse) and has no diff
- **Line budget:** 100-150 lines

#### `soma-bootstrap-kb`
- **Purpose:** One-time seed of wiki with SOMA entity + architecture atoms
- **Invoked by:** Operator, manually, once per SOMA install
- **Mechanism:** Direct Gitea API via `curl` with `GITEA_TOKEN` from `.claude/settings.local.json`. Wiki-sync used ONLY for session summaries per its design; atom/entity creation uses Gitea contents API
- **Idempotent:** reads existing wiki entry first; SHA-compares against generated content; skips if identical
- **Output:** `wiki/entities/soma.md`, `wiki/atoms/soma-*.md` (per key concept), `wiki/molecules/soma-autonomous-loop.md`
- **Line budget:** 120-180 lines

### 5.2 Python scripts

#### `scripts/train_service.py`
Long-lived daemon.
```
python scripts/train_service.py
  [--config configs/current.yaml]
  [--checkpoint-dir checkpoints/]
  [--device auto]
```
Responsibilities:
- PID collision check on startup (exit with error if live PID owns pidfile)
- Checkpoint load chain: `current.pt` → `last_good.pt` → fresh SOMA (writes `fresh_init.flag`)
- Heartbeat `{ts, step, status, device}` → `state/train_heartbeat.json` every 10s (atomic write)
- Status lifecycle: `warming_up` → `running` → (on signal) `paused` / `shutdown`. Warm-up fires only when started from fresh init (no checkpoint loaded — `fresh_init.flag` present): first 100 steps held in `warming_up`. Checkpoint-loaded starts go straight to `running` because the graph already has learned state.
- Signal poll every 1s for `.soma-loop/signals/`:
  - `shutdown` → atomic save, clear PID, `sys.exit(0)`
  - `reload_config` → re-read current.yaml (Phase 2 feature; Phase 1 no-op)
  - `pause` → `paused=True`, update heartbeat, continue polling
  - `resume` → clear `paused`
- Metrics append every 10 steps to `metrics/metrics.current.jsonl` (one JSON per line)
- Atomic checkpoint every `config.checkpoint_interval` steps: `.tmp` write → `os.replace` → update `current.pt` + `current.txt`
- Retention: keep last 20 step checkpoints + every-100th-step checkpoint permanently
- Crash backoff: 3 uncaught exceptions within 5 min → write `train_permanent_failure.json` + exit 2; otherwise 10s backoff retry

#### `scripts/test_harness.py`
B-tier evaluation.
```
python scripts/test_harness.py
  --out .soma-loop/reports/tick-<tick_id>.json
  --chat-out .soma-loop/reports/chat/tick-<tick_id>.jsonl
  [--checkpoint checkpoints/current.pt]
  [--corpus data/tinyshakespeare.txt]
  [--heldout data/heldout.txt]
  [--fixed-prompts data/fixed_prompts.txt]
  [--max-gen-tokens 32]
  [--device auto]
```
Device: `cuda` if available AND heartbeat shows `paused`; CPU fallback with warning.

**Tokenizer consistency:** the harness rebuilds the BPE tokenizer from the corpus. This relies on BPE training being deterministic for fixed `(corpus, vocab_size)`. The `tokenizers` library used in `train_bpe_tokenizer` is deterministic in its standard settings, but this should be verified during Phase 1 pre-launch by training a tokenizer twice and hash-comparing. If divergence is observed, the tokenizer should be serialized alongside the SOMA checkpoint and loaded by the harness rather than re-trained.

Output JSON schema: see Appendix B.

#### `scripts/safety_gate.py`
```
python scripts/safety_gate.py
  [--skip-smoke]
  [--stage format|check|mypy|test|smoke|all]
```
Sequence (stops at first failure):
1. `ruff format src/ tests/`
2. `ruff check src/ tests/`
3. `mypy src/soma/`
4. `pytest -q`
5. Unless `--skip-smoke`: write `signals/pause`, poll heartbeat for `paused` (10s timeout, abort if not reached), try/finally-wrap `python scripts/smoke_train.py --steps 30`, `signals/resume`.

Writes `state/gate_failure.json` on failure. Exit 0/1/2.

#### `scripts/smoke_train.py`
```
python scripts/smoke_train.py [--steps 30] [--device auto] [--timeout 90]
```
Validates that the training loop, gradient flow, and graph ops still work after any code/config edit. Purpose is "does it crash?", not "does current.yaml work end-to-end."

**Config strategy:** uses a hardcoded minimal smoke config (vocab_size=64, text_embed_dim=16, small graph dims) that's independent of `configs/current.yaml`. This avoids vocab-mismatch issues when current.yaml specifies a large vocab but the synthetic corpus has only a handful of strings. The smoke config still exercises the same code path — Graph, Node, Edge, memory systems, growth — because those don't depend on dim sizes.

Synthetic corpus: 8 short strings (`["hello world", "foo bar baz", ...]`). Ephemeral artifacts at `tempfile.gettempdir() / "soma_smoke_<pid>.pt"`, cleaned up in `finally`. Runs N steps; fails on any exception or non-finite loss.

#### `scripts/auto_revert.py`
Watchdog — no args. Run every 2 min via scheduler.
Logic:
1. Read `state/last_tick.json`. Missing or `ts > 1h` → exit 0 (staleness guard)
2. Find last `change_log` entry with `status="in_progress"` AND `commit_sha != null`. None → exit 0. (Entries with `commit_sha=null` are pre-commit states from Phase 7 that never reached Phase 9 — nothing to revert.)
3. Read last 15 min of `metrics/metrics.current.jsonl`. Step counter advanced? No → exit 0 (training paused)
4. Evaluate confirmation criteria (all must hold):
   - Loss rolling mean within 20% of pre-change
   - Curiosity within 20% of pre-change
   - Zero NaN/Inf losses in window
   - Graph nodes didn't drop >50%
   - Step counter advanced >500 in window
   Bucket samples into 1-minute windows.
5. All hold → acquire `git.lock`, update entry `status="confirmed"`, copy current.pt → last_good.pt, **reset `consecutive_failures.json` count to 0** (only successful-confirmation path resets the counter), release lock.
6. Any criterion fails for 3 consecutive minute-buckets → acquire `git.lock`, `git revert --no-edit <commit_sha>`, commit `revert(auto): <reason>`, update entry `status="reverted"` + `failure_signal=<criterion:pass_map>`, restore last_good.pt → current.pt, increment `consecutive_failures.json`, `signals/shutdown` to training, release lock.
7. **Revert-failure path (G70):** if `git revert` fails (pre-commit hook or conflict) → write `state/revert_failure.json`, touch `STOP`, exit 1 (logged alert; operator resolves).

#### `scripts/approve.py` / `scripts/reject.py`
```
python scripts/approve.py --list
python scripts/approve.py <queue-id>
python scripts/reject.py <queue-id> [--reason "..."]
```
`approve.py <id>`:
- Refuses to re-process entries already `approved`, `applied`, `stale`, or `rejected` (prints current status and exits 1)
- Only acts on entries with `queue_status="pending"`
- If entry `base_commit_sha == git rev-parse HEAD`: `queue_status=approved`, `approved_at=now`, `approved_by=os.getenv('USER') or os.getenv('USERNAME') or 'unknown'`
- Else: `queue_status=stale` with reason — user can inspect and requeue via new diagnose+propose cycle if still desired

`--list`: plain-text table (ID(6) CLASS QUEUE_STATUS AGE BASE_SHA(8) DESCRIPTION(truncated))

#### `scripts/bootstrap_loop.py`
```
python scripts/bootstrap_loop.py [--force] [--corpus PATH]
```
Creates (skipped if exists unless `--force`, except `fixed_prompts.txt` and `current.yaml` which are NEVER overwritten):
- `configs/current.yaml` ← copy of `default.yaml`
- `data/heldout.txt` ← deterministic 10% split (seed 42) of corpus
- `data/fixed_prompts.txt` ← 20 hard-coded curated prompts (see Appendix C)
- `data/corpus_token_freq.json` ← empirical token freqs for KL reference
- `.soma-loop/state/change_log.jsonl` (empty)
- `.soma-loop/state/approval_queue.jsonl` (empty)
- `.soma-loop/state/consecutive_failures.json` (`{"count": 0, "last_reset_ts": null, "last_failure_ts": null}`)
- `.soma-loop/signals/`, `.soma-loop/reports/chat/`, `.soma-loop/logs/`, `.soma-loop/state/ticks/`, `.soma-loop/metrics/`, `.soma-loop/pid/` directories

Prints next-step instructions.

#### `scripts/audit_loop.py`
```
python scripts/audit_loop.py
```
Validates (exits 1 if any fails):
- Every `^auto:` commit has a matching `change_log.jsonl` entry OR `Change-log-id:` trailer
- No `^auto:` commit touches carveout paths (grep as in G73)
- `train_heartbeat.json.device` shows cuda consistently across the window
- Disk footprint within G35 bounds
- `consecutive_failures.count == 0`
- Gate ever passed on pristine tree in the window
- Wiki has received ≥5 summary pushes — verified by querying Gitea API: `GET /api/v1/repos/dant123/knowledge-wiki/commits?since=<ts>&path=conversations` and counting entries with `SOMA-session-*` filename pattern

Used in Phase 1 validation and Phase transition runbooks.

#### `scripts/lock.py`
Library (not CLI). `FileLock` class with cross-platform PID-aware locking via `portalocker` (new dep).
```python
class FileLock:
    def __init__(self, path: Path, timeout: float = 30.0, purpose: str = "") -> None: ...
    def acquire(self) -> bool: ...           # True = acquired; False after timeout or held by live PID
    def release(self) -> None: ...
    def __enter__ / __exit__
def is_pid_alive(pid: int) -> bool: ...      # cross-platform (os.kill 0 / OpenProcess)
```
Lock file JSON: `{"pid": <int>, "ts": <epoch>, "purpose": "<str>"}`. Stale = PID dead OR ts > 30 min. Used by `git.lock` and `tick.lock`.

#### `scripts/run_tick.sh` / `scripts/run_tick.ps1`
Shell wrappers that:
1. Set `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`
2. Invoke `claude --print` with the tick prompt, allow/disallow tool whitelists
3. Capture stdout/stderr to `.soma-loop/logs/tick-<timestamp>.log`
4. Derive wrapper exit code from state files (tick_failure.json → 2; STOP/baseline_broken → 1; else 0)
5. Detect Claude CLI failure patterns → write `state/claude_cli_error.json` (G71)

### 5.3 UI patch — spectator mode

When `.soma-loop/state/autonomous_mode.flag` exists, the existing DearPyGUI UI enters read-only:
- Controls panel: Start/Pause/Stop/Step/Reset/Apply+Rebuild buttons disabled
- Config panel: all inputs disabled
- Graph/metrics panels: read from `.soma-loop/metrics/metrics.current.jsonl` and latest snapshot file (not from a shared in-process SOMA)
- Chat panel: loads latest checkpoint in a CPU-only copy for inference (no GPU contention with train_service)

Implemented as additions to existing panel code; no structural changes.

### 5.4 Dependency additions

`pyproject.toml` adds `portalocker>=2.8` (cross-platform file locking).

### 5.5 Existing skills used unchanged
- `wiki-knowledge` — for Phase 4 diagnose reads
- `wiki-sync` — for Phase 11 session summary pushes (throttled to change-tick only per G78)

---

## 6. Phase 1 Validation

### 6.1 Pre-launch checklist (G68)

Before enabling the scheduler:

```
- [ ] pytest -q passes 10 consecutive times (flake check)
- [ ] python scripts/smoke_train.py succeeds manually
- [ ] python scripts/safety_gate.py --skip-smoke passes on pristine tree
- [ ] python scripts/bootstrap_loop.py completed without errors
- [ ] data/fixed_prompts.txt reviewed by operator (and edited if desired)
- [ ] soma-bootstrap-kb invoked once: `claude --print --prompt "Use soma-bootstrap-kb skill for SOMA at E:/Documents/Projects/SOMA"`
- [ ] Wiki verified: SOMA entity page present, ≥10 atoms present
- [ ] configs/current.yaml exists, diffs cleanly from default.yaml (or matches)
- [ ] .soma-loop/STOP absent, state/baseline_broken.json absent
- [ ] train_service.py started manually; heartbeat file updating; ≥1 min of metrics recorded
- [ ] Single manual tick run: `bash scripts/run_tick.sh` (or .ps1) completed successfully with exit 0
- [ ] python scripts/auto_revert.py runs manually; exits 0 (no in-flight change)
- [ ] Task Scheduler entries created but DISABLED
- [ ] (If UI is installed) `touch .soma-loop/state/autonomous_mode.flag` so UI enters read-only spectator mode when the loop goes live
```

When all boxes ticked, enable the scheduled tasks. 48-hour clock starts at first scheduled tick.

### 6.2 Running criteria (48 continuous hours)

**Stability:**
- <5% ticks exit with code 2 (unhandled failure; handled aborts don't count)
- Training service uptime ≥98% (auto-recovered crashes count against)
- Zero loop-caused data-integrity incidents (checkpoint load fallback never reached `fresh_init` during the window; no corrupted state files; no git corruption)
- Zero operator interventions for unplanned issues (STOP/baseline_broken resolutions, approval-queue processing, and planned validation injections are OK — everything else disqualifies)

**Safety:**
```
# carveout verification — must return empty (all carveout paths included)
git log --since="48 hours ago" --grep="^auto:" --name-only \
  | grep -E "^(src/soma/(core|memory|growth)/|src/soma/system\\.py$|data/.*\\.(txt|jsonl)$|scripts/|tests/|pyproject\\.toml$|CLAUDE\\.md$|docs/plans/)"
```
(Note: shape-key changes in `configs/current.yaml` are harder to grep for — audit_loop.py parses the YAML diff per commit and flags any shape-key change on an `^auto:` commit as a carveout violation.)
- No `--no-verify` or equivalent bypass in git history
- Circuit breaker (`consecutive_failures.count` hitting 3) **never fires**; any firing disqualifies and requires investigation
- No git race artifacts (stale git.lock held by dead PIDs must have been detected and reclaimed correctly; never a commit mid-revert; never a partial multi-file apply)

**Effectiveness:**
- ≥1 autonomous change reached `status="confirmed"` (end-to-end pipeline validated)
- ≥1 autonomous change reached `status="reverted"` via watchdog (regression path validated)
- ≥3 distinct diagnosis-description prefixes in `change_log.jsonl` (variety of reasoning)
- ≥5 non-trivial wiki-sync pushes *during the loop* (bootstrap pushes don't count toward this)
- train_service heartbeat shows `device=cuda:*` consistently (no silent CPU fallback)

**Auditability:**
- `reports/tick-summaries.md` reads as a coherent human-readable timeline
- `python scripts/audit_loop.py` exits 0
- Disk footprint within G35 bounds (~5-10 GB over 48h for checkpoints + metrics + logs + reports)

### 6.3 Deliberate validation injections (G72)

Natural events might not cover the `confirmed`/`reverted` paths in 48h. Before declaring Phase 1 complete, operator deliberately:

1. **Revert-path test:** Manually edit `configs/current.yaml` with a known-bad value (e.g., `base_lr: 0.5`), route through `approve.py` to apply it. Watchdog should revert within 15 min. Verify `change_log` + `revert(auto):` commit. Restore normal value.

2. **Gate-failure test:** Queue a synthetic change that fails a gate (e.g., intentional syntax error in a non-carveout file). Verify safety_gate catches it and change_log records `status="reverted_at_gate"` with `failure_signal="check"`.

Both are optional-but-recommended. Document results in `reports/tick-summaries.md`.

### 6.4 Exit condition

All 6.2 criteria AND 6.3 injections completed AND `scripts/audit_loop.py` clean → operator eligible to proceed to Phase 2. Any failure: extend window, investigate, re-run.

---

## 7. Future Work (documented, not built in Phase 1)

### 7.1 Approach 2 — Corpus scale-up
**Trigger:** Phase 1 validated.
**Changes:**
- Download TinyStories + DailyDialog, blend 80/20 → `data/blended_corpus.txt` (gitignored)
- Vocab bump 512 → 8192 via arch-carveout approval
- Checkpoint reset (Phase 1 archived to `checkpoints.phase1/`)
- Re-run `bootstrap_loop.py --force --corpus data/blended_corpus.txt`
- Re-run `soma-bootstrap-kb` (new corpus context)
- Loop unchanged; observation window 14 days
**Transition runbook:** `docs/AUTONOMOUS_LOOP.md#phase-1-to-phase-2-transition`
**New script:** `scripts/download_corpus.py`
**Risks tracked:** first ~10K steps erratic on new corpus (tune watchdog window if needed); graph growth may need `max_nodes` bump
**Rollback plan:** archived Phase 1 state in `checkpoints.phase1/` + `change_log.phase1.jsonl`; swap back configs/current.yaml

### 7.2 Approach 3 — Developmental curriculum
**Trigger:** Phase 2 validated AND (chat plateau OR operator-initiated).
**New component:**
- `scripts/curriculum_controller.py` manages stage transitions
- `configs/curriculum.yaml` defines stages (schema TBD in Phase 3)
- `soma-curriculum-manager` skill invoked alongside diagnose each tick
- Stages (sketch): Stage 0 synthetic text patterns (XOR as "0 0 -> 0" strings) → Stage 1 TinyStories → Stage 2 TinyStories+DailyDialog → Stage 3 broader blend
- Stage promotion is arch-carveout (always approval-gated)
- Stage demotion always manual — never autonomous (G79)
- Per-stage `change_log` scoping (similarity check restricted to same stage; cross-stage hypotheses can re-apply)

### 7.3 Test harness C — Comprehensive
**Trigger:** cloud migration (more compute per tick).
**Additions to B:**
- **Qualitative rubric scoring:** Agent subagent batches all 20 prompts in one call, scores {lexical, syntactic, semantic, responsiveness} 1-5 each (G81 batching). Requires re-enabling Agent tool in allowedTools when harness C active (G80).
- **Ablation probes:** 5-10 ephemeral parallel training runs (new config knob variants, 500 steps each); requires ≥4 GPUs or serial fallback
- **Attention-range proxy:** inject single-sensor pulse, measure propagation depth through graph in one timestep; requires new `Graph.measure_propagation_depth()` method
- **Deeper graph stats:** depth_avg, fanout_p95, edge_age_distribution, maturity_distribution
- Harness gets `--tier C` flag; tick flow unchanged

### 7.4 Orchestration C — Linux cloud substrate
**Trigger:** local 3090 becomes compute bottleneck.
**Changes (substrate only; not the loop logic):**

| Windows (Phase 1) | Linux cloud (Orchestration C) |
|---|---|
| Windows Task Scheduler | `systemd timer` (preferred) or `*/30` cron (fallback) |
| `run_tick.ps1` | `run_tick.sh` (already built cross-platform) |
| `%USERPROFILE%\.claude` | `~/.claude` (re-auth once on VM with Max credentials) |
| `taskkill /F` | `kill -9` / `SIGKILL` |
| `CREATE_NEW_PROCESS_GROUP` | `setsid` / `start_new_session=True` |
| Local disk | Persistent VM disk + periodic snapshot backups |

**VM target options:** A10G 24GB (~$1/hr, matches 3090), L40S 48GB (~$1.5/hr, headroom), A100-40 (~$2/hr), A100-80/H100-80 (~$3-4/hr, Phase 3+).
**Migration procedure:** rsync `.soma-loop/state/` + `checkpoints/` to VM; git clone + install; interactive Claude Code auth once; crontab mirrors Task Scheduler entries.
**Auth survival:** Max tokens cache in `~/.claude/`; periodic manual re-auth ritual (weekly check).
**Backup strategy:** VM provider snapshots + periodic `checkpoints/current.pt` + `last_good.pt` off-disk (S3 / Backblaze / local rsync).

### 7.5 Orchestration D — Python + Anthropic API (only if API access obtained)
**Trigger:** User obtains Anthropic API access.
**Changes:**
- `scripts/loop.py` Python daemon replaces `claude --print` wrapper
- Skills migrate to prompt templates (`prompts/soma_diagnose.md`, `prompts/soma_propose_change.md`)
- Direct Anthropic SDK calls with structured-outputs schema enforcement
- `scripts/wiki_client.py` helper for Gitea API
- Docker-containerizable entry point
- Secrets: `ANTHROPIC_API_KEY`, `GITEA_TOKEN` via env or mounted secret files
**Not built unless API access obtained.**

---

## 8. Risks, Limitations & Open Questions

### 8.1 Known limitations

- **`src/soma/metacognition/` and `src/soma/consolidation/` are NOT in the initial carveout.** The Phase 1 carveout covers brain modules (`core`, `memory`, `growth`, `system.py`), loop infrastructure (`scripts/`, `tests/`, `pyproject.toml`, `CLAUDE.md`, `docs/plans/`), shape keys in `configs/current.yaml`, and `data/*.txt|*.jsonl`. Metacognition (curiosity, homeostasis) and consolidation modules are foundational but autonomously editable in Phase 1. Expand carveout if observed to be too-easily-destabilized.

- **Skill judgment isn't reproducible.** Given the same inputs, Claude may propose different changes. `change_log` records what happened, not the fully-conditioned reasoning. Acceptable for audit purposes; not suitable for strict scientific replication.

- **Git push conflicts with operator local edits.** If operator commits to `main` while loop runs, push fails; loop retries with `git pull --rebase` at next tick. If rebase conflicts, tick aborts with STOP. Primary guard: runbook rule "don't edit tracked files while loop is running" (G22).

- **Claude Code auto-updates.** Claude Code may update mid-run and break a CLI behavior. Consecutive-failures circuit catches (G24); operator inspects and decides whether to rollback Claude Code or adjust loop code.

- **Wiki-sync noise in Phase 2.** ~200 session summaries over 14 days (change-tick-only throttling per G78). Daily digest summarization is possible future improvement.

- **Harness B is loss-dominated.** Chat quality is judged qualitatively by operator reading `reports/chat/tick-<tick_id>.jsonl`. Automated quality signal waits for harness C.

### 8.2 Open questions (revisit after Phase 1)

- Is 37-min tick cadence right? Shorter could burn Max message quota; longer loses responsiveness.
- Does the similarity-signature algorithm (G45) cover enough equivalent-diff cases, or does Claude find workarounds?
- When Claude proposes `class=halt`, does it ever false-positive on noisy metrics?
- Is the 15-min confirmation window long enough to catch slow-onset regressions?

### 8.3 Observability gaps

- No live dashboard (UI spectator mode is the closest). Optional future work: `scripts/dashboard.py` that prints loop health summary.
- No email/push alerting. Task Scheduler can email on non-zero exit (Windows); cron + `mail` similar on Linux. Document options in runbook.

---

## Appendix A — Design decisions by ID (G1–G90)

Consolidated list for cross-reference from implementation plan.

**Section 2 critical (G1-G8):** rate-limit counting by subject prefix (G1); change_log authoritative in commit trailers (G2); checkpoint-shape config keys routed to carveout (G3); HEAD-sha staleness check before apply (G4); git.lock broadened for writers only (G5); Phase 11 uses `chore(auto):` not counted to rate limit (G6); tick.lock prevents overlapping ticks (G7); minimal allowedTools whitelist (G8).

**Section 2 moderate (G9-G18):** heartbeat pause verification (G9); harness/gate try/finally resume (G10); tick-local workspace (G11); similarity revert-loop block (G12); push non-fatal (G13); wiki-sync non-fatal (G14); detached spawn (G15); stale approval by base_commit_sha (G16); approval dedupe by diff hash (G17); force-kill last resort (G18).

**Section 2 minor (G19-G23):** path normalization (G19); `system.py` in carveout (G20); 60s heartbeat timeout (G21); no operator edits during run (G22); cloud state migration copies (G23).

**Section 2 new-round (G24-G36):** consecutive-failure circuit breaker (G24); corpus files in carveout (G25); checkpoint load fallback chain (G26); `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` (G27); pytest 10x stability check (G28); watchdog staleness safeguard (G29); watchdog pause handling (G30); `scripts/reject.py` (G31); Phase 1 restart-all, reload-config deferred (G32); confirmation criteria formalized (G33); Claude Max message-count consideration (G34); disk space plan (G35); train_service warm-up state (G36).

**Section 3 critical (G37-G44):** bootstrap-kb uses Gitea API not wiki-sync (G37); change_log.jsonl schema formalized (G38); diff format for create/delete/multi-edit (G39); no force override (G40); class=halt handled in Phase 5 (G41); tick wrapper derives exit code from state files (G42); `--prompt "$(cat …)"` not `--append-system-prompt-file` (G43); git.lock narrowed to writers (G44).

**Section 3 moderate (G45-G59):** similarity_signature algorithm (G45); skill reads base_commit_sha via Bash (G46); bootstrap-kb idempotency via SHA (G47); reloadable config keys deferred (G48); train_service PID collision check (G49); crash backoff (G50); tokenizer determinism documented (G51); KL reference cached at bootstrap (G52); harness device auto-select (G53); ruff-format diffs part of commit (G54); cross-platform tmp (G55); minute-bucketed regression sampling (G56); Windows username fallback (G57); portalocker dep (G58); consecutive_failures.json bootstrapped (G59).

**Section 3 minor (G60-G65):** fixed_prompts.txt content (G60); approve --list format (G61); max-gen-tokens default 32 (G62); minimal graph stats for v1 (G63); bootstrap --force preservation (G64); diff optional for class=null|halt (G65).

**Section 4 critical (G66-G72):** tick success metric via exit code 2 (G66); loop-sourced wiki growth only (G67); pre-launch checklist (G68); Phase 1→2 transition runbook (G69); revert failure handling (G70); Max subscription failure detection (G71); deliberate validation injections (G72).

**Section 4 moderate (G73-G86):** carveout grep command (G73); circuit breaker zero-firing target (G74); GPU usage validation (G75); disk footprint validation (G76); bootstrap-kb re-run on corpus change (G77); wiki-sync change-only throttling (G78); curriculum demotion manual-only (G79); harness C re-enables Agent (G80); rubric batching (G81); cloud VM specs (G82); cloud auth survival (G83); git push conflict handling (G84); checkpoint backups (G85); version pinning documented (G86).

**Section 4 minor (G87-G90):** artifact count corrected (G87); synthetic curriculum via text (G88); Gitea wiki git-versioned (G89); `scripts/audit_loop.py` added to build-now (G90).

---

## Appendix B — Schemas

### B.1 `change_log.jsonl` entry

```json
{
  "change_log_id": "<uuid4>",
  "tick_id": <int>,
  "ts_proposed": "<iso8601>",
  "ts_applied": "<iso8601>|null",
  "ts_resolved": "<iso8601>|null",
  "commit_sha": "<sha>|null",
  "base_commit_sha": "<sha>",
  "class": "config|bugfix|feature|architecture|halt",
  "description": "<imperative one-sentence summary>",
  "rationale": "<2-3 sentence justification referencing diagnosis>",
  "diff_files": ["<path>", ...],
  "similarity_signature": "<sha256>",
  "pre_change_metrics": {
    "heldout_loss_mean": <float>,
    "curiosity_ema_500": <float>,
    "num_nodes": <int>,
    "num_edges": <int>
  },
  "post_change_metrics": {...} | null,
  "status": "pending | in_progress | confirmed | reverted | reverted_at_gate | commit_hook_blocked | stale | halted_by_diagnose",
  "failure_signal": "<str>|null",
  "confirmed_at": "<iso>|null",
  "reverted_at": "<iso>|null",
  "reverted_by": "watchdog | tick_gate | null"
}
```

Appended only while holding `git.lock`.

### B.2 `pending_change.json` (Phase 5 output)

```json
{
  "change_log_id": "<uuid4>",
  "base_commit_sha": "<sha>",
  "class": "config | bugfix | feature | architecture | halt | null",
  "description": "<imperative one-sentence>",
  "rationale": "<2-3 sentences>",
  "expected_effect": {
    "loss_delta_pct": <float>,
    "graph_growth": "unchanged | expanded | pruned",
    "confirm_window_min": <int>
  },
  "diff": [
    {
      "file": "<relative path>",
      "op": "create | delete | edit",
      "before": "<str>|null",
      "after": "<str>|null",
      "content": "<str>"  // for op=create only
    }
  ],
  "rollback_hint": "<what to watch>",
  "similarity_signature": "<sha256>",
  "blocked_by_similarity": "<change_log_id>|null"
}
```

`diff` optional when `class in (null, halt)`.

### B.3 `approval_queue.jsonl` entry

Contains all fields from `pending_change.json` (B.2) plus queue-management metadata below. Note that `queue_status` is a **separate** state space from `change_log` status — do not conflate.

```json
{
  // All pending_change.json fields ...
  "queue_id": "<uuid4>",
  "queued_at": "<iso>",
  "queue_status": "pending | approved | stale | rejected | applied",
  "approved_at": "<iso>|null",
  "approved_by": "<username>|null",
  "rejected_at": "<iso>|null",
  "rejected_reason": "<str>|null",
  "applied_at": "<iso>|null",
  "applied_tick_id": <int>|null,
  "diff_hash": "<sha256 for dedupe>"
}
```

Lifecycle: `pending` (written by Phase 6) → `approved` (by approve.py) or `rejected` (by reject.py) or `stale` (by approve.py or Phase 3 if HEAD drifted) → `applied` (by Phase 3 after tick consumes the approved entry).

### B.4 Test harness report (`tick-<tick_id>.json`)

```json
{
  "tick_id": <int>,
  "ts": "<iso>",
  "base_commit_sha": "<sha>",
  "global_step": <int>,
  "heldout_loss_mean": <float>,
  "heldout_perplexity": <float>,
  "output_distribution_kl": <float>,
  "graph": {
    "nodes": <int>,
    "edges": <int>,
    "avg_degree": <float>
  },
  "memory": {
    "wm_occupancy": <float>,
    "episodic_entries": <int>,
    "episodic_fill_pct": <float>
  },
  "training": {
    "last_loss": <float>,
    "loss_ema_500": <float>,
    "curiosity_ema_500": <float>,
    "lr_multiplier": <float>
  },
  "growth_last_1k_steps": {
    "syn": <int>,
    "neuro": <int>,
    "prune": <int>
  },
  "health_flags": ["<str>", ...]
}
```

### B.5 `train_heartbeat.json`

```json
{
  "ts": "<iso>",
  "step": <int>,
  "status": "warming_up | running | paused | shutdown",
  "device": "cuda:0 | cpu | ...",
  "pid": <int>
}
```

Atomic-written every 10 seconds.

### B.6 `consecutive_failures.json`

```json
{
  "count": <int>,
  "last_reset_ts": "<iso>|null",
  "last_failure_ts": "<iso>|null"
}
```

Incremented by: tick Phase 8 gate failure, watchdog revert (auto_revert.py step 6), and uncaught tick failures (wrapper detects exit code 2).
Reset to 0 by: watchdog on confirmation (auto_revert.py step 5) only — never by the tick itself. Rationale: a change isn't "successful" until the 15-min watchdog window confirms it; resetting earlier would mask a silent regression.

---

## Appendix C — 20 default fixed prompts

Written by `bootstrap_loop.py` to `data/fixed_prompts.txt` on first run (NEVER overwritten on `--force`).

```
To be, or not to be,
The king said,
O Romeo, Romeo,
What light through
Shall I compare thee
hello
what is your name
tell me about yourself
how do you feel
who are you
The quick brown
Once upon a
In the beginning
Long ago in
The secret of

.
a
a a a a a a a a a a a a a a a a
Now is the winter of our discontent made glorious summer by this sun of York
```

(Prompt 16 is empty, prompt 19 is a long space-separated-`a` filler to exercise long-input paths, prompt 20 is a full Shakespeare line. Operator is expected to review and edit this file to suit their test priorities before enabling the loop.)

---

## Appendix D — Task Scheduler / cron entries

### Windows Task Scheduler

**Loop tick task:**
- Trigger: at user log on (or at system startup if run-as-service)
- Repeat every **37 minutes**, duration **indefinitely** (check "Repeat every" and leave duration unchecked for indefinite)
- Optional start-delay on trigger (e.g., 7 min after log on) to achieve off-minute alignment
- Action: `powershell.exe -ExecutionPolicy Bypass -File "E:\Documents\Projects\SOMA\scripts\run_tick.ps1"`
- Run as: logged-in user (needed to access `~/.claude/` credentials)
- Environment: `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`
- Enabled only after pre-launch checklist passes

**Watchdog task:**
- Same pattern, repeat every **2 minutes** indefinitely
- Action: `cmd.exe /c "cd /d E:\Documents\Projects\SOMA && python scripts\auto_revert.py"`
- Environment: as above

### Linux — systemd timers (preferred)

Cron's 5-field syntax can't express "every 37 minutes" cleanly — repeating that interval produces uneven fire gaps (any 37-min pattern across a 60-min hour is aperiodic modulo the hour). Use `systemd` timer units instead:

```
# /etc/systemd/system/soma-loop.service
[Unit]
Description=SOMA autonomous loop tick
After=network.target

[Service]
Type=oneshot
WorkingDirectory=<REPO_ROOT>
Environment=PYTHONIOENCODING=utf-8 PYTHONUTF8=1
ExecStart=<REPO_ROOT>/scripts/run_tick.sh

# /etc/systemd/system/soma-loop.timer
[Unit]
Description=Run SOMA loop tick every 37 min

[Timer]
OnBootSec=2min
OnUnitActiveSec=37min
AccuracySec=30s

[Install]
WantedBy=timers.target
```

Same pattern (`OnUnitActiveSec=2min`) for `soma-watchdog.timer` calling `scripts/auto_revert.py`. Install both: `systemctl enable --now soma-loop.timer soma-watchdog.timer`.

### Linux — cron fallback

If systemd is unavailable, approximate with every-30-minute cron (losing the off-minute property):

```cron
# Replace <REPO_ROOT> with the actual repo path on the VM (e.g., /opt/soma, /home/user/soma).
# Loop tick every 30 min (compromise)
*/30 * * * * cd <REPO_ROOT> && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 bash scripts/run_tick.sh
# Watchdog every 2 min
*/2 * * * * cd <REPO_ROOT> && PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python scripts/auto_revert.py
```

---

## Appendix E — Operator quick reference

**Start loop:** check pre-launch checklist → start train_service (see below) → enable schedulers
**Start train_service:** `python scripts/train_service.py` (detached background via Task Scheduler at boot, or manual `&`/`start`)
**Stop loop (keep training running):** `touch .soma-loop/STOP` → wait for current tick to complete
**Halt training entirely:** `touch .soma-loop/signals/shutdown` (service saves + exits)
**Resume loop:** `rm .soma-loop/STOP` (train_service not auto-started — operator restarts manually if stopped)
**Approve arch change:** `python scripts/approve.py --list` → `python scripts/approve.py <id>`
**Reject arch change:** `python scripts/reject.py <id> --reason "..."`
**Audit:** `python scripts/audit_loop.py`
**View last tick:** `cat .soma-loop/logs/tick-<latest>.log`
**View decisions:** `cat reports/tick-summaries.md`
**Emergency rollback:** `git log --grep="^auto:"` → pick healthy commit → `git reset --hard <sha>` (destructive; operator ack'd) → restart train_service

Full operator manual: `docs/AUTONOMOUS_LOOP.md`

---

*End of design document.*
