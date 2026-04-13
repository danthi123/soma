# Autonomous Loop — Operator Runbook

## 1. Overview

The autonomous loop is a Claude-in-the-middle improvement cycle that runs train → test → diagnose → propose → gate → commit → restart → wiki-sync on a 37-minute cadence. A long-lived training service (`scripts/train_service.py`) drives SOMA forward continuously; every 37 minutes the Windows Task Scheduler fires `scripts/run_tick.sh`, which invokes Claude Code with the soma-diagnose and soma-propose-change skills against the latest metrics, commits any approved change under an `auto:` trailer, and signals the training service to hot-reload the new checkpoint. A separate watchdog (`scripts/auto_revert.py`) runs every 2 minutes and reverts regressions. The complete design is in `docs/plans/2026-04-12-autonomous-loop-design.md`; this document is the operational manual.

## 2. Prerequisites

- **Python 3.11+** with the repo installed editable (`pip install -e ".[dev]"`).
- **NVIDIA GPU with CUDA** — the loop expects `device=cuda:*` at every heartbeat. Silent CPU fallback disqualifies a validation window.
- **Claude Code authenticated** on the logged-in Windows user. Credentials live under `%USERPROFILE%\.claude\`; the scheduled tasks run as that user so they can read the tokens.
- **Gitea token** stored in `.claude/settings.local.json` under `env.GITEA_TOKEN`. The bootstrap-kb and wiki-sync phases use it to push atom pages.
- **Skills installed** in `~/.claude/skills-repo/skills/domain-specific/`: `soma-diagnose`, `soma-propose-change`, `soma-bootstrap-kb`. Verify with `claude skill list`.
- **Portalocker** (for cross-platform file locking) — pulled in by `pip install -e .`.

## 3. First-time setup

Follow the pre-launch checklist in design §6.1 (`docs/plans/2026-04-12-autonomous-loop-design.md#61-pre-launch-checklist-g68`). Every item must be ticked before enabling the scheduled tasks. Key commands in order:

```bash
python -m pytest tests/ -q                    # run 10 times; zero flakes
python scripts/smoke_train.py --device cuda   # ~30 s
python scripts/safety_gate.py --skip-smoke    # format/check/mypy/pytest
python scripts/bootstrap_loop.py              # seeds .soma-loop/, configs/current.yaml, fixed_prompts
claude --print --prompt "Use soma-bootstrap-kb skill for SOMA at $(pwd)"
python scripts/train_service.py               # manual smoke; Ctrl+C after ~1 min
bash scripts/run_tick.sh                      # one manual tick; exit 0 expected
python scripts/auto_revert.py                 # no in-flight change; exit 0
touch .soma-loop/state/autonomous_mode.flag   # optional — UI enters spectator mode
```

Then create Task Scheduler entries per §4 below, **kept DISABLED** until you are ready to start.

## 4. Starting the loop

The loop is two scheduled tasks plus one long-lived process.

**4.1 Start the training service** (long-lived; survives ticks):

```powershell
# PowerShell — detached window so closing the terminal doesn't kill it
Start-Process python -ArgumentList "scripts\train_service.py" -WorkingDirectory "E:\Documents\Projects\SOMA"
```

Or register it as a third scheduled task triggered "at user log on" (no repeat) if you want it to survive reboots.

Watch `.soma-loop/state/heartbeat.json` tick over — it refreshes every 10 s. Confirm `device` contains `cuda`.

**4.2 Enable Task Scheduler entries.** Per design Appendix D:

| Task name              | Trigger                                  | Action                                                                  |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------------------------- |
| `SOMA Loop Tick`       | At log on, repeat every **37 min**       | `powershell.exe -ExecutionPolicy Bypass -File scripts\run_tick.ps1`     |
| `SOMA Loop Watchdog`   | At log on, repeat every **2 min**        | `cmd.exe /c "cd /d <REPO> && python scripts\auto_revert.py"`            |

Task Scheduler settings for both:
- Run whether user is logged on or not → **unchecked** (needs `~/.claude/` credentials on the interactive session).
- Environment in "Additional properties → Settings": set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` via the action's working directory + a wrapper, or bake into `run_tick.ps1`.
- "If the task is already running" → **Do not start a new instance** (tick.lock already enforces this, but belt-and-braces).

Right-click → Enable once pre-launch checklist is green. The 48-hour validation clock starts at the first scheduled tick.

## 5. Monitoring

Four surfaces cover 99 % of operational questions:

- **`reports/tick-summaries.md`** — append-only human timeline. Each tick writes a short block: tick_id, diagnosis, proposed change (if any), gate result, commit sha. This is the first file to read every morning.
- **`.soma-loop/state/heartbeat.json`** — updated every 10 s by the training service: `{step, loss_ema, device, pid, status, ts}`. Stale (>30 s) means the service is dead or hung.
- **`.soma-loop/logs/tick-<id>.log`** — per-tick full Claude stdout/stderr. Inspect when a tick fails, stalls, or commits something surprising.
- **`git log --grep="^auto:"`** — every autonomous commit. Cross-reference against `change_log.jsonl` (authoritative status field — confirmed / reverted / pending).

Supplementary surfaces:
- **UI spectator mode** — if `autonomous_mode.flag` is touched, `soma-ui` boots read-only and tails `metrics.jsonl` + `graph_snapshot.json` directly from `.soma-loop/`. Safe to run anytime; never interferes.
- **`python scripts/audit_loop.py`** — one-shot validation of the last 48 h: carveout violations, bypass flags, circuit-breaker firings, race artefacts. Must exit 0 at the end of the Phase 1 window.
- **`.soma-loop/state/approval_queue.jsonl`** — architecture-carveout proposals waiting for operator approval. Empty in steady state. Run `python scripts/approve.py --list` to view.

## 6. Common operations

**Stop the loop (keep training running):**
```bash
touch .soma-loop/STOP
# wait for current tick to finish — ticks poll STOP on entry
```

**Halt training entirely (safe shutdown):**
```bash
touch .soma-loop/signals/shutdown
# train_service saves a final checkpoint, unlinks its PID file, exits
```

**Resume the loop:**
```bash
rm .soma-loop/STOP
# train_service is NOT auto-restarted; restart manually per §4.1 if it exited
```

**Pause/resume training without stopping the service:**
```bash
touch .soma-loop/signals/pause    # service drops to idle, keeps heartbeat
rm  .soma-loop/signals/pause && touch .soma-loop/signals/resume
```

**Approve an arch-carveout change:**
```bash
python scripts/approve.py --list          # review pending proposals
python scripts/approve.py <id>            # mark approved; next tick applies it
```

**Reject an arch-carveout change:**
```bash
python scripts/reject.py <id> --reason "too aggressive, base_lr >5x current"
```

**Emergency rollback** (destructive; operator must ack):
```bash
touch .soma-loop/STOP                                          # stop new ticks
touch .soma-loop/signals/shutdown                              # stop training
git log --grep="^auto:" --oneline                              # pick last-known-good sha
git reset --hard <sha>                                         # destructive
python scripts/train_service.py &                              # restart training from the healthy checkpoint
rm .soma-loop/STOP
```

Document the rollback in `reports/tick-summaries.md` with the reason.

**Re-seed the loop state** (nuclear; use only if state is corrupt):
```bash
python scripts/bootstrap_loop.py --force
```
This rebuilds `.soma-loop/state/` from scratch but preserves `checkpoints/` and `change_log.jsonl`.

## 7. Troubleshooting

| Symptom                                                                 | Likely cause                                                | Fix                                                                                                                    |
| ----------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `heartbeat.json` older than 30 s                                        | Training service crashed or hung                            | Check `.soma-loop/logs/train_service.log`; restart per §4.1. `scripts/train_service.py` auto-retries via `CrashBackoff` up to 5 times/5 min before giving up. |
| `device=cpu` in heartbeat                                               | CUDA unavailable at startup; silent fallback                | Stop service, verify `nvidia-smi`, restart. Never leave running — disqualifies the validation window (§6.2).          |
| Tick exit code 2                                                        | Unhandled failure in the tick wrapper                       | Read `.soma-loop/logs/tick-<id>.log`. Common causes: Claude Code updated and broke a CLI contract; Gitea push rebase conflict; stale `git.lock` from a killed process. |
| `.soma-loop/state/consecutive_failures.count` ≥ 3                       | Circuit breaker tripped                                     | Loop self-suspends. Investigate root cause in recent tick logs, fix, `rm .soma-loop/state/consecutive_failures.count`, `rm .soma-loop/STOP` if set. Any firing disqualifies the 48-hour window. |
| Stale `git.lock` held by dead PID                                       | Prior tick killed mid-commit                                | `scripts/lock.py` auto-reclaims after 30 min + PID liveness check. If urgent: confirm PID is dead (`tasklist /FI "PID eq <pid>"`), delete the lock manually. |
| `change_log.jsonl` shows `reverted_at_gate` with `failure_signal="check"` | Proposed change broke ruff/mypy/tests                       | Expected behaviour — gate did its job. No action needed unless it repeats with the same subject prefix (rate limit G1 will kick in automatically). |
| `approval_queue.jsonl` grows unbounded                                  | Operator not processing arch proposals                      | Review + approve/reject. Loop does not stall on a full queue — it simply stops proposing new arch changes until prior ones resolve. |
| Wiki-sync pushes fail with 401                                          | Gitea token expired                                         | Regenerate at Gitea → Settings → Applications; update `.claude/settings.local.json`. Loop retries next tick. |
| Watchdog never fires despite metric regressions                         | `auto_revert.py` task disabled in Scheduler, or `last_good.pt` missing | Confirm task is enabled + running every 2 min (Task Scheduler → History tab). If `last_good.pt` absent, the watchdog exits 0 with a log line — means no confirmed change yet, which is a fresh-install state. |
| UI spectator mode stays at "no snapshot yet"                            | Training service hasn't written `reports/graph_snapshot.json` | Confirm service is running and `snapshot_every` is set (default 100 steps). Service only writes after first consolidation-boundary tick. |
| `smoke_train.py` hangs on tinyshakespeare                               | Known short-line bug, now fixed                             | If it returns: check `_build_corpus_blocks` in `train_service.py` still stitches lines to ≥`chunk_size*4` tokens. |
| `.soma-loop/state/baseline_broken.json` appears                         | A watchdog confirmation criterion tripped 3 consecutive buckets | Loop halts automatically. Inspect the file: it records which criterion tripped and the last commit sha. Manually revert, delete the file, restart. |

**Circuit-breaker interaction with Task Scheduler:** the scheduled tasks keep firing even after the circuit breaker trips; the tick wrapper exits early (code 0) when it sees `.soma-loop/STOP` or `baseline_broken.json`. No Scheduler reconfiguration needed when resuming — just clear the state files.

## 8. Phase 1 → Phase 2 transition runbook

Per design §7.1 ("Approach 2 — Corpus scale-up"). Trigger: Phase 1 validated (§6.4 exit condition met + `audit_loop.py` clean).

```bash
# 1. Halt the loop cleanly
touch .soma-loop/STOP
touch .soma-loop/signals/shutdown
# wait for current tick + service to finish

# 2. Archive Phase 1 artefacts
mv checkpoints checkpoints.phase1
mv .soma-loop/state/change_log.jsonl .soma-loop/state/change_log.phase1.jsonl
cp configs/current.yaml configs/current.phase1.yaml

# 3. Download the blended corpus (script ships in Phase 2)
python scripts/download_corpus.py --target data/blended_corpus.txt
# outputs: 80% TinyStories + 20% DailyDialog, ~200 MB, gitignored

# 4. Queue the vocab bump as an arch-carveout change.
#    vocab_size is a shape key — carveout-protected — so route through the approval queue.
#    Append an entry to .soma-loop/state/approval_queue.jsonl by hand, then approve it:
#      {"id":"manual-vocab-bump-phase2","created_ts":"<iso>","path":"configs/current.yaml",
#       "key":"vocab_size","old_value":512,"new_value":8192,"rationale":"Phase 2 corpus scale-up"}
python scripts/approve.py --list           # confirm the entry is visible
python scripts/approve.py manual-vocab-bump-phase2

# 5. Re-bootstrap with the new corpus
python scripts/bootstrap_loop.py --force --corpus data/blended_corpus.txt

# 6. Refresh the KB against the new corpus context
claude --print --prompt "Use soma-bootstrap-kb skill for SOMA at $(pwd) (Phase 2 corpus scale-up)"

# 7. Restart service + re-enable Scheduler
python scripts/train_service.py &
rm .soma-loop/STOP
# enable Scheduler tasks
```

**Observation window:** 14 days. Rollback plan: `mv checkpoints.phase1 checkpoints && cp configs/current.phase1.yaml configs/current.yaml`, then re-bootstrap.

**Known risks (track explicitly in `reports/tick-summaries.md` during the first week):**
- The first ~10 K steps on the new corpus are erratic. The watchdog tolerance (20 %) may false-trigger; temporarily widen to 40 % in `configs/current.yaml` for the first 24 h if needed (route through carveout like any other change).
- `max_nodes` may need a bump (512 → 1024) as the richer corpus drives more growth. Let the loop propose it; approve when it does.

## 9. Cloud migration

Per design §7.4 ("Orchestration C — Linux cloud substrate"). Trigger: the local 3090 becomes a compute bottleneck (ticks routinely hit their 30-min Claude budget, or training throughput <20 steps/s after Phase 2 growth).

**9.1 VM selection:**

| GPU        | VRAM | ~Cost/hr | Notes                                             |
| ---------- | ---- | -------- | ------------------------------------------------- |
| A10G       | 24GB | ~$1      | Drop-in 3090 replacement                          |
| L40S       | 48GB | ~$1.5    | Headroom for Phase 3 curriculum + growth         |
| A100-40    | 40GB | ~$2      | Throughput, not capacity                          |
| A100-80    | 80GB | ~$3      | Phase 3+                                          |
| H100-80    | 80GB | ~$4      | Overkill for SOMA sizes; only if parallel trials  |

Start with A10G; promote only if growth saturates it.

**9.2 Migration procedure:**

```bash
# On the VM (Ubuntu 22.04+ assumed)
sudo apt install python3.11 python3.11-venv git rsync cron
git clone <gitea_url> /opt/soma && cd /opt/soma
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install claude-code                         # or follow Anthropic install docs

# One-time interactive auth (requires browser-assisted flow once)
claude login                                    # uses Max subscription tokens

# rsync state from Windows host
# (run from the Windows machine; adjust paths)
rsync -avz .soma-loop/state/ user@vm:/opt/soma/.soma-loop/state/
rsync -avz checkpoints/      user@vm:/opt/soma/checkpoints/

# Install systemd timers (preferred over cron — cron can't express "every 37 min")
sudo cp scripts/systemd/soma-loop.{service,timer} /etc/systemd/system/
sudo cp scripts/systemd/soma-watchdog.{service,timer} /etc/systemd/system/
sudo systemctl enable --now soma-loop.timer soma-watchdog.timer

# train_service runs as its own unit (long-lived)
sudo cp scripts/systemd/soma-train.service /etc/systemd/system/
sudo systemctl enable --now soma-train.service
```

`run_tick.sh` is already cross-platform. Windows-specific bits that flip:

| Windows                     | Linux                                 |
| --------------------------- | ------------------------------------- |
| Windows Task Scheduler      | `systemd timer` (preferred), cron fallback |
| `%USERPROFILE%\.claude`     | `~/.claude`                           |
| `taskkill /F`               | `kill -9` / `SIGKILL`                 |
| `CREATE_NEW_PROCESS_GROUP`  | `setsid` / `start_new_session=True`   |
| Local disk                  | VM persistent disk + snapshot backups |

**9.3 Auth survival:** Max tokens cached in `~/.claude/` survive reboots but occasionally need re-auth. Put a weekly calendar reminder to run `claude whoami`; if unauthenticated, re-run `claude login`.

**9.4 Backup strategy:** the VM provider's snapshot facility covers disk-level rollback, but supplement with:
```bash
# Nightly cron: off-disk checkpoint + change_log backup
0 3 * * * tar czf - /opt/soma/checkpoints/current.pt /opt/soma/checkpoints/last_good.pt /opt/soma/.soma-loop/state/change_log.jsonl | rclone rcat backblaze:soma-backup/$(date +\%F).tgz
```

**9.5 Validation on VM:** run the full §6.1 checklist again on the VM before enabling timers. Treat it as a fresh install — the tokens are per-host and the cron syntax differs.

---

*Last updated: 2026-04-12. Design reference: `docs/plans/2026-04-12-autonomous-loop-design.md`. Implementation plan: `docs/plans/2026-04-12-autonomous-loop-implementation.md`.*
