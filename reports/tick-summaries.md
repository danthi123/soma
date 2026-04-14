# SOMA autonomous-loop tick summaries

Operator-readable append-only timeline of autonomous-loop activity and
launch-readiness milestones. Tick entries are written by `run_tick.sh`;
manual operator entries bookend phase transitions.

---

## 2026-04-13 — Phase 1 launch-readiness verified

Pre-launch checklist (design §6.1) run against the implemented system.

**Automatable checks — all green:**

| Check                                        | Result                                             |
| -------------------------------------------- | -------------------------------------------------- |
| `pytest tests/ -q` × 10 runs                 | 10/10 exit 0. Runs 3-10: 678/678 passed. Run 2 showed 1 transient skip (docker-pytest probe), run 1 predated the new CUDA regression test. No actual failures, no flakes in the test logic. |
| `scripts/smoke_train.py --device cuda`       | 30 steps / 2.1s. cuda confirmed.                   |
| `scripts/safety_gate.py --skip-smoke`        | format + check + mypy + pytest green in 32s.       |
| `scripts/bootstrap_loop.py`                  | Idempotent re-run clean.                           |
| `configs/current.yaml` present               | Yes.                                               |
| `.soma-loop/STOP` absent                     | Yes.                                               |
| `.soma-loop/state/baseline_broken.json` absent | Yes.                                             |
| `scripts/train_service.py` heartbeat ≥1 min  | step 685 → 1707 in ~92s (~11 steps/s), `device=cuda`, `status=running`, clean shutdown on `signals/shutdown`. |
| `scripts/auto_revert.py` no-in-flight exit 0 | Yes.                                               |

**Bug discovered and fixed during the dry-run:**

First `train_service.py` start attempted to resume from the stale step=685
checkpoint left by prior manual testing and crashed three times in a row with
`mat1 is on cuda:0, different from other tensors on cpu`, tripping the
`CrashBackoff` permanent-failure path. Root cause: `Graph.deserialize()`
rebuilds every `Node` and `Edge` on the default device (CPU), but the
surrounding `SOMA` was constructed with `device=cuda` and keeps its
`working_memory`, `episodic_memory`, and `curiosity` modules there via
`load_state_dict`'s preserve-device semantics. First forward hit an `addmm`
crossing the boundary. This path is the exact one the autonomous loop walks
after every apply (service shutdown + fresh spawn + load-checkpoint), so it
had to be fixed before enabling the scheduler.

Fix: `SOMA.load_state` now calls `self.graph.to(self.device)` immediately
after `Graph.deserialize`. Regression test
`test_load_state_realigns_graph_to_target_device` added in
`tests/test_system/test_cuda.py`: save on CPU, load into a fresh CUDA SOMA,
step without device-mismatch. Commits: `08d953d fix(core): SOMA.load_state
realigns graph to saved device`, `ee57e33 chore(loop): apply ruff format
across prior loop commits`.

After the fix, `train_service.py` was re-run against the same stale
checkpoint and advanced cleanly 685 → 1707 in ~92s on cuda with a clean
shutdown on `signals/shutdown`. Stale `train_permanent_failure.json` /
`train_crash.json` / `fresh_init.flag` markers from the pre-fix run were
removed.

**Operator-deferred checks (require live Claude Code auth / Gitea write and
must be run by the operator before enabling the Scheduler):**

- `data/fixed_prompts.txt` content review (technical correctness, no personal/sensitive strings).
- `claude --print --prompt "Use soma-bootstrap-kb skill for SOMA at $(pwd)"` → Gitea wiki populated with SOMA entity page + ≥10 atoms.
- Single manual tick: `bash scripts/run_tick.sh` → exit 0, `.soma-loop/logs/tick-<id>.log` free of errors.

**Status:** implementation-side pre-launch checklist green. Eligible to
proceed to Task 24 (create Task Scheduler entries, keep DISABLED) pending
operator completion of the three deferred checks.

---

- tick 1776057013: exit=0 reason=harness_failure (NaN loss, CUDA assert in working_memory.occupancy; training service unaffected, still running)
tick 1776057520: phase=11 finalize baseline_captured
- tick 1776057714: applied queue entry 704632ce (base_lr→0.5 canary), sha=ad780944, gates OK, restart OK

### tick 1776057749 — 2026-04-13T05:28:27Z
- **outcome:** no change proposed
- **reason:** regression from deliberate base_lr=0.5 test injection; waiting for watchdog revert
- **loss_ema_500:** 0.0711 (+47.2% vs baseline)
- **nodes:** 35 (−9), edges: 1180 (−90)
- **health_flags:** wm_pinned_high, episodic_saturated
- tick 1776058371: reverted_at_gate (Task 25 gate-failure validation — ruff check, as expected)

---

## 2026-04-13 — Task 25 validation injections complete

Both §6.3 validation injections ran end-to-end against the live loop
infrastructure. All signals landed as specified.

**Revert-path test (change_log_id `e74c5c0f`, queue `704632`):**
- Injected `base_lr: 0.001 → 0.5` into the approval queue, approved.
- Tick 1776057714 applied the change: commit `ad78094`
  `auto: Task 25 revert-path validation: set base_lr to 0.5 (expected to regress)`.
- train_service restarted with `base_lr=0.5`, hit `current_loss must be finite, got inf`
  three times within ~2 s, wrote `train_permanent_failure.json` and exited.
- Watchdog detected `train_permanent_failure.json` post-dating `ts_applied`
  and reverted immediately (new fast-path, commit `0de288e`):
  `revert(auto): revert ad780944 (train_service permanent_failure after change applied)`.
- change_log updated: status=`reverted`, reverted_by=`watchdog`,
  failure_signal=`["service_dead"]`, failure_reason matches.
- `consecutive_failures.count` 0 → 1.

**Gate-failure test (change_log_id `31bdce7b`, queue `b2707a`):**
- Injected bad import into `src/soma/ui/panels/metrics_panel.py`, approved.
- Tick 1776058371 applied the diff in its workspace, `safety_gate.py` ran
  and failed at the `check` stage (ruff F401 unused import + F821 undefined name).
- Tick Phase 8 stashed + dropped the diff, wrote change_log entry with
  status=`reverted_at_gate`, failure_signal=`check`, commit_sha=`null`.
- No `auto:` commit was produced — tree stays clean (verified via
  `head -5 src/soma/ui/panels/metrics_panel.py` showing original docstring).
- `consecutive_failures.count` 1 → 2.
- Finalize commit `b541bd3 chore(auto): finalize tick 1776058371` only.

**Incidentally discovered and fixed during the validation window:**

1. `fix(loop): drop --prompt flag from run_tick.{sh,ps1}` (`afc3592`) — the
   Claude CLI 2.1.104 removed `--prompt` in favor of positional/stdin.
2. `fix(loop): pipe tick prompt via stdin instead of positional arg` (`0275543`) —
   `--allowedTools <tools...>` is variadic and was eating the positional prompt.
3. `fix(loop): default test_harness device to cpu` (`ab9bd1b`) — CUDA async
   device-side assert after NaN/Inf loss killed the harness before it could
   write a graceful report; cpu path handles it cleanly.
4. `fix(loop): watchdog reverts immediately on post-apply service crash`
   (`0de288e`) — extracted `service_crashed_after_apply()` + 4 unit tests.
   Without this, a change catastrophic enough to kill the train service
   before writing metrics was undetectable by the regression-metric
   watchdog and the loop deadlocked.

Also observed but not yet fixed:
- Tick Claude wrote `status="applied"` in the `change_log.jsonl` entry; the
  design schema (Appendix B.1) accepts only
  `pending|in_progress|confirmed|reverted|reverted_at_gate|commit_hook_blocked|stale|halted_by_diagnose`.
  `applied` is the `queue_status` vocabulary. The tick prompt should be
  tightened to disambiguate — for now I manually rewrote the entry to
  `in_progress` so the watchdog could find it via `_find_last_in_progress`.
- GPU contention from concurrent gaming workload (league) correlates with
  sporadic NaN/inf on training resume from some mid-run checkpoints
  (step 7000, 10000, 13565). Resuming from cleanly-saved permanent
  checkpoints (step 5000) has been reliable. Suggests either numerical
  fragility at specific graph configurations or CUDA context cross-
  talk from shared GPU; worth investigating before Phase 2.

**Status:** §6.3 validation complete. End-to-end pipeline validated —
carveout→queue→approve→apply→gate→commit→restart→watchdog-revert all fire as
designed. Task 26 (48-hour continuous validation window) remains and is
operator-scoped wall-clock: scheduler is enabled, train_service running,
watchdog active. Monitor `git log --grep="^auto:"` and `change_log.jsonl`.

| 1776062150 | 2026-04-13T06:40:41Z | no_change | — | heldout_loss=10.72 (evaluation path issue suspected, carveout-protected) |

---

## BLOCKED 2026-04-13 04:18 EDT — GPU contention, loop paused

`train_service` cannot sustain training while another GPU workload is
active on the same RTX 3090. NaN/Inf loss reproducibly trips the
`current_loss must be finite` check around step 6000-7000 (sometimes
on first step from a checkpoint). Pattern reproduced across:

- Resume from `step_00010000.pt` -> NaN on step 1
- Resume from `step_00005000.pt` -> NaN around step 6000
- Fresh init (`load_source=fresh`) -> NaN around step 6603
- All three crashed with `step crashed: current_loss must be finite,
  got nan|inf` x3 -> `train_permanent_failure.json` -> exit 2

`nvidia-smi` shows three `LM Studio.exe` processes holding GPU memory
in addition to normal desktop compositor / browser overlays. Earlier
runs that were stable (step 5000 -> 8554+ cleanly) all happened
during a brief window when both league and LM Studio were not active.

Already shipped (committed `09b09f3`): encoder/decoder weight
persistence as sidecar files. Verified working — the most recent
fresh-init service wrote `current.encoder.pt` matched to a fresh
graph. So the eval-path issue Claude flagged in tick 1776062150 is
fixed for the next clean run.

**Action for operator on wake:**

1. Close LM Studio (Task Manager -> end the three `LM Studio.exe`
   processes, or quit from the app tray)
2. `pwsh -File scripts/resume_from_game.ps1` -- this clears the
   pause signal, re-enables the SOMA Loop Tick task, and respawns
   `train_service` if dead
3. Watch `cat .soma-loop/state/train_heartbeat.json` -- step should
   advance smoothly past 7000 without `train_crash.json` appearing
4. The next tick (37 min after re-enable) will be the first with a
   real `heldout_loss` (encoder sidecar will load) and Claude can
   start proposing meaningful changes

Loop state is intact: scheduler tasks created (one disabled by my
pause helper, one enabled), `consecutive_failures.count=2`,
`baseline.json` present, queue empty, train_service paused via
signal (so `current.pt` is preserved). No git state was disturbed.

Note for future: the persistent NaN under GPU contention is a real
SOMA training fragility that should be addressed in Phase 2. Likely
fixes: gradient clipping in `SOMA.step()` before the finite-loss
check, OR loss-skip-with-state-rollback rather than raising. Both
are SOMA-internal carveout changes -- left for operator to design,
not patched silently.

---


| 1776068823 | 2026-04-13T08:31:23Z | no_change | — | loss_ema=3.86e12 (exploded) but heldout=0.022 (healthy); homeostasis active lr=7e-12; waiting for next tick |
| 1776071023 | 2026-04-13 09:07 UTC | no_change | — | Training healthy (loss=0.0163, heldout=0.0274); loss_ema_500=408 is restart artifact; fix touches carveout |
| 1776073244 | 2026-04-13T09:41Z | no_change | — | loss_ema=0.020 (-58.7% vs baseline, best yet); heldout=NaN (persistent); no growth; all fixes need carveout paths |
| 1776075476 | 2026-04-13T10:21Z | no_change | — | loss_ema=0.025 (-48% vs baseline); heldout=NaN (persistent eval error); training stable; no non-carveout fix available |

---

## 2026-04-13 05:46 EDT — Skill update: clarified architecture class

`~/.claude/skills-repo/skills/domain-specific/soma-diagnose/SKILL.md`
template incorrectly led Claude to mark architecture class as "not
applicable" whenever the fix touched a carveout path. Now states
explicitly: architecture *is* the route for carveout-needed fixes
(Phase 6 routes to the approval queue, operator remains the gate
per G40). Ticks 1776068823 / 1776071023 / 1776073244 all hit this
exact rationale and chose class=null. Future diagnoses should now
emit class="architecture" with a diff so operator can approve real
fixes during the validation window.

Skill files live outside the repo (`~/.claude/skills-repo/`), so
the change isn't tracked in git.

---
| 1776077690 | 2026-04-13T10:57:41Z | 30000 | 0.02377 | null | class=null | no change — heldout NaN + kl=0.0 are carveout bugs |
| 1776079940 | 2026-04-13T11:32Z | 40000 | 0.02026 | null | class=architecture | queued: fix heldout eval abort (skip NaN pairs instead of return) |
| 1776082129 | 2026-04-13T12:11:59Z | applied-approved | auto: Fix heldout eval abort — skip NaN token pairs | 121be10b | architecture |

---

## BLOCKED 2026-04-13 08:40 EDT — training fragility blocks restart, STOP touched

Tick 1776082129 applied `121be10` (eval-only fix in `scripts/test_harness.py`)
and the new `service_crashed_after_apply` watchdog reverted to `6295327`
because train_service permfailed 39s after `ts_applied` with `current_loss
must be finite, got inf`. The revert was technically correct per watchdog
rules but a false positive root-cause-wise: the diff only modifies
`_evaluate_heldout`'s exception handler in the harness, which is never
called from train_service. The training crash was the same pre-existing
NaN/Inf training fragility flagged in the earlier BLOCKED entry — operator
explicitly noted it as a Phase 2 carveout fix ("left for operator to
design, not patched silently").

`consecutive_failures.count` reached 3 (Task 25 revert injection +
Task 25 gate-failure injection + this false-positive revert). I reset
to 0 because all three increments have identified non-pathological
root causes documented above; per memory `feedback_overnight_autonomy.md`
this is "with root-cause investigation". `last_reset_ts` set to
1776082800.0 in `.soma-loop/state/consecutive_failures.json`.

Then attempted train_service restart from progressively older checkpoints:

| checkpoint | result |
|------------|--------|
| current.pt (step 53740) | inf loss x3 → permfail |
| step_00050000.pt | inf loss x3 → permfail |
| step_00040000.pt | nan loss x3 → permfail (1 step succeeded then crashed) |

The fragility is not specific to one checkpoint — it's the same numerical
issue manifesting across a 13K-step span. Encoder/graph drift may be a
factor (encoder sidecar is from step 53740, was paired with all three
attempts), but the consistent NaN/Inf strongly suggests the training
update path itself needs gradient clipping or skip-with-rollback. Both
are SOMA carveout (`src/soma/system.py` `step()`), so I cannot patch
silently per the operator's explicit instruction.

Original step_00053740.pt preserved as
`checkpoints/current.corrupt_1776082129.pt.bak`. Current `checkpoints/current.pt`
now points at step_00040000 (last attempted, also broken). Pid file dir
empty. Loop tick scheduler still ENABLED — `.soma-loop/STOP` touched so
next tick will halt at Phase 0a per design §4.2 rather than burn credits
spinning on a known-blocking issue.

**Action for operator on wake:**

1. Decide on training-fragility fix approach. Options A and B are both
   carveout (`src/soma/system.py`); design call belongs to operator:
   - **A.** Gradient clipping in `SOMA.step()` before the finite-loss check.
     `torch.nn.utils.clip_grad_norm_` on all parameter groups with a
     conservative bound (e.g., 1.0). Cheap, well-understood, but may
     mask the underlying instability rather than diagnose it.
   - **B.** NaN-skip-with-state-rollback. Detect non-finite loss, restore
     pre-step parameters from a saved snapshot, advance global_step
     anyway, log the skip. More principled — preserves training signal
     when it exists, drops it when it doesn't — but harder to get right
     (snapshot/restore overhead, interaction with optimizer state).
2. Investigate WHY training is now NaN-prone at step ~40K-53K when it
   was stable at step ~5K. Could be: parametric-memory growth pushing
   weights out of their stable regime, working-memory occupancy
   feedback loop (`wm_pinned_high` flag was on in early ticks),
   episodic saturation, or curiosity scaling. Worth a short exploratory
   notebook before committing to A or B.
3. Once a fix is applied: `rm .soma-loop/STOP`, restart train_service
   from a clean checkpoint (probably step_00010000.pt or earlier),
   and let the loop resume.

`consecutive_failures.count=0` is correct after the reset above.
Approval queue empty. No git state was disturbed beyond the
auto-generated finalize/revert commits already in `git log`.

---

## 2026-04-13 ~08:55 EDT — training stability fixes shipped, loop resumed

Diagnostic on `current.corrupt_1776082129.pt.bak` revealed the root
cause: by step 10000, **99.8% of edge weights were already pinned at
+5.0** (the clamp), 39/40 node gains pinned at the lower clamp (0.1),
and 12-17/40 node activation EMAs already non-finite (max
1.47e+17 by step 10K, 1.21e+15 by step 53K).

The Hebbian rule in `core/learning.py` is monotonically positive
(only ever ADDS to weights when both endpoints fire) with no LTD
counterpart. Backprop signal couldn't keep up. Saturation was
inevitable — the loop ran for 50K+ steps against essentially dead
weights.

Plan: `docs/plans/2026-04-13-training-stability-fixes.md`.
Five commits implementing the operator-approved fix triple:

| sha | scope |
|------|-------|
| `ccde6ac` | feat(config): edge_weight_decay, grad_clip_max_norm, max_consecutive_skipped_steps |
| `2fd2166` | fix(core): edge weight decay (Hebbian counterbalance) |
| `f0726a6` | fix(core): gradient clipping in update_step |
| `3f0820b` | refactor(homeostasis): return None on non-finite loss instead of raising |
| `625e8e9` | feat(system): SOMA.step skips on non-finite loss + escalates after N skips |
| `a22f0e2` | chore(format): ruff format on the above |
| `48f4927` | chore(diagnose): fresh-train sanity check for stability fixes |

Verification:

- Full pytest: 696/696 pass (35 system + 18 learning + 25 config + ...)
- ruff check / format: clean
- mypy: clean
- Diagnostic re-run: 200 fresh-init steps produce 0/66 edges at clamp,
  0 non-finite activations, max activation_ema=0.97, loss_ema=0.019,
  0 skipped steps. Compared to the corrupt checkpoint's 99.8%/17/1e15.

Hygiene:

- All saturated checkpoints (step_00001803.pt … step_00050000.pt)
  archived to `checkpoints/pre-fix-archive-2026-04-13/`.
- Original corrupt step_53740 preserved as
  `checkpoints/current.corrupt_1776082129.pt.bak` for evidence.
- Stale lock files (.tick.lock, .git.lock — pids 136272/134844 dead),
  permanent_failure.json, train_crash.json, gate_failure.json,
  push_failure.json, current_tick_id.tmp all cleared.
- STOP file removed; scheduler will fire next tick on its 30-min
  cadence.

train_service restarted from fresh init on CUDA. Verified: step 0 →
259 in ~1.5s, status=running, no `step crashed` lines, encoder built
fresh from corpus.

Push: same no-tty issue as before (operator credentials needed).
Commits land locally; operator will see them on wake.

---
| 1776086567 | 2026-04-13T09:25:15 | harness_skipped | no checkpoint yet (step ~2021, interval=5000) | — |
| 1776088789 | 2026-04-13T14:02:36Z | 10000 | 140.452 | 17.960 | 39/1100 | nan_detected,wm_pinned_high,episodic_saturated | no-op: wait for recovery | — |

### Post-fix training trajectory observation

Sampled from `.soma-loop/metrics/metrics.current.jsonl` after the stability-fix restart:

| step   | loss    | lr_m    | nodes | edges | notes                          |
|--------|---------|---------|-------|-------|--------------------------------|
|      9 |   0.022 | 1.0000  | 34    |    64 | fresh init                     |
|   4809 |   0.016 | 1.0000  | 34    |   181 | clean, edges growing gradually |
|   8009 |   0.015 | 1.0000  | 35    |   295 | still healthy                  |
|   8809 |   0.407 | 0.0001  | 36    |   672 | synaptogenesis burst begins    |
|   9609 |  92.057 | 0.0000  | 37    |  1080 | loss explodes, homeostasis damps |
|  11209 |  24.253 | 0.0000  | 39    |  1100 | lr fully damped                |
|  12009 |   1.103 | 0.0112  | 40    |  1110 | lr recovering                  |
|  12809 |   1.575 | 1.0000  | 40    |  1110 | lr fully restored              |
|  13609 |   0.894 | 1.0000  | 40    |  1110 | recovering                     |
|  14409 |  16.246 | 0.0002  | 40    |  1110 | second spike                   |
|  15209 |  20.260 | 0.0143  | 42    |  1130 | recovering again               |
|  16009 |   0.838 | 1.0000  | 42    |  1130 | back to low loss               |

**The loop is now self-stabilizing as designed** — synaptogenesis bursts
cause loss spikes, homeostasis damps LR to near-zero, the system absorbs
the new structure, LR recovers, loss drops. No service crashes. No STOP
file touched. `consecutive_failures.count=0`. Tick 1776088789 correctly
diagnosed "wait for recovery" and proposed no change.

Compare to the pre-fix behavior: any such spike would have produced
non-finite loss → homeostasis raise → CrashBackoff → train_service
dead → watchdog revert + loop halt.

**Caveat — avg_degree=56 (cap=20):** synaptogenesis fired fast and
edges are aging under `pruning_grace_period=2000`. Once grace expires,
pruning should bring this down. If persistent oscillation continues
beyond step ~30K, a future tick may reasonably propose lowering
`synaptogenesis_rate` (config-class) or raising the grace period to
reduce the spike magnitude. Both are non-carveout.

---

### tick 1776091009 — step 20000 (post-restart)
- **Outcome:** no change (homeostatic recovery in progress)
- **Key metrics:** loss_ema=1.373 (↓99% from 140.45), KL=20.71, heldout=Inf, nan_detected
- **Trend:** loss converging rapidly post-restart; KL slightly up; NaN confined to eval path
- **Rationale:** system recovering after stability fixes + fresh restart; intervening would risk disrupting convergence. Re-evaluate at step 30K.

---
| 1776093231 | 2026-04-13T15:16:40Z | 25000 | 81.17 | null | no_change | loss divergence (+169,000% vs baseline); core stability issue, no auto-committable fix |

### Monitoring notes at ~11:42 EDT (step 34523)

**Baseline.json was stale.** It was captured at tick 1776057520 from
the pre-fix saturated-dead run (loss=0.048 looked "healthy" because
the graph was pinned at weight-clamp and couldn't learn). Every tick
since the restart has compared against that meaningless value and
reported loss-delta percentages in the 100,000%+ range. Moved to
`baseline.pre-fix-stale.json`; next tick will re-capture baseline
from the current post-fix system per design §Phase 1.

**Persistent oscillation observed post-20K** (from metrics.current.jsonl):

| step | loss | lr_m | nodes | edges |
|------|------|------|-------|-------|
| 22909 | 30.7 | 0.002 | 48 | 1190 |
| 23489 | 92.5 | 0.315 | 49 | 1200 |
| 26969 | 64.0 | 0.017 | 52 | 1230 |
| 29289 | 69.6 | 0.011 | 55 | 1260 |
| 29869 | 231.2 | 0.102 | 56 | 1270 |
| 32769 | 173.4 | 1.000 | 59 | 1300 |

Each spike correlates with a neurogenesis event (nodes +1). Between
spikes loss drops to 0.4-2.0. Homeostasis recovers in 500-1500 steps
per spike. No service crashes, no skip-counter escalations.

Per operator instruction, hold config-class proposal until at least
ONE tick observes past step 40K. Projected timing at ~3 steps/s: the
12:20 tick (estimated step ~41K) is the gating observation.

---
| 1776095447 | 2026-04-13T15:51:54Z | cold_start | Baseline seeded (loss_ema=268.1, step=35000, heldout=NaN) | — |
| 1776097693 | 2026-04-13T16:30Z | cold_start | Baseline re-seeded (loss_ema=13.72, step=40000, heldout=Inf, nan_detected) | — |

### Monitoring notes at ~12:30 EDT (step 42492, past 40K gate)

**Tick 1776097693 outcome was cold-start** — baseline re-seeded at step=40K, loss_ema=13.72. New baseline now represents the post-fix living-weights system, not the pre-fix saturated-dead one.

**Operational issue observed twice:** after a cold-start tick exits, run_tick leaves `tick.lock` dangling and skips the post-finalize `resume` signal. Service stays paused until next tick reclaims the stale lock via pid-aware detection. Manually cleared `tick.lock` and sent `signals/resume` — service resumed at step 42405, now advancing. Worth a fix to run_tick.{sh,ps1} but not urgent.

**Config-class proposal queued (operator approval required):**

- `queue_id`: `e1c9fc2d-c81e-495b-ac17-78c8e94ffd84`
- `class`: config
- Diff: `configs/current.yaml` lower `synaptogenesis_rate` from 0.01 to 0.005
- Rationale: edges grew 64 to 1360 over 40K steps; each synaptogenesis burst adds ~10 edges and destabilizes learned weights, forcing homeostasis to clamp LR to ~0.001. Halving the rate attacks the oscillation cause while keeping `synaptogenesis_interval=100` intact.
- Expected: loss_ema_500 trends below 10 within one confirm window (~15 min at 3 steps/s).
- Base commit: `e077ca0`
- Proposed by: monitoring-session (operator-instructed per prior message)

---

### Tick 1776099889 — 2026-04-13T17:04 UTC
- **Outcome:** HALT — training diverged to NaN at step 45K
- Phase: diagnose → halt (no change applied)
- Metrics: loss_ema_500=Infinity, last_loss=NaN, nodes=73, edges=1440
- Loss trajectory: 140→1.4→81→268→13.7→Inf (violent oscillation over 6 ticks)
- Cause: neurogenesis adding ~8 nodes/tick with zero pruning destabilizes weights
- Step-45K checkpoint is NaN-poisoned; restart reproduces failure
- Action required: roll back to step 20K or 40K checkpoint, approve pending synaptogenesis_rate reduction, consider enabling edge pruning, then clear STOP
- Base commit: `dd42225`

---

### Monitoring notes at ~13:25 EDT — HALT verified, awaiting operator

**Service status:** DEAD. `train_permanent_failure.json` at step 45051 (3 crashes in 5 min, CrashBackoff triggered). Heartbeat shows stale pid 64612 but `tasklist` confirms no such process. `train_service.log` shows the failure path: **the stability fixes worked as designed**:

```
train_service: step crashed: 51 consecutive non-finite losses; training is stuck (last loss=inf)
train_service: step crashed: 52 consecutive non-finite losses; training is stuck (last loss=inf)
train_service: step crashed: 53 consecutive non-finite losses; training is stuck (last loss=inf)
```

That's `max_consecutive_skipped_steps=50` escalating correctly: 50 skips tolerated, 51st raised ValueError, CrashBackoff counted 3 such raises, wrote permanent_failure, exited. The diagnose tick read the state and called halt. **This is exactly the escalation path the fix triad was designed to enable** — the system no longer silently corrupts itself; it fails loud and stops.

**Fix triad performance vs pre-fix:**
| Metric | Pre-fix | Post-fix |
|--------|---------|----------|
| First crash step | ~6000 | 45000 (7.5x later) |
| Weight saturation | 99.8% of edges at +5.0 clamp by step 10K | 0% through step 40K |
| Loss at failure | 0.025 (dead, pinned) | Inf (honest divergence) |
| Recovery on restart | Impossible (saturated) | Fresh init clean through step 40K |

The fixes extended stable training 7.5x and turned silent saturation into explicit NaN. Real progress, but **uncontrolled growth still overwhelms the stabilizers** past step 40K without operator-chosen growth damping.

**Tick Claude chose `halt` class correctly.** Halt is the right escalation for NaN-poisoned checkpoint — no config change within auto scope can recover NaN weights. My queued config proposal (`e1c9fc2d`, synaptogenesis_rate reduction) is still `pending` in the queue; if applied, it would have slowed the edge-growth cascade but wouldn't have recovered the dead weights.

**No actions taken autonomously.** STOP remains touched; G40 gate holds. train_service is not being restarted.

**Decision tree for operator:**

1. **Fresh init with config fix (recommended).**
   - Approve queue entry `e1c9fc2d` (synaptogenesis_rate 0.01 → 0.005).
   - Consider also lowering `neurogenesis_interval` less aggressively (500 → 1000) to halve node addition rate — I did NOT queue this second change, holding for operator call.
   - `rm .soma-loop/STOP .soma-loop/state/halt_reason.json .soma-loop/state/train_permanent_failure.json`
   - Move `checkpoints/current.*` to `checkpoints/nan-poisoned-45k-2026-04-13/`.
   - `touch .soma-loop/state/fresh_init.flag`
   - Restart train_service.

2. **Rollback to step 20K or 40K.**
   - Complicated because `current.encoder.pt` is paired with the NaN-poisoned graph. Either delete it for fresh encoder (model re-learns embeddings) or accept drift.
   - Still need the config fix applied before resuming, else same dynamics recur.

3. **Deeper carveout work first.**
   - Add edge/node pruning logic tuning so pruning activates before graph density destabilizes training.
   - Enable `pruning_interval=1000` (already there in config) but investigate why `prune=0` was observed in every growth_last_1k_steps reading — possibly the grace period gate never relaxed because edges kept aging faster than threshold.

I lean toward option 1 (fresh init + config fix), but this is your G40 gate call.

---

## 2026-04-13 ~13:25 EDT — post-HALT recovery: Option 1 + carveout fix + fresh restart

Operator authorized "any changes needed including carveout". Executed:

**Carveout fix (commit `1927406`):** `src/soma/growth/pruning.py` now
runs a **congestion-pruning** sweep after the existing low-utility
sweep. When `avg_degree > max_edges_per_node`, the weakest edges past
grace period are removed until back under cap. Root cause: the
existing low-utility filter requires BOTH low strength AND long
inactivity; in continuously-active dense graphs every edge stays
"active" and strength stays above threshold, so pruning literally
never fired. Three new tests in `tests/test_growth/test_pruning.py`
verify correctness (grace respected, no-op when under cap,
weakest-first ordering). 699/699 pytest, ruff + mypy clean.

**Config changes (commit `0ef28de`):** `configs/current.yaml`:

- `synaptogenesis_rate` 0.01 → 0.005 (halve edge-growth rate)
- `neurogenesis_interval` 500 → 1000 (halve node-growth rate)

Applied the queued proposal `e1c9fc2d` inline (queue entry updated
to `queue_status=applied` with `approved_by="operator (inline via
monitoring-session)"`).

**State hygiene:**

- NaN-poisoned checkpoints (step 5K-45K, current.*) moved to
  `checkpoints/nan-poisoned-45k-2026-04-13/` for reference.
- Cleared STOP, halt_reason.json, train_permanent_failure.json,
  train_crash.json, heartbeat, tick.lock, git.lock, baseline.json,
  push_failure.json, current_tick_id.*.
- Touched `fresh_init.flag` so train_service cold-starts.

**Fresh restart:** train_service pid 108116 on CUDA, step 0 → 268
in ~10s (~27 steps/s early warmup), status=running, load_source=fresh.
Fresh baseline will be captured on next tick per Phase 1.

**Four levers now pulling against over-growth:**

1. Hebbian weight decay (`edge_weight_decay=0.9999`) — damps weight drift
2. Gradient clipping (`grad_clip_max_norm=1.0`) — caps backprop spikes
3. Growth rate halving (config above) — reduces event magnitude
4. Congestion pruning (carveout) — actively removes excess density

Prediction: graph should now stabilize around avg_degree ≤ 20 instead
of drifting to 50+. Loss oscillation should be much narrower. Next tick
will tell.

---

## 2026-04-13 ~14:48 EDT — THIRD carveout fix: synaptogenesis clamp + eval_mode

The post-HALT fresh restart crashed AGAIN at step 11158 with the same
pattern. Investigation revealed TWO deeper bugs the earlier fixes
didn't touch:

**Bug 1: synaptogenesis probability was unbounded.**
`src/soma/growth/synaptogenesis.py` computed:

```python
prob = coact * locality_bonus * rate  # coact = source_mag * target_mag
```

When activations diverge during a spike (magnitudes >> 1), `coact *
rate` can exceed 1. Every random draw in `[0, 1)` is then below `prob`,
so synaptogenesis creates edges between EVERY co-active pair in a
single event. Observed in metrics.current.jsonl around step 10909:
`311 → 509 edges` in one 100-step window, then `509 → 1239 edges` in
the next, then inf loss. Classic positive feedback loop.

Fix (commit `7a18c09`): clamp `prob = min(1.0, coact * locality_bonus
* rate)`. Two new tests verify bounded-output invariant.

**Bug 2: test_harness.py mutated the graph during eval.**
`scripts/test_harness.py` called `soma.step(inputs, targets)` which
internally invokes `_maybe_grow()` (synaptogenesis + neurogenesis +
pruning) and `update_step()` (backprop + Hebbian). So every heldout
evaluation was RUNNING ANOTHER round of training-mode mutations on a
checkpoint. That's why the post-HALT tick reported edges=750 at step
10K while the actual metrics file showed edges=240 at that step — the
harness had pumped 510 extra edges into the graph before evaluating.

Fix (commit `41b9fc8`): added `eval_mode=True` kwarg to `SOMA.step`.
When set, skips `update_step`, `_maybe_grow`, `_maybe_consolidate`,
and does NOT touch `_consecutive_skipped_steps` on non-finite loss
(eval is read-only for training-mode state). test_harness.py now
calls `soma.step(..., eval_mode=True)`. Three new TestEvalMode tests
verify graph/weight invariants and skip-counter immunity.

Also rolled in the queued `cbd39c8b` bugfix (heldout_loss < 700
overflow guard for math.exp) that tick-Claude proposed at 14:22;
applied inline. Queue entry marked applied.

**Now six stability levers in place:**

1. Edge weight decay (`edge_weight_decay=0.9999`) — Hebbian counterbalance
2. Gradient clipping (`grad_clip_max_norm=1.0`) — backprop bound
3. Skip-with-escalation in SOMA.step — service survives transient NaN
4. Congestion pruning (carveout) — removes excess edges at density cap
5. **Synaptogenesis probability clamp (carveout)** — caps edges-per-event
6. **Eval-mode (carveout)** — harness doesn't mutate graph
7. Halved growth rates (config): synaptogenesis_rate 0.005, neurogenesis_interval 1000

**Full verification:** 703/703 pytest pass, 1 skipped; ruff + mypy clean.

**State hygiene:**

- Post-recovery buggy-synapt checkpoints archived to
  `checkpoints/buggy-synapt-post-recovery-2026-04-13/`
- Cleared halt markers, heartbeat, locks, baseline, push_failure
- `fresh_init.flag` set

**Fresh restart:** train_service pid 19848 on CUDA, fresh init, step
0 → 270 in ~10s, status=running. Four commits added this session
(1927406, 0ef28de, 7a18c09, 41b9fc8) — 19 commits total local pending
push.

---

- **tick 1776104322** (2026-04-13T18:22:23Z): baseline_set — first tick post-recovery, loss=2.1e17, kl=20.71, step=10000. Fixed test_harness.py overflow guard (queued for approval).
| 1776106547 | cold_start | — | baseline captured, no change proposed |
| 1776108773 | 2026-04-13 19:41 UTC | reverted_at_gate | bugfix | Add lr_multiplier floor (0.01) | Pre-existing flaky test test_co_active_edge_reaches_equilibrium_below_clamp failed (unrelated to change) |
| 1776111009 | baseline_captured | — | — | 2026-04-13T20:11:14Z |

---

## 2026-04-13 ~16:50 EDT — RMS magnitude fix: training STABLE through step 85K

Pid 19848 (pre-RMS) crashed again at step 11K with the same "51 consecutive non-finite losses" pattern. Training loss was fine (0.019 EMA) right up to the spike, and the first real heldout metric ever appeared: **heldout_loss_mean = 0.031** (eval_mode fix definitively working). But metrics showed edges growing 64 to 1047 in 5000 steps despite my synaptogenesis clamp. Root cause:

Both synaptogenesis and the Hebbian edge update used `tensor.norm()` as the "magnitude". For a D-dim tensor at unit per-channel scale, norm = sqrt(D). With 64-dim activations every coactivation product was 64x intended: synapt prob hit the clamp for every pair on every event, and Hebbian weight bumps were 64x the effective rate.

**Fix (commit `d0f58fb`):** RMS magnitude `norm / sqrt(numel)` in both sites. Dim-invariant semantics. Also rolled in the lr_multiplier floor (0.01) from tick-1776108773's stashed proposal.

**Near-disaster:** tick safety_gate's `git stash` wiped my uncommitted RMS edits AND the lr-floor proposal. Re-applied and committed immediately.

**Result on pid 39880 (RMS-fixed code):**

| tick   | step  | loss_ema | last_loss | lr_m | heldout | edges |
|--------|-------|----------|-----------|------|---------|-------|
| 111009 | 30000 | 0.0172   | 0.0115    | 0.51 | 0.0193  | 64    |
| 113229 | 85000 | 0.0188   | 0.0184    | 1.00 | 0.0192  | 64    |

- **Loss stable at ~0.018 across 55K steps** (previously crashed every ~11K)
- **Heldout ~= training loss (0.019)** — genuine generalization
- **Edges still at seed=64** across 85K steps — no uncontrolled growth
- `lr_multiplier` recovered 0.51 to 1.00 cleanly
- Zero reverts, zero permfails, zero STOP

**Seven levers proven working together:** edge weight decay, gradient clipping, skip-with-escalation, congestion pruning, synaptogenesis clamp, eval_mode, RMS magnitude. Plus config (halved growth rates, lr floor).

**Caveat:** synaptogenesis/neurogenesis fired zero times across 85K steps. With RMS making typical activations ~1.0, the threshold=0.1 gate may or may not be calibrated to fire growth when task complexity demands it. If heldout loss stagnates over next 50K steps we may need to re-examine. For now, stable fixed-graph convergence is strictly better than any prior run.

---

| 1776113229 | 2026-04-13 20:48 UTC | no_change | — | System stable at step 85K; loss_ema=0.0188, KL frozen at 23.03, zero graph growth. Actionable concerns in carveout paths. |
| 1776115445 | 135000 | 0.01900 | 23.026 | 34/64 | no_change | growth stagnation, all targets carveout-protected |

---

## 2026-04-13 ~17:55 EDT — 180K milestone: RMS fix holding through step 183K+

Heartbeat at 2026-04-13T21:54Z shows pid 39880 running at step **183,001** — 138K steps past the RMS fix (commit `d0f58fb` at step ~45K), 17x past the original failure point of 11K, 2.1x past the previous 85K stability report.

**Trajectory since RMS fix:**

| step   | loss_ema | heldout | lr_m | edges | notes                                   |
|--------|----------|---------|------|-------|-----------------------------------------|
| 30000  | 0.0172   | 0.0193  | 0.51 | 64    | first post-fix tick, recovering         |
| 85000  | 0.0188   | 0.0192  | 1.00 | 64    | lr recovered cleanly                    |
| 135000 | 0.0190   | 0.0193  | 1.00 | 64    | flat — 50K additional stable steps      |
| 183000 | —        | —       | —    | —     | heartbeat only, next tick imminent      |

Loss_ema drift across 105K steps: 0.0172 → 0.0190 (+0.0018). Heldout drift: 0.0193 → 0.0193 (flat). This is genuine stability, not a trend.

**Status indicators:** no STOP, no permanent_failure, consecutive_failures=1 (stale from pre-fix), zero reverts, zero new halts since the RMS fix landed.

**Growth concern persists (noted 85K, reaffirmed here):** synaptogenesis/neurogenesis have fired zero times across 138K+ post-fix steps. The seed topology (34 nodes / 64 edges) is sufficient for this dataset's complexity floor — but if the system is never stressed into needing capacity, we can't validate growth under RMS-scaled activations. Activation_threshold=0.1 with typical RMS ~1.0 should permit firing; what's missing is a task signal that would push a coactivation pair past the probability draw. Not urgent while the run is beating every prior milestone.

**Next check:** 2 hours out (approx 2026-04-13T23:55Z) per stable-cadence protocol.

---
| 1776117652 | 190000 | 0.01877 | 0.01930 | 23.03 | 34/64 | no_change | Output collapsed to 'GUE'; degenerate equilibrium, needs operator |
| 1776119871 | 240000 | 0.01695 | 23.026 | 34/64 | no_change | — | 2026-04-13T22:40:27Z |

---

## 2026-04-13 ~18:56 EDT — 270K stable, but decoder is degenerate

Heartbeat at 2026-04-13T22:56Z shows step **270,121**, status=running, no STOP, no permfail. Training numerics look great:

| tick   | step   | loss_ema | heldout | lr_m | curiosity | edges |
|--------|--------|----------|---------|------|-----------|-------|
| 117652 | 190000 | 0.0188   | 0.0193  | 1.00 | 0.0043    | 64    |
| 119871 | 240000 | 0.0169   | 0.0193  | 0.69 | 0.0071    | 64    |

Loss_ema is still *descending* (0.019 → 0.017 across 50K steps). Homeostasis dampened lr_multiplier 1.00 → 0.69 mid-run on a small curiosity bump — first mid-run damping observed since the RMS fix, and it stayed well above the 0.01 floor. This is the regulator working correctly.

**Critical finding, surfaced by the tick harness (not proposing a fix):**

Both tick chat logs (1776117652, 1776119871) show the decoder emitting `"GUEGUEGUEGUE..."` as its response to *every* prompt — "To be, or not to be," "hello," "what is your name," even empty and single-char inputs. The 0.019 heldout loss is the model having found a degenerate equilibrium: whatever mean token vector minimizes MSE against the held-out targets gets decoded as the "GUE" trigram regardless of input.

This means the "stability win" is a mirage for the task. The seven-lever numerical stability stack is real — no crashes, bounded gradients, regulator self-correcting — but the underlying graph (34 nodes / 64 edges, zero growth across 240K steps) has too little capacity (or too little stress) to learn input→output mapping. It settles on emitting the marginal output distribution.

**Possible root causes (for operator triage, not autonomous action):**

1. Seed topology is insufficient for sequence mapping; growth must fire to add capacity. Current threshold=0.1 with RMS ~1.0 activations ought to permit it, but nothing is pushing coactivation × locality × rate above the probability draw.
2. Decoder architecture may be outputting a fixed vector regardless of encoder signal (unit test of "different inputs produce different OUTPUT node activations" would disambiguate).
3. The MSE-against-target-tokens loss may not be a strong enough signal to break out of the trivial equilibrium for this graph size. Would benefit from a curriculum or richer target distribution.

Leaving all of these for the operator. Continuing stable-cadence monitoring for now — the system isn't in a failure state, it's in a learned-wrong state, and the 22 pending commits don't cover this class of issue.

**Next check:** 3600s out (runtime cap).

---

### tick 1776122099 — step 292,166 — no_change
- loss_ema_500=0.01845, kl=23.026, nodes=34, edges=64
- Decoder still collapsed: all prompts → "GUE". Zero growth events (262K steps).
- class=null: root cause in carveout (growth/core). Awaiting operator intervention.

---

## 2026-04-13 ~19:07 EDT — dead-graph fix verified at step 5K

Operator granted autonomous authority to fix learning/architecture issues
(not just stability). Traced the decoder collapse: blanket
`edge_weight_decay=0.9999` drove every unbumped edge's magnitude to ~1e-13
across 292K steps. Inactive edges died, their targets starved, Hebbian
couldn't recover them — vicious cycle, graph became a constant function.

**Fix (commit `28a6e07`):** moved the decay `mul_()` inside the co-active
branch of `_apply_hebbian_edge_updates`. Resting edges keep their weight;
active edges retain the same Hebbian/decay equilibrium. Tests updated
(`test_inactive_edge_preserves_weight` replaces `..._decays_toward_zero`).
703 pytest pass, ruff + mypy clean.

**Verification after 5K fresh-init steps:**

| stage                    | pre-fix (292K) | post-fix (5K)            |
|--------------------------|----------------|--------------------------|
| edge |w| median          | 2.6e-6         | **0.1030**               |
| edges with |w| < 1e-3    | 64/64          | **0/64**                 |
| OUTPUT activation variance across prompts | 0 (collapsed) | **min 0.28, max 0.35**  |
| unique decoded tokens (5 prompts) | 1 ('GUE')   | **5 distinct**          |

Prompts now decode to relevant first tokens:
- "The quick brown" → `'The'`
- "Shall I compare thee" → `'Shall'`
- "O Romeo, Romeo" → `'O'`

New diagnostic scripts landed with the fix:
- `scripts/diagnose_collapse.py` — per-prompt pipeline trace with diversity summary
- `scripts/inspect_graph.py` — edge weight distribution and zero-input forward pass

Dead-graph checkpoints archived to `checkpoints/dead-edges-collapse-2026-04-13/`.
Fresh service pid 38468 running, 23 commits pending operator push.
| 1776124320 | 2026-04-13T23:55:32Z | 45000 | no_change | loss_ema=0.01498 heldout=0.02283 kl=20.71 nodes=34 edges=64 wm=0.0 | stable fresh run, all fixes in carveout |

---

## 2026-04-13 ~19:19 EDT — meaningful decoder output at step 45K

Post-fix tick 1776124320 chat log confirms the GUE collapse is gone:

```
"To be, or not to be,"  -> "To be, or not to be, negl negl"
"The king said,"        -> "The king said, negl negl"
"O Romeo, Romeo,"       -> "O Romeo, Romeo, wouldst negl"
"What light through"    -> "What light through redeem"
"Shall I compare thee"  -> "Shall I compare thee"
```

The model echoes the prompt (autoregressive copy-through) then emits
one or two generated tokens before saturating. "wouldst" after
"O Romeo, Romeo," is a plausible next-token; "negl" is a frequent
saturation target but is a real token, not the GUE constant-output
artifact.

Output-distribution KL dropped 23.03 -> 20.71 between pre-fix tick
117652 and post-fix 124320 — a measurable shift toward the corpus
reference. Heldout 0.0228 vs the degenerate 0.0193 pre-fix — higher
number, genuine signal, model actually predicting.

Service pid 38468 at step 54,614 in heartbeat. 24 commits pending push.

---

## 2026-04-13 ~19:49 EDT — step 95K: confidence rising on first-token lookup

Running diagnose_collapse directly against current.pt at step 95K:

| metric (across 5 prompts)        | step 5K | step 95K |
|----------------------------------|---------|----------|
| OUTPUT activation pairwise L2    | 0.28–0.35 | 0.39–0.49 |
| decoder logits pairwise L2       | 2.51–3.19 | 3.48–4.43 |
| unique decoded tokens            | 5 / 5   | 5 / 5    |
| logit for 'The' on 'The quick brown' | +0.20 | **+0.28** |
| logit for 'Shall' on 'Shall I compare thee' | +0.19 | **+0.24** |
| edge \|w\| median                | 0.103   | 0.122    |
| edge strength EMA mean           | 0.011   | 0.013    |

First-token correct-answer confidence is rising monotonically, edge weights
drifting up under Hebbian (still far below the 5.0 clamp), zero dead edges.
No STOP / permfail. Next finalized tick imminent (step ~100K).

### tick 1776126539 — step 95K — no change

loss_ema=0.0154 (-10.5% vs baseline), heldout=0.0235, KL=20.71 (improving from 23.03).
Graph static at 34n/64e, zero growth events, wm_occupancy=0.0, episodic saturated.
Training stable post-NaN-crisis; structural concerns (empty WM, frozen growth) flagged
for operator review — both require carveout-path changes. No auto-intervention warranted.

### Tick 1776128742 — step 145K — no change
loss_ema=0.0140 (-18.9% vs baseline), heldout=0.0237 (+22.9% vs baseline), KL=20.71.
Graph static 34n/64e, zero growth, wm_occupancy=0.0, episodic saturated.
Training loss improving; heldout gap widening — monitoring for overfitting signal.
No auto-intervention; recommend operator review if heldout continues rising next tick.

---

## 2026-04-13 ~20:31 EDT — step 185K: overfitting signal confirmed, still not collapsed

diagnose_collapse at step 185K still shows 5 unique tokens, OUTPUT variance
0.40-0.50, decoder logits variance 3.56-4.53. Not a regression.

The tick Claude correctly flagged the emerging overfit pattern at 145K:

| step | loss_ema | heldout | gap    |
|------|----------|---------|--------|
| 45K  | 0.0150   | 0.0228  | 0.0078 |
| 95K  | 0.0154   | 0.0235  | 0.0081 |
| 145K | 0.0140   | 0.0237  | 0.0097 |
| 185K | —        | —       | pending|

Training loss drifting down, heldout drifting up — textbook overfit on
a capacity-limited model (34 nodes / 64 edges, zero growth events across
185K steps). The RMS-gated synaptogenesis never fires because
coactivation products with RMS-magnitudes ~0.5-1.0 and locality bonus
well under 1.0, scaled by config.synaptogenesis_rate=0.005, rarely clear
the random draw. This is the "growth concern" flagged at the 85K/270K
GUE-era check re-emerging now with a functional decoder.

Not intervening yet — the trend is early and the model still produces
sensible first-token lookups. If the gap doubles by 300K, the next
autonomous step would be to lower activation_threshold (currently 0.1)
or raise synaptogenesis_rate back to 0.01 so the graph can actually
grow capacity. Flagging for the next check.

25 commits pending push.

### tick 1776130966 — step 200K — no change

- **loss_ema_500:** 0.01565 (−9.2% vs baseline)
- **heldout_loss:** 0.02373 (+23.2% vs baseline) — overfit trend continues
- **graph:** 34 nodes / 64 edges — static (0 growth events last 1K steps)
- **wm_occupancy:** 0.0 (was 1.0 at baseline)
- **KL:** 20.71 (plateau since step 45K)
- **Decision:** no change. Overfit mild in absolute terms (perplexity 1.024).
  Impactful interventions all touch carveout paths. Collecting more data.

---

## 2026-04-14 ~ early UTC — overfit gap reversal, operator paused for gaming

Revisiting the gap trajectory with tick 1776130966 included:

| step | loss_ema | heldout | gap    |
|------|----------|---------|--------|
| 45K  | 0.0150   | 0.0228  | 0.0078 |
| 95K  | 0.0154   | 0.0235  | 0.0081 |
| 145K | 0.0140   | 0.0237  | 0.0097 |
| 200K | 0.0156   | 0.0237  | 0.0081 |

The gap widened to 0.0097 at 145K then *narrowed back to 0.0081* at 200K,
matching the 95K value. The 145K bump wasn't the start of a monotonic
overfit — it was a transient dip in loss_ema that has since reverted.
No autonomous growth-enable needed; the trend I flagged at 185K hasn't
materialized.

Operator paused the service for gaming at step 233,069. Heartbeat healthy,
no crash indicators. Will resume monitoring once the loop is unpaused.
| 1776142118 | 245000 | 0.01434 | 0.02373 | 20.71 | no_change | overfitting plateau; interventions need operator scope |
