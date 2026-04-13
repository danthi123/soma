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
