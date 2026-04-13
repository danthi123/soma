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
