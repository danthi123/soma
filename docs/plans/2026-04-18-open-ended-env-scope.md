# Open-Ended Learning Environment — Scope

**Date:** 2026-04-18
**Parent:** [2026-04-18-developmental-findings-and-direction.md](2026-04-18-developmental-findings-and-direction.md)
**Status:** Scoping (pre-implementation)

---

## Objective

Build a minimal environment that exercises SOMA's native mechanisms
(neurogenesis, consolidation, structural plasticity, critical periods)
on a task where adaptation speed — not static retrieval accuracy — is
the rate-limiting metric.

## Constraints (from the broader roadmap)

- No LLM in the loop (the "virtual baby" needs a mouth, but not to
  test core mechanisms; retain verbalizer as optional add-on)
- Lives in the existing SOMA codebase (`src/soma/`) rather than as
  a separate repo — defer the repo split until the env is working
- Runs on the 3090 (not a distributed setup)
- Uses the existing execution engine, memory tiers, consolidation,
  and growth modules. Does NOT introduce a new substrate.

## Minimal Viable Environment (MVE)

The smallest env that tests the hypothesis "structural plasticity
measurably improves adaptation speed after distribution shift."

### v0: Synthetic sequence prediction (this week)

**Not a grid world, not an agent, no reward.** Pure next-input
prediction on an evolving 2D sequence.

**Observation stream**:
- Generate (x, y) coordinates from one of N "regimes":
  - Regime 1: random uniform in [0,1]²
  - Regime 2: periodic (e.g. x = sin(t), y = cos(t))
  - Regime 3: linear drift
  - Regime 4: two clusters switching
- At regime boundaries (every K steps), regime switches without warning

**SOMA's task**: predict next (x, y) from the stream; learning drives
the graph via prediction error.

**Metrics**:
- Prediction error over time (running window)
- Steps-to-adaptation after each regime change (time until error
  returns to within 20% of pre-switch baseline)
- Graph growth (node/edge count) aligned to regime changes
- Consolidation effect: compare with/without consolidation cycles
  between regimes

**Expected signal** (the thing that would falsify the "retrieval
ceiling" framing):
- SOMA's adaptation speed should exceed a static MLP trained on
  Regime 1 only (degenerate case) and a static MLP trained on all
  regimes simultaneously (capacity-matched control). The brain-
  inspired mechanisms should specifically help on the *boundary*
  dynamics — the response right after a regime switch.

### v1: Simple grid world (2-4 weeks from now)

Once v0 validates the plumbing and shows adaptation dynamics:

- 8×8 grid
- Agent, food items, simple obstacles
- Actions: 4-directional move + stay
- Observation: 3×3 local window + hunger state
- Reward: find food, avoid starvation
- Adaptation events: food distribution shifts, new obstacle types

v1 adds:
- Action selection (simple policy head trained from a TD signal)
- Episodic structure (episodes end on death)
- Richer observations

### v2+: Further scope (TBD)

Only consider after v0 and v1 land.

## Implementation plan (v0, this session)

### File structure
```
src/soma/environments/
    __init__.py
    sequence_env.py       # v0 environment
    regime_schedule.py    # regime generators
research/developmental/
    env_sequence_v0.py   # runner, metrics, comparison baseline
tests/test_environments/
    test_sequence_env.py  # basic env behavior
```

### Components

1. **`RegimeSchedule`** — produces a stream of (x, y) tuples, marking
   regime boundaries for post-hoc analysis.
2. **`SequenceEnv`** — wraps `RegimeSchedule` with a step() interface
   that returns the current observation. Tracks `current_step` and
   `current_regime`.
3. **`run_sequence_env`** — drives SOMA through K=5000 steps across
   4 regime changes; logs prediction error per step, graph size per
   regime, consolidation events, adaptation windows.
4. **`BaselineMLP`** — same input/output dims as SOMA's
   PredictiveSOMA; trained via Adam. Two variants: "frozen on
   regime 1" (demonstrates that adaptation matters) and "trained
   online" (fair comparison).

### Success criteria for v0

- SOMA runs without crash on 5K-step sequence
- Prediction error visibly drops within each regime
- Prediction error spikes at regime boundaries, then recovers
- Adaptation window measurable and reportable

### Stretch for v0

- With/without consolidation comparison (toggle between the two)
- Growth vs no-growth comparison (freeze neurogenesis/pruning)
- Curiosity module effect (if critical period timing is set right)

## What this does NOT do

- Language. The verbalizer + LLM interface stay off the critical path.
- Full agent behavior (action selection, reward). v0 is prediction-only.
- New architecture. No new node types, no new memory tiers. Just uses
  what's built.
- Benchmarks against other developmental systems (CLS, ACT-R). Too
  much lift for today.

## Decision point

If v0 shows clear adaptation dynamics (prediction error recovers
after boundaries; consolidation helps; graph growth correlates with
novelty), we proceed to v1 and start building the actual grid-world
agent. If v0 is flat — no measurable adaptation benefit over a
capacity-matched static MLP — we stop and reconsider the env
approach.

## Open questions for you (non-blocking)

1. **Paper release timing**: should the arXiv preprint of the
   ceiling paper go up before or after v0 runs? (I assume after, so
   we can include "here's what the mechanisms do on the tasks
   they're actually for" as a final section.)
2. **Repo split**: the direction review proposed moving
   developmental/ to its own repo. This env will extend
   developmental/ significantly. Split now or after v0?
3. **Verbalizer fate**: the LLM verbalizer is on shelf. Keep it
   in-tree for eventual re-integration, or move to a separate
   branch?

None of these block v0 build-out — I'll proceed with v0 in-tree
under the existing layout unless you redirect.
