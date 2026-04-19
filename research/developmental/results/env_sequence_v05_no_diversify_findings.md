# no_diversify ablation — findings

**Date:** 2026-04-19
**Commit:** 941a263 (runner), analysis run ca9f4ae
**Data:** `research/developmental/results/env_sequence_v05_no_diversify.json`
**Seeds:** {0, 1, 42}, 5 variants.

## Headline

**`_diversify_activations` is essentially a no-op for `neuro_only`**
(mean deltas ±0.0002 across all regimes and seeds) and produces
**highly variable, inconsistent effects on `full`** (seed=42 wins
6/8, seed=0 wins 0/8).

This weakens Direction 2's core hypothesis that "learnable
projections fix the random-projection problem" — on the one
mechanism that multi-seed-validates (`neuro_only`), the projections
don't carry signal either way.

## Per-seed scorecards

**full_nodiv beats full:**
- seed=0: 0/8
- seed=1: 2/8
- seed=42: 6/8

Zero consistency across seeds. Diversification variability doesn't
reliably help OR hurt.

**neuro_only_nodiv beats neuro_only:**
- seed=0: 2/8, seed=1: 2/8, seed=42: 1/8

Near-random; diversification is neutral on neuro_only.

## Per-regime mean MSE across seeds

| Regime    | no_growth     | neuro_only    | neuro_only_nodiv | full          | full_nodiv    |
|-----------|---------------|---------------|------------------|---------------|---------------|
| mlp_2x16  | 0.0075±0.0010 | 0.0075±0.0010 | 0.0076±0.0010    | 0.0074±0.0007 | 0.0081±0.0009 |
| mlp_2x32  | 0.0008±0.0003 | 0.0008±0.0003 | 0.0008±0.0003    | 0.0011±0.0003 | 0.0012±0.0001 |
| mlp_3x16  | 0.0007±0.0001 | 0.0008±0.0000 | 0.0005±0.0001    | 0.0014±0.0004 | 0.0016±0.0002 |
| mlp_3x32  | 0.0006±0.0001 | 0.0003±0.0002 | 0.0005±0.0002    | 0.0017±0.0004 | 0.0020±0.0002 |
| mlp_2x64  | 0.0014±0.0002 | 0.0006±0.0003 | 0.0006±0.0002    | 0.0041±0.0006 | 0.0042±0.0006 |
| mlp_3x64  | 0.0020±0.0005 | 0.0010±0.0005 | 0.0010±0.0005    | 0.0061±0.0015 | 0.0056±0.0010 |
| mlp_4x32  | 0.0018±0.0002 | 0.0011±0.0000 | 0.0011±0.0001    | 0.0048±0.0004 | 0.0044±0.0002 |
| mlp_4x64  | 0.0011±0.0002 | 0.0002±0.0002 | 0.0004±0.0001    | 0.0054±0.0012 | 0.0061±0.0003 |

**Consistent pattern**: neuro_only ≤ no_growth across all hard
regimes regardless of diversification. Diversification is not what
makes neuro_only work.

## What IS making neuro_only work?

Process of elimination:
- Not diversification (this experiment).
- Not Hebbian updates (neuro_only has few strong edges; most edges
  stay near their init weight).
- Not lateral inhibition alone (that's on for all variants).
- Not synaptogenesis (it's OFF in neuro_only).

Remaining candidates:
1. **Positional-neighbor wiring from neurogenesis.** When a new
   node is born via neurogenesis, it wires to positional neighbors
   (see `src/soma/growth/neurogenesis.py`). These wirings create
   LOCAL structural patterns that the random-projection baseline
   doesn't have.
2. **Capacity matching.** The hard regimes (mlp_3x64+) are MLP
   teachers with more hidden-layer capacity than the 14-associator
   SOMA can fit. Adding new nodes (neurogenesis) brings SOMA's
   capacity closer to the teacher's, improving fit.
3. **Random-functional diversity.** New neurogenesis nodes come
   with freshly-initialized MLPs, adding functional diversity
   independent of projections. This is what the 2x2 run suggested
   earlier — see `project_env_findings.md`.

Most likely: a combination of (1) and (2). The positional-neighbor
wiring provides useful LOCAL correlations, and capacity adds the
bandwidth to exploit them.

## Implication for Direction 2

Direction 2's hypothesis ("random projections are the root cause
of structural-without-semantic failure") is partially refuted by
this experiment. On neuro_only, the random projections are not
the source of failure — they're nearly irrelevant.

However, a subtler interpretation survives: projections matter
ONLY when combined with synaptogenesis (because synap admits
co-active pairs, and the projections shape which pairs are
co-active). That would explain why `full_nodiv` shows more
variance than `neuro_only_nodiv` — the projections feed into
synap's admission decisions, and synap is the mechanism that
depends on them most.

Under this interpretation, Direction 2's real goal should be:
**fix synap by making the projections that drive co-activation
trainable, so synap admits meaningful pairs instead of arbitrary
ones.** That's more aligned with the plan-doc's original vision.

My current Direction 2B implementation doesn't do this — it
treats projections as a post-processing layer on the summary, not
a pre-processing layer on node inputs. Direction 2B's outcome will
tell us whether even that cruder integration carries signal; if
not, a 2C-style implementation (projections feeding node input,
gradients through the graph) would be the next step.

## Decision

1. **Let Direction 2B multi-seed finish** (already launched).
   Expected: neutral or small variable effect, similar to no_diversify.
2. **If Direction 2B is null**: the "per-pair PE signal" class of
   mechanisms (Directions 1, 3) AND the "summary-level learnable
   transform" (Direction 2B post-processing) both fail on v0.5.
   The fundamental issue is that synap on v0.5 is broken
   regardless of post-hoc gating.
3. **Strategic pivot options**:
   - Investigate WHY neuro_only works (characterize the positive
     finding; potentially publishable).
   - Accept that v0.5 isn't the right substrate for plasticity
     research; move to a text task where Direction 4 (LLM-as-teacher)
     is natural.
   - Try Direction 2C (gradient through the graph) as a more
     faithful implementation of the plan-doc Direction 2.

## Files

- `research/developmental/env_sequence_v05_no_diversify.py` (runner)
- `research/developmental/results/env_sequence_v05_no_diversify.json` (raw)
- `research/developmental/results/env_sequence_v05_no_diversify.log` (text)
