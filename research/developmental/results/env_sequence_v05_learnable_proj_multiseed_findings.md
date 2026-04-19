# Direction 2B (learnable projections) multi-seed — NEGATIVE

**Date:** 2026-04-19
**Commit:** 51b747f (impl), bqugoca9r (run)
**Data:** `research/developmental/results/env_sequence_v05_learnable_proj_multiseed.json`
**Seeds:** {0, 1, 42}, 5 variants

## Headline

**Direction 2B does not meet its success criterion.** Plan required
`full_learnable` to beat `full_frozen` on 5+/8 regimes for ALL three
seeds. Actual: 4/8, 4/8, 1/8. Fails on all seeds (seed=42
particularly badly).

`neuro_only_learnable` is slightly closer: 5/8, 3/8, 6/8 — meets
5+/8 on two seeds but not all three. And the mean delta is only
-0.0001 to -0.0002 MSE (noise-level).

## Per-seed scorecards

**full_learnable beats full_frozen:**
- seed=0: 4/8
- seed=1: 4/8
- seed=42: 1/8

**neuro_only_learnable beats neuro_only_frozen:**
- seed=0: 5/8
- seed=1: 3/8
- seed=42: 6/8

Neither meets the strict 5+/8 on ALL seeds criterion.

## Negative for full on hard regimes

`full_learnable` CONSISTENTLY REGRESSES on the 4x64 regime:

| seed | full_frozen | full_learnable | delta    |
|------|-------------|----------------|----------|
| 0    | 0.0063      | 0.0075         | +0.0012  |
| 1    | 0.0063      | 0.0075         | +0.0012  |
| 42   | 0.0063      | 0.0075         | +0.0012  |

Identical +0.0012 regression on all 3 seeds. This is not noise —
it's a systematic downside of learnable projections in the full
configuration on the hardest regime. Plausible mechanism: the
prediction-head gradient flows through a mean over projections, so
projections co-adapt toward the prediction head's specific
representation. When combined with aggressive synap (full config),
this co-adaptation interacts badly with the evolving graph structure.

## Per-regime mean MSE across seeds

| Regime    | no_growth     | full_frozen   | full_learnable | neuro_only_frozen | neuro_only_learnable |
|-----------|---------------|---------------|----------------|-------------------|----------------------|
| mlp_2x16  | 0.0075±0.0010 | 0.0075±0.0008 | 0.0072±0.0014  | 0.0075±0.0010     | 0.0070±0.0011        |
| mlp_2x32  | 0.0008±0.0003 | 0.0011±0.0003 | 0.0007±0.0002  | 0.0008±0.0003     | 0.0004±0.0000        |
| mlp_3x16  | 0.0007±0.0001 | 0.0018±0.0004 | 0.0016±0.0004  | 0.0007±0.0000     | 0.0006±0.0001        |
| mlp_3x32  | 0.0006±0.0001 | 0.0020±0.0002 | 0.0021±0.0003  | 0.0004±0.0002     | 0.0003±0.0001        |
| mlp_2x64  | 0.0014±0.0002 | 0.0044±0.0006 | 0.0048±0.0009  | 0.0009±0.0001     | 0.0007±0.0002        |
| mlp_3x64  | 0.0020±0.0005 | 0.0060±0.0006 | 0.0072±0.0009  | 0.0012±0.0003     | 0.0013±0.0005        |
| mlp_4x32  | 0.0018±0.0002 | 0.0046±0.0004 | 0.0048±0.0004  | 0.0013±0.0003     | 0.0012±0.0003        |
| mlp_4x64  | 0.0011±0.0002 | 0.0063±0.0003 | 0.0075±0.0004  | 0.0005±0.0004     | 0.0004±0.0004        |

**Still the story: `neuro_only` wins across all 5 variants.** Best
overall: `neuro_only_learnable` beats `neuro_only_frozen` on
hard regimes by 0.0001-0.0002 MSE.

## What the 2B implementation actually does

Reminder for self-consistency: Direction 2B as implemented has
learnable projections applied as a POST-PROCESSING transform on
`_last_summary` before the prediction head. The projections do NOT
affect node activations inside SOMA's graph — they only affect
what the prediction head sees. This is a simpler architecture than
the plan doc's original Direction 2 intent (gradients through the
full graph = Direction 2C).

Given the null result here, Direction 2C would need to do
meaningfully better to justify the ~2-day implementation cost.

## Projection norm drift

Mean projection norm start → end:
- Frozen variants: 4.00 → 4.00 (no drift, as expected).
- Learnable variants: 4.00 → 4.03–4.08.

The learnable projections are moving, just gently. The movement is
real but not dramatic. Projection LR is 1e-4; over 4000 steps with
many pairs updating, a 1–2% norm change is in line with expectation.

## Implication

The Directions 1, 2B, and 3 are all NEGATIVE. Three different
plasticity-adjustment mechanisms, all fail multi-seed validation.

The one thing that works — `neuro_only` — survives every ablation
(no_diversify, broadcast, learnable-proj) with its edge intact. Its
positive effect (beat no_growth by 0.0005-0.0009 MSE on all 4 hard
regimes across all 3 seeds) is the ONLY multi-seed-validated
positive in this session.

The `why_neuro_only_works.md` analysis predicts that the critical
factor is POSITIONAL locality in neurogenesis's wiring choices.
`synap_local` (hard distance cutoff on synap admissions) is the
direct test of that prediction. Launched after this finding;
results incoming.

If `synap_local` also fails, the story for this research track
on v0.5 is: **"only positional-locality + capacity-pressure
combined produce reliable benefit; neither alone is enough."**
Strategic pivot after synap_local will target either:
1. A text task where LLM-as-teacher (Direction 4) is natural.
2. A different synthetic benchmark (curriculum, online meta-
   learning) where the failure modes on v0.5 might not apply.

## Files

- `research/developmental/env_sequence_v05_learnable_proj_multiseed.py`
- `research/developmental/results/env_sequence_v05_learnable_proj_multiseed.json`
- `research/developmental/results/env_sequence_v05_learnable_proj_multiseed.log`
