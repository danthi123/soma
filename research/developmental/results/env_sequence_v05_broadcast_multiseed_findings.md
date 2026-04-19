# Direction 3 (plasticity broadcast) multi-seed validation — NEGATIVE

**Date:** 2026-04-19
**Commits:** 27d50ba (impl), 685de25 (clamp fix), 894695d (runner)
**Experiment:** `research/developmental/env_sequence_v05_broadcast_multiseed.py`
**Data:** `research/developmental/results/env_sequence_v05_broadcast_multiseed.json`
**Seeds:** {0, 1, 42}, 7 variants

## Headline

**Direction 3 does not meet its success criterion.** Plan doc
required `full_bcast` to beat `full` on 5+/8 regimes. Per-seed
results:

- seed=0: 5/8
- seed=1: 3/8
- seed=42: 0/8

Only seed=0 meets the criterion; seeds 1 and 42 fail it. This
single-seed selectivity mirrors Direction 1's seed=42-outlier
failure — except Direction 3 is outlier in the opposite direction
(seed=42 makes broadcast LOSE).

Like Direction 1, the mechanism shows inconsistent per-seed
behavior that averages to no reliable signal.

## But a SURPRISING positive baseline discovery

Multi-seed confirmed what earlier 2x2 results hinted at: **`neuro_only`
(neurogenesis without synaptogenesis) is the best-performing variant
across all 3 seeds on the hard regimes.**

| Regime    | no_growth     | neuro_only    | delta     |
|-----------|---------------|---------------|-----------|
| mlp_2x16  | 0.0075±0.0010 | 0.0075±0.0010 | tie       |
| mlp_2x32  | 0.0008±0.0003 | 0.0008±0.0003 | tie       |
| mlp_3x16  | 0.0007±0.0001 | 0.0006±0.0002 | -0.0001   |
| mlp_3x32  | 0.0006±0.0001 | 0.0004±0.0002 | -0.0002   |
| mlp_2x64  | 0.0014±0.0002 | 0.0008±0.0004 | **-0.0006** |
| mlp_3x64  | 0.0020±0.0005 | 0.0011±0.0005 | **-0.0009** |
| mlp_4x32  | 0.0018±0.0002 | 0.0011±0.0001 | **-0.0007** |
| mlp_4x64  | 0.0011±0.0002 | 0.0006±0.0004 | **-0.0005** |

On the four highest-capacity regimes, `neuro_only` is 30-60% better
than `no_growth`. This is a real, multi-seed-validated effect. The
added capacity from neurogenesis is providing useful structure.

## Broadcast per-sub-treatment

**synap_only_bcast vs synap_only** (per-seed wins):
- seed=0: 7/8, seed=1: 0/8, seed=42: 8/8
- Mean deltas across seeds: mostly negative (broadcast helps on avg)
  but seed=1 is strongly positive (broadcast hurts).
- Broadcast gain range during runs: [0.42, 1.62], so the gain is
  actually firing over a meaningful range.

Interpretation: like Direction 1, the global PE-ratio signal happens
to align well with some seeds' plasticity-damage pattern but not
others. The mechanism catches genuine surprise events but the
per-seed effect depends on whether those surprise events coincide
with actually-harmful admissions.

**neuro_only_bcast vs neuro_only** (per-seed wins):
- seed=0: 0/8, seed=1: 3/8, seed=42: 3/8
- Mean deltas near zero across all regimes.

Interpretation: the broadcast fires (gain range [0.44, 1.57]) but
neurogenesis is insensitive to it. The gate-ease on synapse admission
doesn't matter when there's no synaptogenesis to gate, and Hebbian
LR scaling has no visible effect here — probably because neuro_only's
edges are newly-created with low weights that Hebbian rarely moves
meaningfully over 4000 steps anyway.

**full_bcast vs full** (per-seed wins):
- seed=0: 5/8, seed=1: 3/8, seed=42: 0/8
- Mean deltas: mixed, near zero.

Interpretation: full is a noisy regime (high-capacity + both growth
mechanisms + their interactions). Broadcast's effect gets swamped.

## Verdict

**Direction 3 is NEGATIVE on its primary claim.** Rejected.

**Code disposition**: broadcast stays in-tree as opt-in
(`plasticity_broadcast_mode="pe_scaled"`). Default is `"off"`.
22-test suite enforces correct semantics.

## Real signal to pursue

**Focus on neuro_only.** It's the one multi-seed-validated positive
mechanism on v0.5. The path forward is to understand *why* it works
(positional-neighbor wiring + capacity matching?) and either:

1. **Keep synap turned off.** If neuro_only works and synap doesn't,
   maybe synap is the wrong primitive for this substrate and should
   be ablated. Production config already does this
   (`SOMAConfig.production()` → `synaptogenesis_interval=0`).

2. **Fix synap so it composes.** The 2x2 and this experiment both
   show synap hurts, but `full` (synap+neuro) is only MILDLY worse
   than `neuro_only` on some regimes. Perhaps a truly task-supervised
   synap (Direction 2 + Direction 1 combined) could admit only
   edges that improve prediction — but we'd need to solve the
   per-pair signal problem first (Direction 1 failed on exactly this).

3. **Attack the root cause** (Direction 2: learnable projections).
   If the problem is that activations are semantically arbitrary,
   making projections learnable via prediction-loss gradient is the
   fundamental fix. This is the next direction to pursue.

## Implication for Direction 2 & 4

Direction 2 is now the obvious next step. No_diversify ablation
(launched right after this finding — b3vw4hpqh) tests whether the
current random-projection diversification is even carrying its
weight. Direction 2's full implementation then targets replacing
the random matrices with learnable ones that actually encode task
signal.

Direction 4 (LLM-as-teacher) remains gated on a text task (v0.5's
synthetic tensor inputs don't meaningfully embed via an LLM). That
direction is more natural when we pick up the retrieval benchmark.
