# Direction 1 Phase 1 — multi-seed validation FAILED

**Date:** 2026-04-19, commit 685de25 (source), b9h6hgeod (run)
**Schedule:** v0.5 capacity (8 regimes × 500 steps)
**Seeds:** {0, 1, 42}

## Headline

**Direction 1's Phase 1 result does not reproduce.** The single-seed
(seed=42) finding of 7/8 regimes improvement for `synap_only_pe` over
`synap_only` is not replicated on seeds 0 or 1. Across all 3 seeds,
the mean per-regime effect is **+0.0002 MSE** (supervision slightly
hurts on average).

## Per-seed scorecard

Wins at noise floor 0.00005 MSE:

| | seed=0 | seed=1 | seed=42 |
|---|--------|--------|---------|
| synap_only_pe beats synap_only | 1/8 | 1/8 | 8/8 |
| full_pe beats full | 4/8 | 2/8 | 3/8 |

## Per-regime mean MSE (mean ± stddev across 3 seeds)

| Regime    | no_growth     | synap_only    | synap_only_pe | full          | full_pe       |
|-----------|---------------|---------------|---------------|---------------|---------------|
| mlp_2x16  | 0.0075±0.0010 | 0.0081±0.0011 | 0.0074±0.0012 | 0.0075±0.0007 | 0.0079±0.0012 |
| mlp_2x32  | 0.0008±0.0003 | 0.0011±0.0004 | 0.0013±0.0005 | 0.0012±0.0002 | 0.0013±0.0005 |
| mlp_3x16  | 0.0007±0.0001 | 0.0011±0.0002 | 0.0015±0.0005 | 0.0015±0.0002 | 0.0016±0.0004 |
| mlp_3x32  | 0.0006±0.0001 | 0.0015±0.0003 | 0.0017±0.0006 | 0.0016±0.0005 | 0.0021±0.0006 |
| mlp_2x64  | 0.0014±0.0002 | 0.0036±0.0008 | 0.0042±0.0013 | 0.0042±0.0009 | 0.0035±0.0002 |
| mlp_3x64  | 0.0020±0.0005 | 0.0053±0.0008 | 0.0060±0.0018 | 0.0052±0.0004 | 0.0051±0.0007 |
| mlp_4x32  | 0.0018±0.0002 | 0.0043±0.0006 | 0.0047±0.0006 | 0.0041±0.0004 | 0.0042±0.0006 |
| mlp_4x64  | 0.0011±0.0002 | 0.0054±0.0006 | 0.0057±0.0005 | 0.0068±0.0007 | 0.0055±0.0014 |

Observations:
- **Synap_only_pe stddev is much larger than the mean effect.** On mlp_3x64,
  synap_only_pe stddev is ±0.0018 but the mean delta vs synap_only is
  +0.0007. Effect is completely dominated by seed variance.
- **no_growth is the most consistent low-stddev variant.** Expected —
  no growth means no stochastic admission.
- **full_pe on mlp_4x64 has the largest effect**: −0.0013 (helps). This
  is one of the two regimes where full_pe wins consistently across
  seeds 0 and 1 (though seed=42 has +0.0004).

## Interpretation

Direction 1's supervision mechanism ships a global PE-trend gate
(since the per-pair EMA collapses degenerately on this graph). The
gate pauses synap admission when the running ratio of recent-vs-
baseline error is high. What we saw on seed=42 was a lucky
alignment between this pause and a regime-transition pattern that
happened to coincide with bad admissions. On seeds 0 and 1, the same
mechanism catches different moments — sometimes useful, often not —
and the average effect is essentially zero or slightly negative.

This is a **negative result**. The mechanism as implemented does NOT
provide a reliable per-seed improvement. The plan doc's success
criterion (`synap_only_pe beats synap_only on 6+/8 regimes`) was
met on only 1 of 3 seeds, which is within what we'd expect from
random noise for a 50/50-ish mechanism.

## Implementation note

Code stays in-tree as opt-in (`synaptogenesis_supervision="pe_conditional"`).
Default remains `"none"`. The test suite (40 supervision + waiver tests)
continues to enforce the behavior is correct per the spec; the spec
just turns out not to help reliably.

## Implications for Direction 3

Direction 3 (plasticity broadcast) explicitly globalizes the same
signal Phase 1 used degenerately, AND extends the gating to Hebbian
learning (not just synap admission). The multi-seed negative here
weakens prior confidence that any PE-ratio-based global gate will
help. But Direction 3 has two genuinely different properties:

1. It gates Hebbian LR proportionally (scale-by-gain, not binary
   pause), so small surprise changes produce small gating effects
   rather than all-or-nothing.
2. It affects Hebbian weights, which are the primary "learning"
   mechanism. Direction 1 only affected "structure" (new edges).
   If the real SOMA learning happens through Hebbian not through
   synap admission, gating Hebbian may have teeth where gating
   synap did not.

Direction 3 should be multi-seeded from the start to avoid repeating
this seed-42-outlier mistake.

## Decision

- **Direction 1**: negative result, logged. Mechanism stays opt-in.
- **Direction 3**: test with multi-seed from day 1. No single-seed
  "looks promising" spiral.
- **Direction 2 and 4**: unchanged priority, but reorder so Direction 2
  (learnable projections, attacks the root cause directly) comes
  before Direction 4 (LLM-as-teacher, much bigger lift).
