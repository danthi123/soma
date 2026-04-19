# Position-scramble multi-seed — **SEED-DEPENDENT**; specific positions usually matter

**Date:** 2026-04-19
**Commit:** bf37133 (runner), b6hr5vlv0 (run)
**Data:** `research/developmental/results/env_sequence_v05_synap_position_scramble_multiseed.json`
**Seeds:** {0, 1, 42}, 5 variants

## Headline

Replacing node positions with a fresh random draw immediately after
construction (preserving distance distribution, breaking any
correlation with other node attributes) **destroys the locality
benefit on 2 of 3 seeds**:

| Contrast | seed=0 | seed=1 | seed=42 |
|----------|--------|--------|---------|
| local vs synap_only (reference positive)   | 6/8 | 8/8 | 8/8 |
| scramble vs synap_only                      | 0/8 | 7/8 | 8/8 |
| **scramble vs local (does scramble match?)** | **0/8** | **0/8** | 5/8 |

On seeds 0 and 1, scramble produces significantly worse MSE than
local (+0.003 to +0.007 on hard regimes). On seed 42 alone, scramble
approximately matches or slightly beats local.

**Interpretation**: specific initial positions carry signal beyond
"any coherent distance metric." Scrambling positions largely (not
always) destroys this signal. Likely mechanism: positions and
projection weights share the same rng state at node construction
— nearby original positions → similar projection weights → similar
activations → edges between them are genuinely useful. Scramble
breaks this correlation.

## Per-regime MSE (mean across 3 seeds)

| Regime    | no_growth | synap_only | synap_only_local | **scramble** | neuro_only |
|-----------|-----------|------------|------------------|--------------|------------|
| mlp_2x16  | 0.0075    | 0.0075     | 0.0073           | 0.0078       | 0.0075     |
| mlp_2x32  | 0.0008    | 0.0009     | 0.0007           | 0.0011       | 0.0008     |
| mlp_3x16  | 0.0007    | 0.0017     | **0.0002**       | 0.0010       | 0.0006     |
| mlp_3x32  | 0.0006    | 0.0018     | **0.0002**       | 0.0012       | 0.0004     |
| mlp_2x64  | 0.0014    | 0.0042     | **0.0005**       | 0.0029       | 0.0007     |
| mlp_3x64  | 0.0020    | 0.0063     | **0.0009**       | 0.0046       | 0.0009     |
| mlp_4x32  | 0.0018    | 0.0049     | **0.0010**       | 0.0036       | 0.0011     |
| mlp_4x64  | 0.0011    | 0.0061     | **0.0001**       | 0.0038       | 0.0002     |

Scramble is consistently **between** synap_only and synap_only_local:
the coherent-distance-metric effect exists (scramble beats synap_only
on hard regimes, 15/24 seed-regime combinations), but it's much weaker
than the specific-position effect (local beats scramble 11/24 seed-
regime combinations).

## Why seed 42 is different

seed=42's pattern is anomalous: scramble approximately matches local
(5/8 wins on seed=42, versus 0/8 on seeds 0, 1). Looking at raw
numbers:

- seed=42, local:     52 synap admissions, MSE 0.0012
- seed=42, scramble:  38 synap admissions, MSE 0.0011

On seed=42, scramble admits FEWER edges but achieves slightly lower
MSE. Possible explanations:
- Seed=42's scrambled-position distribution happened to pick a
  particularly favorable subset (no test-retest to verify).
- Seed=42's original-position-to-projection correlation was weaker
  than seeds 0, 1, so scrambling was less destructive.
- 3-seed sample is too small to distinguish seed-specific variation
  from mechanism differences.

Extending to more seeds (say {0, 1, 2, 3, 4, 42, 100, 1024}) would
clarify whether seed=42 is an outlier or reveals genuine variability.

## What this means for the mechanism

Combining position-scramble with the prior cumulative evidence:

| Experiment | Tests | Result |
|------------|-------|--------|
| synap_local vs synap_only | does locality help? | YES (8/7/8) |
| cap1/cap2 vs local | does temporal redistribution suffice? | NO |
| rate063/rate030 vs local | does rate reduction suffice? | NO |
| cutoff sweep | is there a spatial aperture? | YES, 0.5 is peak |
| **scramble vs local** | **do SPECIFIC positions matter?** | **mostly YES (0/0/5 wins)** |

The mechanism now reads as:
1. Random input projections generate arbitrary but deterministic
   node response patterns.
2. Nodes created in rng sequence have their positions AND their
   projections drawn from the same rng state. Small-random positions
   correlated with small-random projection weights.
3. The locality filter (cutoff 0.5) admits pairs with nearby
   positions, which (via the shared-rng correlation) also have
   similar projection-induced response patterns.
4. Edges between response-similar nodes are useful — they connect
   units that already agree on input patterns, so their
   hebbian-updated weights converge to meaningful consensus rather
   than spurious noise.
5. Scrambling positions breaks step 3's correlation. The filter
   then admits pairs with unrelated projection responses — edges
   aren't systematically useful (seeds 0, 1).
6. On some seeds (like 42), the scrambled distribution happens to
   preserve enough structure that the effect is weaker.

## Research implications

This finding is important because it constrains the mechanism:
- It's NOT just "any spatial coherence" — scramble preserves
  spatial coherence but loses most of the benefit.
- It IS about positions that align with some other initialization
  signal (most likely projection weights from shared rng).

This has testable predictions:
1. If we EXPLICITLY correlate positions with projection init (e.g.,
   position = PCA-1 of projection matrix), the locality effect
   should be AS STRONG OR STRONGER than the default randn positions.
2. If we use DIFFERENT rng streams for position and projection init
   (eliminating shared-rng correlation entirely), the default
   locality effect should weaken.

Both are doable with small config changes and would clarify whether
the shared-rng hypothesis is the actual mechanism.

## Product implications

Minimal: the `memory_layer()` preset's default (locality=0.5 with
default position initialization) is safe regardless. The product
story isn't affected by whether the mechanism is "spatial" or
"shared-rng-correlated-with-projection" — the filter still works
on default configs.

But: if future deployments use a DIFFERENT position initialization
(e.g., embedding-based positions for semantic locality), the
"specific positions matter" finding suggests we should expect
different behavior and re-tune the cutoff.

## Files

- `research/developmental/env_sequence_v05_synap_position_scramble_multiseed.py`
- `research/developmental/results/env_sequence_v05_synap_position_scramble_multiseed.json`
- `research/developmental/results/env_sequence_v05_synap_position_scramble_multiseed.log`
