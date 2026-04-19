# synap sparsity-control multi-seed — **PARTIAL** (implementation subtlety; locality still wins)

**Date:** 2026-04-19
**Commit:** aa7b21d (impl), 0df1f63 (runner), bowda7azy (run)
**Data:** `research/developmental/results/env_sequence_v05_synap_sparsity_control_multiseed.json`
**Seeds:** {0, 1, 42}, 6 variants

## Headline

The random-K admission cap **does not reduce total admissions** — it
spreads the same admissions across more fires (cap=1 converts bursts
of 5-6 admissions into 1-per-fire over many steps). So this is NOT a
clean sparsity-vs-locality control.

**But the result is still informative.** At MATCHED TOTAL ADMISSIONS
(158 edges for cap1, cap2, and synap_only), `synap_only_local` still
beats both cap variants on 7-8 of 8 regimes across all seeds. The
locality filter's benefit cannot be explained by temporal spreading
of admissions alone.

Per-seed scorecards (treatment beats base, noise floor 0.00005):

| Contrast | seed=0 | seed=1 | seed=42 |
|----------|--------|--------|---------|
| cap1 vs synap_only       | 5/8 | 6/8 | 0/8 |
| cap2 vs synap_only       | 7/8 | 8/8 | 1/8 |
| synap_only_local vs synap_only      | 7/8 | 8/8 | 8/8 |
| synap_only_local vs cap1 | 7/8 | 7/8 | 8/8 |
| synap_only_local vs cap2 | 5/8 | 7/8 | 8/8 |

## Why the cap doesn't reduce total admissions

The cap is applied per call to `synaptogenesis()` (every
`synaptogenesis_interval` steps, i.e., once per 10 steps). Trimming
5-6 admissions down to 1 only prevents THOSE particular pairs from
wiring THIS TIME — they remain eligible candidates for future calls
because `has_edge()` still returns False.

Uncapped `synap_only` saturates its eligible pool in the first ~760
steps (76 fires × ~2 admissions each), then goes quiet because most
co-active pairs already have edges. Capped variants deplete their
eligible pool slower because only 1-2 pairs get admitted per fire,
leaving the other pairs available for later.

Per-fire admission distribution (seed=0):

| Variant | Total | Fires | Distribution |
|---------|-------|-------|--------------|
| synap_only       | 158 | 76  | {1:40, 2:14, 3:7, 4:8, 5:5, 6:2} |
| synap_only_cap1  | 158 | 158 | {1:158} |
| synap_only_cap2  | 158 | 95  | {1:32, 2:63} |
| synap_only_local | 32  | 27  | {1:23, 2:3, 3:1} |

Both cap variants converge to the same total (158) as uncapped, just
with different temporal profiles. **This was not the test I intended.**

## What the results DO show

### Temporal spreading alone ≈ no-op (cap=1 vs synap_only)

Mean MSE (seed=0):
- synap_only: 0.0044
- synap_only_cap1: 0.0044 (identical within noise)

Spreading the same admissions across more fires doesn't change the
final MSE. The *graph structure* at steady-state is what matters, not
the temporal path taken to reach it. cap=1 is the most extreme
spreading (1 per fire for 158 fires); if this had any effect, it
would show here.

### cap=2 provides partial benefit (matches neuro_only on hard regimes)

Per-regime MSE across seeds (mean):

| Regime    | synap_only | cap1  | cap2  | local | neuro |
|-----------|-----------|-------|-------|-------|-------|
| mlp_2x64  | 0.0035    | 0.0040| **0.0020** | 0.0008 | 0.0010 |
| mlp_3x64  | 0.0055    | 0.0058| **0.0035** | 0.0012 | 0.0011 |
| mlp_4x32  | 0.0045    | 0.0046| **0.0030** | 0.0012 | 0.0011 |
| mlp_4x64  | 0.0055    | 0.0056| **0.0035** | 0.0004 | 0.0002 |

cap=2 consistently halves the hard-regime error vs synap_only but
still runs ~2-3× higher than synap_only_local. So cap=2 captures
~50% of the locality benefit via faster-convergence-at-matched-total.

### Locality still wins at matched total admissions

Pairwise (`synap_only_local` vs `synap_only_cap1/cap2`):
- vs cap1: 7/8, 7/8, 8/8 (overwhelming)
- vs cap2: 5/8, 7/8, 8/8 (strong)

Since synap_only_local has FEWER total admissions (32-64) than the
cap variants (158), this comparison is also imperfect — it conflates
locality with fewer-edges. The cleanest interpretation:

**Even when cap2 matches synap_only_local's early-convergence
profile, synap_only_local's specific edges (locality-biased) beat
cap2's random edges at roughly 3x the edge count.** Locality is
doing something the random selection cannot replicate.

## True sparsity control is still needed

The clean sparsity-only control is a RATE reduction: lower
`synaptogenesis_rate` so the total admissions match synap_only_local's
32-64 count. That would produce random admissions (not locality-
biased) at matched total.

If synap_only at reduced rate matches synap_only_local's MSE, the
benefit is sparsity. If synap_only_local still wins, locality is
primary. Based on the current data I expect locality to win, but
need the clean test.

**Queuing**: `synap_only_lowrate` with `synaptogenesis_rate ~= 0.63`
(0.63 = 2.0 × 50/158 to match admission count) as a follow-up.

## Secondary finding: convergence speed matters

cap=2 vs cap=1 both have 158 total but differ sharply:
- cap=1: 158 edges built over 1580 steps (1 per fire × 158 fires)
- cap=2: 158 edges built over ~950 steps (1.7/fire × 95 fires)
- cap=2 has ~3000 steps of stable-graph learning after convergence;
  cap=1 has ~2400 steps.

cap=2 wins ~0.002 MSE on hard regimes; cap=1 is indistinguishable
from synap_only. Interpretation: graph must stabilize before stable
learning can accumulate. Reaching steady-state structure quickly
(cap=2) beats dragging it out (cap=1).

This is consistent with but not sufficient to explain synap_only_local:
- local converges even FASTER (27-39 fires, total 32-64 edges)
- local has ~3500 steps of post-convergence stable learning
- AND local's specific edges are locality-biased (better than random)

So "converge fast" + "pick good edges" compose to give synap_local
its benefit. cap=2 captures the first factor but not the second.

## Implementation caveat

The `synaptogenesis_max_admissions_per_step` cap is correctly
implemented — it DOES limit per-call admissions to K. The experiment
misdesign was assuming per-call limits would bound totals; in fact
they just redistribute timing because removed edges stay eligible.

A true per-run sparsity cap would need either:
- A "global admission count" ceiling tracked in SOMA state, OR
- A rate reduction (simpler).

Not worth adding the global ceiling; the rate reduction is the clean
scientific test.

## Paper implication

The synap_local finding remains intact. This experiment REDUCES but
doesn't eliminate the sparsity-vs-locality confound:
- Matched-total (158 edges): locality still wins vs random cap (7/8
  scorecard on most seeds).
- Matched-convergence-profile (cap=2): locality still wins, 5-7/8
  scorecard.
- True matched-total-at-low-count: not yet tested. Next experiment.

Paper narrative (§4.7) can emphasize the structural principle
(locality in where edges form), noting that sparsity alone is
insufficient (cap variants at matched total don't close the gap).

## Next probes (in priority)

1. **Rate-based sparsity control** — synap_only with
   `synaptogenesis_rate=0.63` to match synap_local's total admission
   count via rate dilution. Tests if matched-total-random matches
   locality. (Simplest, cleanest; ~80 min.)
2. **Locality cutoff sweep** (already queued at commit bb9cbb9):
   characterize mechanism across cutoff values.
3. **Position-scramble control**: if sparsity control confirms
   locality dominant, test whether the specific positions carry
   information beyond "coherent distance metric."

## Files

- `src/soma/core/config.py` — synaptogenesis_max_admissions_per_step
- `src/soma/growth/synaptogenesis.py` — cap with rng-subsample
- `tests/test_growth/test_synaptogenesis.py` — 6 cap tests
- `research/developmental/env_sequence_v05_synap_sparsity_control_multiseed.py`
- `research/developmental/results/env_sequence_v05_synap_sparsity_control_multiseed.json`
- `research/developmental/results/env_sequence_v05_synap_sparsity_control_multiseed.log`
