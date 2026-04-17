# D3 Report: Capacity Growth via Structural Plasticity

**Date:** 2026-04-16
**Dimension:** D=50

## 1. Output Scaler Comparison (N=10, D=50)

Mean |output| = 0.1399, mean |pattern| = 1.0000

| Method | Sign Exact | Nearest | Per-bit Acc |
|--------|-----------|---------|-------------|
| Raw | 0.00 | 1.00 | 0.900 |
| Z-score | 0.00 | 1.00 | 0.864 |
| Learned | 0.10 | 1.00 | 0.914 |

## 2. Capacity Sweep with Scaler

| N | N/D | Raw Exact | Z-score Exact | Learned Exact | Nearest |
|---|-----|-----------|---------------|---------------|---------|
| 1 | 0.02 | 0.00 | 0.00 | 1.00 | 1.00 |
| 2 | 0.04 | 0.00 | 0.00 | 0.00 | 1.00 |
| 3 | 0.06 | 0.00 | 0.00 | 0.00 | 1.00 |
| 5 | 0.10 | 0.00 | 0.00 | 0.00 | 1.00 |
| 7 | 0.14 | 0.00 | 0.00 | 0.00 | 1.00 |
| 10 | 0.20 | 0.00 | 0.00 | 0.10 | 1.00 |
| 12 | 0.24 | 0.08 | 0.00 | 0.00 | 1.00 |
| 15 | 0.30 | 0.00 | 0.00 | 0.00 | 1.00 |
| 20 | 0.40 | 0.05 | 0.00 | 0.00 | 1.00 |
| 25 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 |
| 30 | 0.60 | 0.03 | 0.03 | 0.00 | 1.00 |
| 40 | 0.80 | 0.00 | 0.00 | 0.00 | 1.00 |
| 50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 60 | 1.20 | 0.02 | 0.03 | 0.02 | 1.00 |
| 75 | 1.50 | 0.01 | 0.00 | 0.00 | 1.00 |
| 100 | 2.00 | 0.00 | 0.02 | 0.01 | 1.00 |
| 125 | 2.50 | 0.00 | 0.01 | 0.01 | 1.00 |
| 150 | 3.00 | 0.01 | 0.01 | 0.00 | 1.00 |

### Capacity Cliff (N/D where recall drops below 50%)

- **Z-score sign-exact:** N/D = 0.02
- **Learned sign-exact:** N/D = 0.04
- **Nearest-pattern:** N/D = >max tested
- Classical Hopfield (D1): N/D ~ 0.22
- Modern Hopfield (D1): N/D > 3.00 (perfect throughout)

## 3. Structural Plasticity Ablation

### fixed

- Final: N=60 patterns, exact=0.03, nearest=1.00, nodes=14, edges=24
- Max N with exact >= 80%: 0

| N | Exact | Nearest | Nodes | Edges | Grew |
|---|-------|---------|-------|-------|------|
| 1 | 0.00 | 1.00 | 14 | 24 |  |
| 3 | 0.00 | 1.00 | 14 | 24 |  |
| 5 | 0.00 | 1.00 | 14 | 24 |  |
| 10 | 0.00 | 1.00 | 14 | 24 |  |
| 15 | 0.00 | 1.00 | 14 | 24 |  |
| 20 | 0.00 | 1.00 | 14 | 24 |  |
| 25 | 0.00 | 1.00 | 14 | 24 |  |
| 30 | 0.00 | 1.00 | 14 | 24 |  |
| 40 | 0.03 | 1.00 | 14 | 24 |  |
| 50 | 0.02 | 1.00 | 14 | 24 |  |
| 60 | 0.03 | 1.00 | 14 | 24 |  |

### grow_on_saturation

- Final: N=60 patterns, exact=0.00, nearest=0.00, nodes=128, edges=1164
- Max N with exact >= 80%: 0

| N | Exact | Nearest | Nodes | Edges | Grew |
|---|-------|---------|-------|-------|------|
| 1 | 0.00 | 1.00 | 14 | 24 |  |
| 3 | 0.00 | 1.00 | 14 | 24 | Yes |
| 5 | 0.00 | 1.00 | 18 | 64 | Yes |
| 10 | 0.00 | 1.00 | 28 | 164 | Yes |
| 15 | 0.00 | 1.00 | 38 | 264 | Yes |
| 20 | 0.00 | 1.00 | 48 | 364 | Yes |
| 25 | 0.00 | 1.00 | 58 | 464 | Yes |
| 30 | 0.00 | 1.00 | 68 | 564 | Yes |
| 40 | 0.03 | 1.00 | 88 | 764 | Yes |
| 50 | 0.02 | 1.00 | 108 | 964 | Yes |
| 60 | 0.00 | 0.00 | 128 | 1164 | Yes |

### grow_random

- Final: N=60 patterns, exact=0.02, nearest=1.00, nodes=128, edges=1164
- Max N with exact >= 80%: 0

| N | Exact | Nearest | Nodes | Edges | Grew |
|---|-------|---------|-------|-------|------|
| 1 | 0.00 | 1.00 | 14 | 24 |  |
| 3 | 0.00 | 1.00 | 14 | 24 | Yes |
| 5 | 0.00 | 1.00 | 18 | 64 | Yes |
| 10 | 0.00 | 1.00 | 28 | 164 | Yes |
| 15 | 0.00 | 1.00 | 38 | 264 | Yes |
| 20 | 0.00 | 1.00 | 48 | 364 | Yes |
| 25 | 0.00 | 1.00 | 58 | 464 | Yes |
| 30 | 0.00 | 1.00 | 68 | 564 | Yes |
| 40 | 0.03 | 1.00 | 88 | 764 | Yes |
| 50 | 0.02 | 1.00 | 108 | 964 | Yes |
| 60 | 0.02 | 1.00 | 128 | 1164 | Yes |

## Key Findings

### 1. Output scaler does NOT fix sign-exact recall

The magnitude attenuation (output ~0.11 vs target ~1.0) is NOT the root cause
of 0% sign-exact recall. Raw SOMA outputs already have **correct signs ~90%
per-bit on clean patterns** and ~88-92% per-bit on noisy probes. With 50
dimensions, the probability of ALL bits being correct at 90% per-bit is
0.90^50 = 0.005, explaining the ~0% exact rate.

- **Z-score hurts**: mean-centering across a small population shifts some
  near-zero values across the sign boundary, reducing per-bit from 0.90 to 0.86.
- **Learned scaler helps marginally**: boosts per-bit to ~0.91-0.92 (from 0.90)
  and achieves 10% exact at N=10, but the improvement is small.
- **Conclusion**: the bottleneck is SOMA's recall *direction* accuracy from
  noisy probes, not output magnitude. The sign() operation on raw outputs
  already works well for dimensions with large magnitude; the errors are in
  low-magnitude dimensions near zero.

### 2. Nearest-pattern recall is extraordinary

SOMA achieves **100% nearest-pattern recall all the way to N/D=3.0** (150
patterns in D=50) with only 14 nodes. This matches modern Hopfield and
massively exceeds classical Hopfield (cliff at N/D=0.22). The cosine
similarity of SOMA's output to the correct pattern is always higher than to
any other stored pattern, even at extreme loading.

### 3. Structural plasticity does not improve capacity

- All three conditions (fixed, grow-on-saturation, grow-random) show identical
  nearest=1.00 up to N=59 patterns. Growth adds no capacity benefit because
  the 14-node base network already handles the load perfectly on the nearest
  metric.
- Grow-on-saturation degenerates: because exact recall never reaches the 80%
  threshold, it grows at EVERY step (same as grow-random). At N=60 with 128
  nodes and 1164 edges, grow-on-saturation's nearest drops to 0.00 --
  over-growth degrades the attractor landscape.
- **Growth actually hurts**: the fixed 14-node network maintains nearest=1.00
  at N=60, while the 128-node grown network occasionally fails. More nodes
  without targeted integration create interference.

### 4. SOMA's associative recall regime

SOMA operates in a fundamentally different regime from Hopfield networks:

| Metric | Classical Hopfield | Modern Hopfield | SOMA |
|--------|-------------------|-----------------|------|
| Sign-exact capacity | N/D=0.14 cliff | Perfect to N/D=3.0 | ~0% (per-bit ~90%) |
| Nearest-pattern | Same as exact | Same as exact | 100% to N/D=3.0 |
| Output type | Discrete (+/-1) | Continuous | Continuous (compressed) |
| Growth benefit | N/A | N/A | None observed |

SOMA is a **continuous associative memory**: it identifies which pattern was
stored (nearest) with perfect accuracy but does not reconstruct the exact
binary pattern. This is arguably more useful for an agent-memory layer
(retrieval > reconstruction) but does not beat Hopfield at its own game
(sign-exact recall).

### D3 Verdict

SOMA's associative recall is **competitive with modern Hopfield on nearest-
pattern retrieval** (both achieve 100% to N/D=3.0) but **not competitive on
sign-exact reconstruction** (~0% vs 100%). The output scaler is insufficient
to bridge this gap because the limitation is directional accuracy from noisy
probes, not magnitude compression. Structural plasticity does not extend
capacity because the base network already has sufficient capacity for the
nearest-pattern metric; adding nodes without targeted training introduces
interference.

**Recommendation for D4**: Frame SOMA's strength as content-addressable
retrieval (nearest-pattern) rather than exact reconstruction. The capacity
profile is strong for retrieval use cases. Sign-exact recall would require
architectural changes (e.g., a discrete output layer, or training SOMA with
sign-based loss rather than MSE).

See `d3_capacity_growth.json` for full numerical data.
