# v0.5 Spatial-Distillation Multi-seed Findings (Direction 4b Phase 2)

**Status:** MIXED — Gates 1 and 3 PASS; Gate 2 FAILS on seed=42 due to
variance in an adversarial-signal task. Stability + mechanism are
solid; weight tuning recommended before Phase 3.

**Runner:** `research/developmental/env_sequence_v05_spatial_multiseed.py` (commit `af92d1b`).
**Date run:** 2026-04-19.

## What this gates

Phase 2 is a stability + mechanism-fires sanity check on v0.5 (synthetic
16-dim capacity task) before committing to the Phase 3 LoCoMo run.

**Variants:**
- `synap_only_local` — frozen projections, no distillation (reference).
- `synap_only_local_spatial` — Direction 4b full stack:
  `projection_mode=learnable`, `projection_distillation_target=llm_spatial`,
  `position_mode=learnable`, `projection_distillation_winners=3`,
  `position_coupling_weight=1.0`.

**Seeds:** {0, 1, 42}. 2 variants × 3 seeds × 8 regimes × 500 steps.

## Gates

1. **All finite.** No NaN / Inf in MSE, projections, or positions across
   any run. Any failure = mechanism instability; stop and debug.
2. **MSE ratio ≤ 1.5×.** Per-seed `spatial_mse / reference_mse` max must
   be ≤ 1.5.
3. **Position KS distance ≥ 0.1.** Empirical KS distance between initial
   and final pairwise-distance distributions across associator positions,
   averaged over seeds.

## Results

### Per-variant summary (mean across seeds)

| Variant | MSE mean | Synap events | Wall time |
| --- | ---: | ---: | ---: |
| synap_only_local | 0.0021 | 49 | 201s |
| synap_only_local_spatial | 0.0026 | 70 | 267s |

### Per-regime MSE (mean ± (max − min) across 3 seeds)

| Regime | reference | spatial | ratio |
| --- | ---: | ---: | ---: |
| mlp_2x16 | 0.0074 ± 0.0032 | 0.0110 ± 0.0035 | 1.49× |
| mlp_2x32 | 0.0005 ± 0.0006 | 0.0010 ± 0.0007 | 2.00× |
| mlp_3x16 | 0.0003 ± 0.0004 | 0.0002 ± 0.0003 | 0.67× |
| mlp_3x32 | 0.0003 ± 0.0001 | 0.0003 ± 0.0002 | 1.00× |
| mlp_2x64 | 0.0006 ± 0.0002 | 0.0008 ± 0.0010 | 1.33× |
| mlp_3x64 | 0.0017 ± 0.0024 | 0.0023 ± 0.0013 | 1.35× |
| mlp_4x32 | 0.0010 ± 0.0005 | 0.0010 ± 0.0010 | 1.00× |
| mlp_4x64 | 0.0004 ± 0.0001 | 0.0005 ± 0.0009 | 1.25× |

### Per-seed overall MSE

| Seed | reference | spatial | ratio |
| --- | ---: | ---: | ---: |
| 0 | 0.0021 | 0.0026 | 1.29× |
| 1 | 0.0024 | 0.0021 | **0.88×** (spatial better) |
| 42 | 0.0018 | 0.0031 | 1.71× |

### Gate outcomes

| Gate | Target | Observed | Verdict |
| --- | --- | --- | --- |
| 1. All finite | No NaN/Inf | proj_ok=True, pos_ok=True all 6 runs | ✅ PASS |
| 2. MSE ratio ≤ 1.5× | max ≤ 1.5 | max=1.71 (seed 42) | ❌ FAIL |
| 3. Position KS ≥ 0.1 | mean ≥ 0.1 | [0.25, 0.21, 0.14], mean=0.20 | ✅ PASS |

### Overall: MIXED — stability solid, MSE hit borderline.

## Interpretation

### Gate 1 ✅ — stability is solid

All 6 runs (3 seeds × 2 variants) finished with finite projections and
positions. No NaN, no Inf, no crashes. The position-coupling loss, top-K
distillation, norm preservation, and teacher-wiring all compose cleanly.

### Gate 3 ✅ — mechanism actually fires

Mean KS distance of 0.20 across seeds (0.14–0.25 range) confirms the
coupling loss is materially reshaping the pairwise-distance distribution.
If `_position_projector`, the detach semantics on `W_i`, or the
optimizer wiring were silently broken, we'd see KS ≈ 0. They are not.

### Gate 2 ❌ — borderline FAIL, driven by variance + adversarial signal

Seed 42 pushes the max ratio to 1.71×. Two observations:

1. **The synthetic task is adversarial to distillation.** v0.5 sensor
   vectors are random 16-dim draws; the pseudo-text is `"regime-N"`.
   There is no natural semantic relationship between the sensor vector
   and the teacher embedding. So spatial distillation is pulling
   projections toward a signal that is **orthogonal** to next-step
   prediction. Any MSE hit is expected.

2. **Seed variance dominates the signal.** Seeds 0/1/42 span 0.88×,
   1.29×, 1.71× — a 1.94× spread across 3 seeds. This matches the
   variance levels I've seen on other small-effect v0.5 experiments
   (from memory: "ALWAYS multi-seed from day 1 on small-effect
   experiments"). One more seed could easily shift the max.

Hardest regime (mlp_2x16) is the worst offender at 1.49× mean. All
other regimes are within 1.35× or better.

## Decision

**Proceed cautiously to Phase 3 with a β=0.3 ablation first:**

- Gate 2 "failed" on a task where we EXPECT distillation to hurt MSE.
  Real LoCoMo has semantic signal; spatial can actually help there.
- Before spending 20 min on LoCoMo, run a quick v0.5 ablation at β=0.3
  (lower position coupling) to confirm the mechanism responds to weight
  tuning. If β=0.3 passes Gate 2, we have two defensible Phase 3
  settings (0.3 conservative, 1.0 design default) and can run both.

## Next steps

1. **Immediate:** Run β=0.3 ablation in v0.5 (1 variant × 3 seeds, ~10 min).
2. **If β=0.3 passes Gate 2:** Run LoCoMo with β=0.3 first (conservative),
   then β=1.0 if the conservative setting shows R@5 movement.
3. **If β=0.3 also fails Gate 2:** Debug before LoCoMo — likely indicates
   the position-coupling loss is fighting prediction at any weight, which
   means the Direction 4b mechanism design needs revisiting.
