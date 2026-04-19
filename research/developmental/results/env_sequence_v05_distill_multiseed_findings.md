# Direction 4a Phase 2: v0.5 distillation sanity — PASS

**Date:** 2026-04-19
**Commit:** `5284e50` (runner), `b5o798fwr` (run)
**Data:** `research/developmental/results/env_sequence_v05_distill_multiseed.json`
**Seeds:** {0, 1, 42}, 4 variants

## Headline

Direction 4a's distillation loss is **stable under training** on the
v0.5 capacity schedule. All 12 runs (4 variants × 3 seeds) produced
finite MSE and finite projections end-to-end. Sanity gate: **PASS**.

On this substrate, distillation marginally worsens the prediction
MSE on some seeds (+0.0008 to +0.0018 vs. reference synap_only_local).
This is expected — the teacher embeds per-regime pseudo-text strings
that have no natural relationship to the v0.5 synthetic random
vectors, so the distillation signal competes against useful prediction
signal without adding any information.

The important property for Phase 3 is: **the code path works and
doesn't destabilize the substrate.** It does.

## Per-seed MSE (overall average)

| Seed | synap_only_local | _learnable | _distilled (α=0.5) | _distilled10 (α=1.0) |
|------|------------------|------------|--------------------|-----------------------|
|   0  | 0.0022           | 0.0019     | 0.0030             | 0.0032                |
|   1  | 0.0021           | 0.0015     | 0.0021             | 0.0022                |
|  42  | 0.0017           | 0.0022     | 0.0035             | 0.0032                |

Synap event counts identical across variants (seed=0: 32, seed=1: 64,
seed=42: 52) — distillation doesn't affect growth admissions (as
expected; the loss only touches projection weights).

## Per-regime mean MSE (across 3 seeds)

| Regime    | local         | learnable     | distilled α=0.5 | distilled α=1.0 |
|-----------|---------------|---------------|-----------------|-----------------|
| mlp_2x16  | 0.0073±0.0033 | 0.0068±0.0033 | 0.0118±0.0060   | 0.0123±0.0053   |
| mlp_2x32  | 0.0005±0.0009 | 0.0002±0.0003 | 0.0018±0.0012   | 0.0021±0.0014   |
| mlp_3x16  | 0.0002±0.0003 | 0.0001±0.0002 | 0.0004±0.0005   | 0.0003±0.0002   |
| mlp_3x32  | 0.0002±0.0002 | 0.0003±0.0003 | 0.0003±0.0004   | 0.0002±0.0000   |
| mlp_2x64  | 0.0004±0.0007 | 0.0006±0.0009 | 0.0008±0.0009   | 0.0006±0.0011   |
| mlp_3x64  | 0.0016±0.0022 | 0.0013±0.0004 | 0.0026±0.0011   | 0.0024±0.0021   |
| mlp_4x32  | 0.0009±0.0005 | 0.0009±0.0005 | 0.0011±0.0011   | 0.0010±0.0006   |
| mlp_4x64  | 0.0003±0.0006 | 0.0003±0.0006 | 0.0005±0.0009   | 0.0003±0.0006   |

Distillation only meaningfully hurts on the easiest regime (mlp_2x16:
+0.0045 to +0.0050), where prediction MSE is already dominated by the
input noise floor (0.0068-0.0123). On the capacity-stressed regimes
(mlp_3x64, mlp_4x64) the distillation delta is small (±0.001).

## Interpretation

**Why distillation hurts on v0.5:**
The v0.5 inputs are synthetic random vectors drawn from independent
distributions per regime. The pseudo-text "regime-{N}" has no
relationship to the actual vector content. So when the distillation
loss says "make your projections align with mxbai.embed('regime-3')",
it's pulling projection weights toward an arbitrary constant target
that doesn't help the prediction task.

**Why this is OK for Phase 3:**
On LoCoMo, the input IS text (conversational turns), and the teacher
embeds THAT text. So the distillation target IS the actual semantic
signal of the input. Under this condition, the projection learning
should actually converge to something useful.

**Infrastructure verified:**
- Teacher caching works (8 cached embeddings, repeated variants hit cache)
- Distillation loss is differentiable and gradient flows through
- No NaN/Inf accumulates over 4000 training steps
- Projections remain finite
- Synap event counts match reference (no coincidental impact on growth)

## Next: Phase 3

Phase 3 (LoCoMo retrieval benchmark) is where the real test happens.
Runner ready: `benchmarks/run_locomo_distill.py`. 3-way compare:
- chroma-mxbai (baseline)
- soma-random (frozen projections + graph rerank)
- soma-distilled (learnable projections + mxbai distill + graph rerank)

Primary hypothesis: on text inputs where teacher signal is meaningful,
distillation should make the graph rerank add value over pure cosine
retrieval.
