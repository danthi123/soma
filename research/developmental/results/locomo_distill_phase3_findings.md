# Direction 4a Phase 3: LoCoMo distillation — NULL, honest

**Date:** 2026-04-19
**Commits:** `f344eae` (runner), `bgdcu32ji` (run)
**Data:** `benchmarks/reports/locomo_distill_full.json`
**Seeds:** 1 (paired; rng seed=0 across variants)
**Benchmark:** full LoCoMo (10 conversations, 5882 turns, 1982 queries)

## Headline

**Direction 4a's primary hypothesis fails on LoCoMo at default
hyperparameters.** LLM-distilled projections do not improve retrieval
over pure mxbai-based cosine retrieval (Chroma), and actually perform
*slightly worse than* a SOMA variant with random projections.

| System         | R@1   | R@5   | R@10  | retrieve_avg |
|----------------|-------|-------|-------|--------------|
| chroma-mxbai   | 0.147 | **0.349** | 0.447 | 34.3ms       |
| soma-random    | 0.130 | 0.343 | 0.441 | 16.0ms       |
| soma-distilled | 0.127 | 0.334 | 0.436 | 16.3ms       |

Per the design doc's decision matrix:
- **Primary** (soma-distilled > chroma-mxbai by > 0.02): FAIL (−0.015)
- **Tertiary** (soma-distilled > soma-random): FAIL (−0.009)

## Per-category R@5 at full scale

| Category    | chroma-mxbai | soma-random | soma-distilled | Δ(dist−chroma) |
|-------------|--------------|-------------|----------------|----------------|
| single-hop  | 0.262        | 0.248       | 0.245          | −0.017         |
| multi-hop   | 0.361        | 0.368       | 0.340          | −0.021         |
| temporal    | 0.228        | 0.239       | 0.239          | **+0.011**     |
| open-domain | 0.357        | 0.352       | 0.344          | −0.013         |
| adversarial | 0.406        | 0.390       | 0.388          | −0.018         |

The subset (n=2) showed a strong temporal delta (+0.091). At full
scale (n=10) the temporal effect shrinks to +0.011 — indistinguishable
from noise. The subset result was a false positive from small-sample
variance.

## What the null tells us

The architecture I built has a **real gap** between what distillation
trains and what the retrieval path uses:

1. **Distillation trains `_input_projections`** in PredictiveSOMA's
   `_diversify_activations`. These are per-associator matrices that
   modulate how each node responds to its input.

2. **Retrieval uses `_get_node_fingerprint`** which is built from
   node activations → projected via scatter_add to a fingerprint
   vector. Node activations depend on node weights (trained via
   Hebbian learning), not directly on input projections.

3. The chain from distilled projection → modified activation →
   different fingerprint → different top-K is **multiple layers deep**,
   and each layer dilutes the distillation signal.

4. The locality filter — which was the v0.5 positive — operates on
   node POSITIONS, which are initialized randomly in the
   `position_dim` space and NEVER updated by distillation.

So there's no mechanism by which distillation should improve locality-
based filtering. The v0.5 → retrieval transfer story was always
conditional on a stronger mechanism than what's wired up.

## What would need to change for distillation to actually land

Three increasingly-invasive options:

**A. Minimal (hyperparameter tuning):**
- Lower alpha (0.1–0.2 vs tested 0.5) so distillation doesn't dominate
- Higher `sensor_output_dim` (256 or 512) to preserve mxbai signal
- Different `rerank_weight` in retrieve_hybrid (tested 0.3)

Worth a few hours. Cache is populated so iteration is fast.

**B. Moderate (architectural):**
- Add distillation signal to node weights, not just input projections
- Train positions to align with distilled projection centroids so
  locality filter operates on semantic space

This is ~1-2 days of refactoring and re-testing.

**C. Major (rethink):**
- Drop the "graph rerank" framing; use PredictiveSOMA as a feature
  extractor whose outputs *replace* mxbai embeddings (not complement
  them)
- Full retraining loop with distillation as the primary objective

This becomes a new research arc, not Direction 4a.

## Decision

**Per the design doc's STOP condition:** Phase 3 primary FAIL →
document honestly, pivot.

**Before declaring full STOP:** I'll try one cheap iteration (Option A
minimal hyperparameter sweep) since the embedding cache is populated
and the marginal cost is small. Specifically:
- Lower alpha to 0.1
- Add locality filter (synaptogenesis_max_distance=0.5)
- target_dim=256 (preserve more semantic info)

If that also returns null, we have a clean STOP with a clear next-arc
candidate (Option B or C).

## Notes on speed

Subset (788 turns, 302 queries, 3 systems): ~55s
Full (5882 turns, 1982 queries, 3 systems): ~10 min

Most of the time is PredictiveSOMA's process_input — not Ollama
(cached). The ~2x retrieve latency advantage (SOMA 16ms vs Chroma 34ms)
is real and might be useful regardless of R@k outcome.
