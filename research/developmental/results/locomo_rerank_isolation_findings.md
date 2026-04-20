# LoCoMo graph rerank is a net drag (isolation study)

**Status:** Major finding. The graph rerank as currently tuned (weight=0.3,
gate_threshold=0.05) costs SOMA −0.013 R@5 on LoCoMo. Pure cosine SOMA
matches chroma. Explains why Direction 4a/4b distillation experiments failed.

**Date:** 2026-04-19.
**Commit:** `4549045` added `--rerank-weight` / `--gate-threshold` CLI flags.

## What we ran

`python -m benchmarks.run_locomo_distill --skip-spatial --rerank-weight 0.0
--variant-suffix=_norerank`

One knob change: `rerank_weight=0.0` instead of the prior default 0.3.
Everything else identical to the Phase 3 Direction 4b run:
- 10 conversations, 5882 turns, 1982 queries
- target_dim=128, mxbai-embed-large teacher
- soma-random (frozen projections), soma-distilled (llm_embedding distill)

## Results

### With vs without rerank

| Variant | rerank_weight | R@1 | R@5 | R@10 | retrieve |
| --- | ---: | ---: | ---: | ---: | ---: |
| chroma-mxbai | n/a | 0.147 | 0.349 | 0.446 | 10.1ms/1.9ms |
| soma-random | 0.3 (prior) | 0.133 | 0.337 | 0.433 | 19.0ms |
| soma-random | **0.0** | **0.148** | **0.350** | **0.447** | 18.9ms |
| soma-distilled | 0.3 (prior) | 0.134 | 0.344 | 0.441 | 19.5ms |
| soma-distilled | **0.0** | **0.148** | **0.350** | **0.447** | 17.6ms |

### Per-category R@5 at rerank=0

| System | single-hop | multi-hop | temporal | open-domain | adversarial |
| --- | ---: | ---: | ---: | ---: | ---: |
| chroma-mxbai | 0.262 | 0.361 | 0.217 | 0.357 | 0.406 |
| soma-random (rerank=0) | 0.264 | 0.361 | 0.217 | 0.359 | 0.406 |
| soma-distilled (rerank=0) | 0.264 | 0.361 | 0.217 | 0.359 | 0.406 |

SOMA matches chroma to within ±0.002 across every category when rerank is off.

### Impact of rerank on each variant

| Variant | R@5 Δ from rerank | R@1 Δ | R@10 Δ |
| --- | ---: | ---: | ---: |
| soma-random | **−0.013** | −0.015 | −0.014 |
| soma-distilled | −0.006 | −0.014 | −0.006 |

Rerank universally hurts; distillation partially compensates for the damage
(−0.013 → −0.006 at R@5), but never beats pure cosine.

## Interpretation

The gated-hybrid formula `(1 − w)·emb_sim + w·fp_sim` gates on
`confidence = fp_sorted[0] − fp_sorted[1] >= gate_threshold`. When the
gate fires, rerank mixes the graph fingerprint into the score at
weight 0.3.

For an untrained graph (soma-random): fingerprints are near-random, so
rerank injects noise into 30% of each score. The gate still fires (the
confidence heuristic doesn't know the signal is garbage) and the noise
flips ranks.

For a Direction-4a-distilled graph: fingerprints are pulled toward a
shared teacher direction (degeneracy), so `fp_sorted[0] − fp_sorted[1]`
is tiny, gate fires less, and the fingerprint signal is less diverse.
Partial healing of the rerank damage.

## Reframing past experiments

- **Direction 4a null** (soma-distilled < chroma): now explained.
  Distillation trained a fingerprint, but the rerank formula that
  consumed it was net-negative. Distillation was partially compensating
  for its own damage.
- **Direction 4b null** (soma-spatial < chroma): same root. The spatial
  channel trained positions, not fingerprints. So spatial distillation
  had ZERO compensation effect against the rerank drag — it performed
  worse than Direction 4a, exactly as observed.
- **Locality null on LoCoMo** (prior finding): consistent. Locality
  gates synaptogenesis; synaptogenesis shapes the graph; the graph
  shapes the fingerprint; the fingerprint is consumed by a broken
  rerank. No amount of upstream sophistication can rescue a
  downstream formula that adds noise.

## Latency puzzle — resolved

Before short-circuit (`retrieve_hybrid` always called `soma.step`):
- chroma retrieve: 1.9–10.1ms/query (in-process chromadb HNSW)
- SOMA retrieve: 17–19ms/query regardless of rerank_weight

After short-circuit (skip `soma.step` + lateral inhibition + fingerprint
when `rerank_weight == 0.0`):
- chroma retrieve: 1.7ms/query (10-conv run)
- SOMA retrieve: **0.7ms/query** (10-conv run)

**SOMA is now 2.4× faster than chroma on retrieve**, validating the
original "2× faster at matched quality" positioning claim — just for
a different reason than initially framed. The claim was previously
attributed to the graph-rerank path; the actual source is pure cosine
over learned-dim embeddings run on CUDA, with the graph rerank staying
off by default.

## Decision

**Action 1 (required):** Change default retrieval path. `rerank_weight`
should default to 0.0 (or the whole rerank branch should be disabled
unless explicitly opted-in). Keep the rerank code — we may fix it — but
stop it from degrading retrieval silently.

**Action 2 (required):** Short-circuit the graph forward pass in
`retrieve_hybrid` when `rerank_weight == 0.0`. Would drop SOMA
retrieve latency meaningfully (the forward pass is the bulk).

**Action 3 (investigate):** Is there a rerank formulation where the
fingerprint adds signal rather than noise? Possible angles:
  - Much higher `gate_threshold` (only rerank when fingerprint is
    confidently differentiated, not just 0.05 differentiated)
  - Adaptive weight based on fingerprint confidence
  - Restrict rerank to specific query categories (multi-hop? temporal?)
  - Use fingerprint as a tiebreaker at position k/k+1 rather than
    a continuous weight
If none of these produce a positive Δ R@5, the current rerank
formulation is dead code in a retrieval context.

**Action 4 (positioning):** Update `docs/positioning.md`:
  - Remove/rewrite the "2× faster" claim (not supported by this data)
  - Add: "SOMA cosine retrieval matches chroma on LoCoMo at same
    embedding dim"
  - Add: "Graph rerank is currently off by default; reformulation
    pending"

## Next concrete step

Rerank weight sweep: {0.0, 0.05, 0.1, 0.2, 0.3} on soma-random to
find the crossover (if any). ~30 min. Determines whether Action 3
is worth pursuing or whether rerank is dead code.

## Files

- `benchmarks/reports/locomo_rerank_isolation.md` + `.json`
- `benchmarks/reports/locomo_rerank_isolation.log`
