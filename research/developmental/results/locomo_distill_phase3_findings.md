# Direction 4a Phase 3: LoCoMo distillation — NULL across configs

**Date:** 2026-04-19
**Commits:** `f344eae` (runner), `bgdcu32ji` (main), `bwoi03hct` (sweep)
**Data:** `benchmarks/reports/locomo_distill_{full,sweep_a01_local}.json`
**Benchmark:** full LoCoMo (10 conversations, 5882 turns, 1982 queries)

## Headline

**Direction 4a's primary hypothesis fails on LoCoMo at every config
tested.** LLM-distilled projections do not improve retrieval over
pure mxbai-based cosine retrieval (Chroma). The gap ranges from
−0.008 to −0.015 on R@5.

## Results across configs

| System                          | R@1   | R@5   | R@10  | Δ vs chroma |
|---------------------------------|-------|-------|-------|-------------|
| chroma-mxbai                    | 0.147 | **0.349** | 0.447 | baseline    |
| soma-random                     | 0.130 | 0.343 | 0.441 | −0.006      |
| soma-distilled (α=0.5, no loc)  | 0.127 | 0.334 | 0.436 | −0.015      |
| soma-distilled (α=0.1 + loc=0.5)| 0.127 | 0.341 | 0.440 | −0.008      |

Lower alpha + locality filter recovers some ground (−0.015 → −0.008)
but doesn't close the gap. All variants remain below chroma-mxbai.

## Per-category R@5 across configs

| Category    | chroma | soma-random | distilled α=0.5 | distilled α=0.1 + loc |
|-------------|--------|-------------|-----------------|------------------------|
| single-hop  | 0.262  | 0.248       | 0.245           | 0.241                  |
| multi-hop   | 0.361  | 0.368       | 0.340           | 0.343                  |
| temporal    | 0.228  | 0.239       | 0.239           | 0.228                  |
| open-domain | 0.357  | 0.352       | 0.344           | 0.356                  |
| adversarial | 0.406  | 0.390       | 0.388           | 0.397                  |

Locality helps open-domain and adversarial slightly but hurts
temporal (from 0.239 to 0.228). No category shows a convincing win.

## Decision matrix (per design doc)

| Criterion                              | Result |
|----------------------------------------|--------|
| Primary: distilled > chroma by 0.02    | **FAIL** (best = −0.008) |
| Secondary: distilled+local > distilled | TIE (+0.007, within noise) |
| Tertiary: distilled > random           | TIE (−0.002 to −0.009) |

Per the design doc: **STOP condition.** Primary fails. Document
honestly and pivot.

## The honest interpretation

**Distillation trains the wrong component for retrieval.** My
implementation puts the distillation loss on PredictiveSOMA's
`_input_projections` — these modulate node activations during
storage but the RETRIEVAL path uses the node fingerprint, which is
dominated by node weights (trained via Hebbian), not projections.

So the chain is: distilled projection → slightly modified activation
→ slightly different fingerprint → essentially-same top-K. Each
layer dilutes the distillation signal to near-zero.

**The locality filter also misses its target.** On v0.5, locality
worked because node positions happened to correlate with projection
weights (shared rng state at init). In the LoCoMo benchmark, the
locality filter still operates on position-space distances, but
distillation never touches positions. So "semantic locality" was
never actually implemented — positions stayed as randn-init values.

## The silver lining

**SOMA variants retrieve ~2× faster than Chroma** (16–24ms vs 34ms)
at effectively-identical retrieval quality (soma-random within
−0.006 of chroma-mxbai). This is a latency product feature even
without a quality differentiator.

**Graph rerank is safe.** Even with random projections, the graph
doesn't degrade retrieval meaningfully (−0.006 at most). This
validates the product positioning that enabling SOMA's graph doesn't
introduce retrieval-quality risk at default alpha.

## Next research arcs (pivot options)

Per the design doc's STOP branch, several directions are available:

**Option 1: Deeper re-architecture**
Move distillation from input projections to node weights, and train
positions via the same signal so locality filter operates in
semantic space. ~1-2 days of refactoring.

**Option 2: Latency / continual-learning positioning**
Lean into the 2× latency advantage and the v0.5 prediction-substrate
findings. Position SOMA as "local-first memory layer, faster than
vector DB, continuous learning" rather than "beats cosine retrieval."

**Option 3: Different problem**
The v0.5 prediction result is real and multi-seed validated.
Benchmarks that stress continual learning or temporal prediction
might show SOMA's advantage more directly than LoCoMo retrieval does.
B3 ACC/BWT results (from earlier session) already show SOMA's anti-
forgetting property — extend that line instead.

## Phase 4 (cutoff sweep) / Phase 5 (category analysis) — CANCELLED

These were conditional on Phase 3 showing signal. With Phase 3 null,
both are exercises in refining a dead hypothesis. Shelf them
until/unless the mechanism is fixed.

## Commit trail (Direction 4a, 2026-04-19)

```
e212798 Direction 4a design doc
559b145 Implementation plan
9041d18 Config fields (Phase 1.1)
de71346 OllamaEmbedder (Phase 1.2)
cb72845 CachedEmbedder (Phase 1.3)
3daed4d Distillation loss in PredictiveSOMA (Phase 1.4)
352a8a4 End-to-end smoke test (Phase 1.5)
fd0cbba v0.5 backward-compat (Phase 1.6)
5284e50 Phase 2 runner
673274b Phase 2 findings (sanity PASS)
2384072 SomaAdapter distill flow-through
5e8bc93 CATEGORY_NAMES iteration fix
fcaf7a0 Phase 3 runner (initial)
ba4883d Phase 3 scoring bug fix
f344eae Phase 3 per-sample protocol (dia_id collision fix)
2a9c1fc Phase 3 subset result (+0.091 temporal, n=2)
5693272 Phase 3 full result (NULL primary)
db6ee54 Phase 3 runner parameterization
[pending] Phase 3 sweep + final synthesis
```

## Total investment

- Design + planning: 2 hours
- Implementation (Phase 1): 3 hours (TDD all seven sub-tasks)
- Phase 2 sanity: 1 hour (including background-run wait)
- Phase 3 implementation + debug + runs: 2 hours
- Synthesis + docs: 1 hour

**Total: ~9 hours over a single autonomous session.** Matches the
design doc's 10-15h estimate; timeboxed as planned when null became
clear. Clean documentation of the null preserved for future work.
