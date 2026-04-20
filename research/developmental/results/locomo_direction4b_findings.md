# Direction 4b LoCoMo Phase 3 Findings

**Status:** STOP. Direction 4b is a null on LoCoMo retrieval. Primary,
secondary, and tertiary-1 gates all fail. Same failure surface as
Direction 4a closed against.

**Date:** 2026-04-19.
**Runner:** `benchmarks/run_locomo_distill.py` (commit `1b8bf33`, extended
with `soma-spatial` system + `--spatial-beta`/`--spatial-winners` flags).
**Scope:** Full LoCoMo, 10 conversations, 5882 turns, 1982 queries.

## Results

### Full R@k matrix

| System | locality | R@1 | R@5 | R@10 | retrieve |
| --- | ---: | ---: | ---: | ---: | ---: |
| chroma-mxbai | n/a | 0.147 | **0.349** | **0.446** | 10.1ms |
| soma-random | 0.0 | 0.133 | 0.337 | 0.433 | 19.0ms |
| soma-distilled (4a) | 0.0 | 0.134 | 0.344 | 0.441 | 19.5ms |
| soma-spatial (4b) | 0.0 | 0.134 | 0.335 | 0.437 | 15.9ms |
| soma-distilled_locality | 0.5 | 0.135 | 0.337 | 0.436 | 16.7ms |
| soma-spatial_locality | 0.5 | 0.132 | 0.328 | 0.436 | 14.2ms |

### Gate outcomes

| Gate | Definition | Observed | Verdict |
| --- | --- | --- | --- |
| Primary | spatial > chroma on R@5 by ≥ 0.02 | Δ = −0.014 | ❌ FAIL |
| Secondary | spatial > distilled on R@5 | Δ = −0.009 | ❌ FAIL |
| Tertiary 1 | locality ON > OFF for spatial | Δ = −0.007 | ❌ FAIL |
| Tertiary 2 | latency ≤ 32ms | 15.9ms (14.2 with locality) | ✅ PASS |

### Per-category R@5 (locality=0.0)

| System | single-hop | multi-hop | temporal | open-domain | adversarial |
| --- | ---: | ---: | ---: | ---: | ---: |
| chroma-mxbai | 0.262 | 0.361 | 0.217 | 0.357 | 0.406 |
| soma-random | 0.245 | 0.361 | 0.239 | 0.342 | 0.386 |
| soma-distilled | 0.259 | 0.364 | 0.239 | 0.344 | 0.404 |
| soma-spatial | 0.252 | 0.336 | 0.239 | 0.344 | 0.390 |

Spatial wins no category. The multi-hop drop (0.364 → 0.336) is the
largest single-category hit.

## Why did it fail?

### The mechanism ran correctly

Phase 2 Gate 3 (KS ≥ 0.1) passed at 0.20 across seeds — positions DID
reshape during training. The `_position_projector` buffer, top-K
winner selection, position-coupling loss, norm preservation, and
save/load machinery all composed cleanly (67 developmental tests, full
suite 2551 green).

### But the reshaping encoded the wrong thing

Position coupling anchored each node's position to
`normalize(P.T · W_i.flatten().detach())`:

- `W_i` is the input-projection weight matrix for node `i`. These start
  from seeded random init, and drift under the distill + prediction
  gradient. Their content is "how node `i` views input vectors."
- `P` is a fixed random buffer. Its purpose is Johnson-Lindenstrauss
  preservation of pairwise distances — a compression, not a semantic
  decoder.
- So `p_i` becomes a fixed-random-projected view of `W_i`'s flat
  weights — it encodes **the projection's current state**, not **the
  kind of content that projection is trained to see**.

Locality uses positions to filter synaptogenesis edges. Locality ON
with spatial positions made retrieval WORSE (0.335 → 0.328). Running
locality based on these positions actively gates out useful edges.
This is the key diagnostic: the coupling produced positions that are
NEGATIVE signal for retrieval, not just neutral.

### Connection to Direction 4a's null

Direction 4a closed because the mean-target distill loss pulled all
projections toward a shared direction (degeneracy). Direction 4b was
supposed to fix that with competitive top-K + spatial channel. The
top-K part works (KS passes, per-node specialization is visible). The
spatial channel is where 4b reintroduces a new degeneracy: tying
positions to projection-weight-norms means positions become a
reformulation of the same projection information that's already in
the graph, compressed through an arbitrary random map. No new signal
enters.

## Decision: STOP

Per the design doc's decision matrix:

> fail / fail: Spatial distillation is itself a null. Pivot to
> different failed direction from the meta-principle table (e.g.,
> spatial PE-supervised synap, spatial plasticity broadcast,
> spatial graph rerank).

Direction 4b joins 4a in the closed-branches folder. Keep
infrastructure — the spatial stack is clean and other experiments
(Option C pairwise-distance, Option D spatial rerank) could re-use
it without re-implementing.

## What we learned (generalizes beyond 4b)

1. **Random projection P isn't a semantic decoder.** Distilling
   `p_i = P.T · W_i` just rotates the projection into a different
   coordinate system. The teacher signal never reaches it through
   that channel — the teacher trains `W_i` via the distill loss on
   projected views, not through `P`.

2. **Positional locality's v0.5 win was prediction-substrate-specific.**
   Memory already flagged LoCoMo locality null; confirmed again here.
   The v0.5 finding that "positional locality is the key factor"
   applies to synthetic next-step MSE, not to retrieval.

3. **Competitive top-K alone is not sufficient.** Breaking 4a's
   degeneracy requires real per-node specialization, which requires
   the distill signal to actually MAP inputs to teacher embeddings
   per-node. We had the gating but not the mapping.

4. **Teacher cache + harness worked.** Phase 3 full run took ~18 min
   end-to-end because teacher embeddings were cached. Infrastructure
   is solid; next spatial experiment benefits from this.

## Next steps

Per the design doc's future-work section, viable follow-ups ordered by
my guess at payoff:

- **Option D (spatial rerank)** — add per-memory positions, use
  `position_sim(query_pos, memory_pos)` as a third rerank term
  alongside embedding and fingerprint. This targets the retrieval
  path directly rather than going through positions → locality →
  synaptogenesis → retrieval.
- **Option C (pairwise distance distillation)** — contrastive loss
  on `(||p_i − p_j||, ||teacher_i − teacher_j||)`. Most mathematically
  principled; ties positions directly to teacher distances rather
  than to weight matrices.
- **Spatial rerank / spatial PE-supervised synap / spatial plasticity
  broadcast** from the meta-principle table — each revisits a prior
  null with spatial mechanisms.

Independent of which branch to pursue next: the latency win remains
(soma-random is ~2× faster than chroma on `retrieve_avg_ms` under
matched R@5, per earlier positioning update). That story is intact.

## Files

- `benchmarks/reports/locomo_direction4b.md` + `.json` — β=1.0, locality=0.0
- `benchmarks/reports/locomo_direction4b_locality.md` + `.json` — β=1.0, locality=0.5
- `research/developmental/results/env_sequence_v05_spatial_multiseed.json` — Phase 2 β=1.0
- `research/developmental/results/env_sequence_v05_spatial_multiseed_beta03.json` — Phase 2 β=0.3
- `research/developmental/results/env_sequence_v05_spatial_multiseed_findings.md`
