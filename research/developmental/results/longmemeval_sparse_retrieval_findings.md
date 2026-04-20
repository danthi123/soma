# LongMemEval sparse retrieval — Path B Phase 3 (NO-GO)

**Date:** 2026-04-20.
**Status:** CLOSED — Phase 3 NO-GO confirmed at N=100.
**Tracks:** `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md` § Phase 3.
**Script:** `benchmarks/industry/longmemeval/run_sparse_retrieval.py`.
**Analysis:** `scripts/analysis/compare_sparse_hybrid.py --suffix _n100_sparse`.

## Headline

**Adding a sparse-overlap retrieval score (from k-WTA codes over
random-normal projection of sbert embeddings) to SOMA's hybrid path
substantially hurts rank-1 retrieval on LongMemEval.** At N=100,
rank-1 drops from 0.790 (hybrid-only) to 0.660 (hybrid + sparse) —
a 13-point regression. Paired breakdown: sparse loses 27×, wins 5×,
ties 68×.

## N=100 final results

| variant | n | hit@5 | rank-1 | notes |
|---|---:|---:|---:|---|
| hybrid_only | 100 | 0.960 | **0.790** | baseline (0.7·cos + 0.3·bm25) |
| hybrid_plus_sparse | 100 | 0.910 | 0.660 | 0.5·cos + 0.2·bm25 + 0.3·sparse |
| sparse_only | 100 | 0.560 | 0.340 | sparse signal alone — sanity |

**Per-question-type breakdown (100 items covers two types):**

| question-type | N | variant | hit@5 | rank-1 |
|---|---:|---|---:|---:|
| multi-session | 30 | hybrid_only | 0.967 | **0.900** |
| multi-session | 30 | hybrid_plus_sparse | **1.000** | 0.867 |
| multi-session | 30 | sparse_only | 0.933 | 0.700 |
| single-session-user | 70 | hybrid_only | 0.957 | **0.743** |
| single-session-user | 70 | hybrid_plus_sparse | 0.871 | 0.571 |
| single-session-user | 70 | sparse_only | 0.400 | 0.186 |

One mild silver lining: on **multi-session** queries, `hybrid_plus_sparse`
improves hit@5 from 0.967 → 1.000 (recovers one miss into the top-5),
though rank-1 slips from 0.900 to 0.867. On **single-session-user**
(the dominant question type by count), sparse is a clear loss on both
metrics.

Paired (hybrid_plus_sparse vs hybrid_only, all 100 items):
- **Sparse wins**: 5 (5.0%)
- **Sparse loses**: 27 (27.0%)
- Ties: 68 (68.0%)
- Net: **-22**

### GO/NO-GO gate

From design doc § Phase 3:
- `hybrid_plus_sparse` R@5 ≥ `hybrid_only` R@5 on N=100 with at least one
  question-type showing rank-1 lift ≥ 5pp — **FAIL**.

**Verdict: NO-GO on Primitive 1 (k-WTA sparse codes).**

## Why sparse codes don't lift retrieval

Three likely causes, listed in order of our best guess:

1. **Top-k boundary is unstable for similar dense inputs.** With
   dim=4096 and k=32, the gap between the 32nd and 33rd ranked
   projected activations is small (≈0.01 std under a random-normal
   projection of standardised inputs). Two semantically similar sbert
   embeddings get projected to vectors that differ by a small
   perturbation — but that perturbation can easily swap the boundary
   indices, so their top-32 sets come out mostly disjoint. Random-
   projection k-WTA is NOT a locality-sensitive hash; it fails to
   preserve the near-neighbour relationships we need for retrieval.

2. **Semantic structure lives in direction, not in high-activation
   indices.** sbert embeddings separate concepts by direction (cosine
   similarity). Projecting them to random-normal 4096d and picking
   top-k indices throws away the directional signal — the top-k
   indices are essentially a hash, not a semantic lookup.

3. **BM25 already captures what sparse codes try to capture.** The
   "sparse orthogonal code per concept" intuition from biology is
   that similar concepts share some active dims but differ in others.
   BM25 already does this at the token level: two queries about the
   same topic share high-IDF terms but differ on specifics. BM25 has
   the benefit of being grounded in actual corpus statistics — ours
   contributes real semantic distinction, whereas random-projection
   sparse codes contribute mostly noise.

Any one of these explains the paired-loss pattern. All three probably
contribute.

## What we did NOT try (and why)

- **LSH / sign-random-projection codes**: could be more stable under
  small perturbations; is orthogonal to the k-WTA primitive the design
  doc calls for. Deferred — if Phase 3 had been GO on k-WTA we'd
  explore LSH as a faster-retrieval variant.
- **Learned projection**: the design doc called this out as Q1 (open
  question on projection choice). A learned projection would need a
  training pipeline we don't have; Phase 3 was meant to test whether
  the *primitive class* (sparse codes + overlap) helps at all.
  Random-normal result says no.
- **Larger k (256 or 512 at dim=4096)**: would produce denser codes
  with smoother top-k boundaries; would also dilute the sparse-code
  concept toward "a dense noisy version of the same score you already
  have." Low information value for our design question.

## Consequence for Path B and Path A

Per design doc § Phase 3 recovery:
> If this fails, sparse codes aren't adding retrieval signal over
> hybrid; we pivot to Primitive 3 or 4 testing before declaring failure.

**Next primitives to test (conditional):**

- **Primitive 3: modern Hopfield attractor retrieval** —
  `src/soma/memory/attractor.py`. Tests whether "noisy query lands on
  correct attractor" pattern helps paraphrased queries. Distinct
  mechanism from Primitive 1 (it uses the dense codes directly, not
  the sparse codes). Worth a shot.
- **Primitive 4: engram tagging (recency + frequency)** —
  `src/soma/memory/engram.py`. Small, cheap, independent of the
  sparse-code question. Measures whether per-memory recency/frequency
  weighting adds signal on multi-session queries.

Both are self-contained and don't require calibration. If either
adds retrieval signal on top of hybrid, that's Phase 4 or Phase 5 GO.
If neither does, Path B closes as a cohesive negative result, and the
conclusion extends to Path A: **biological-primitives-as-retrieval-
mechanisms do not beat hybrid on LongMemEval**. Path A's
artificial-life research track (Phase 6+) remains open as non-product
research.

## Files / reproduction

```bash
# The N=100 run that produced this finding
python -m benchmarks.industry.longmemeval.run_sparse_retrieval \
    --limit 100 --top-k 5 --sbert-device cuda \
    --sparse-k 32 --sparse-dim 4096 --out-suffix _n100_sparse

# Paired analysis
python -m scripts.analysis.compare_sparse_hybrid --suffix _n100_sparse
```

Results:
- `benchmarks/industry/longmemeval/results/sparse_retrieval_hybrid_only_n100_sparse.jsonl`
- `benchmarks/industry/longmemeval/results/sparse_retrieval_hybrid_plus_sparse_n100_sparse.jsonl`
- `benchmarks/industry/longmemeval/results/sparse_retrieval_sparse_only_n100_sparse.jsonl`
- `benchmarks/industry/longmemeval/results/sparse_retrieval_summary_n100_sparse.json`

## Cross-reference

- `research/developmental/results/sim_ca3_baseline_findings.md` —
  Phase 1 NO-GO (sim doesn't produce separated codes)
- `research/developmental/results/sparse_codes_synthetic_sweep.json` —
  Phase 2 calibration (primitives work on synthetic data, sep ratio 5.66)
- `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md` —
  full plan
- `docs/milestones/2026-04-20-hybrid-retrieval-validated.md` — the M1
  baseline this tested (and failed to improve)
