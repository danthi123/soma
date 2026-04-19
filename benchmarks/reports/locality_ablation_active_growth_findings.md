# Locality ablation on retrieval (active-growth) — WEAK POSITIVE

**Date:** 2026-04-19
**Commit:** d1665a1 (impl), b7ta2tbgr (run)
**Data:** `benchmarks/reports/locality_ablation_active_growth.md`
**Dataset:** 50 facts, 26 queries, paired seeds {0, 1, 42}

## Headline

The v0.5 locality finding DOES NOT cleanly transfer to the synthetic
retrieval benchmark. Locality=0.5 vs 0.0 shows:
- No effect at alpha ∈ {0.00, 0.10, 0.20}
- **Weak effect at alpha=0.30**: mean Δ +0.013, but only 1/3 seeds
  show a real difference

The single-seed effect is real (seed=1: 0.788 → 0.827, +0.038) but
doesn't generalize. The other two seeds show 0.923 → 0.923 and
0.942 → 0.942 — no change.

## Interpretation: downside-protection, not upside-creation

Look closely at alpha=0.30 per-seed:

| seed | locality=0.0 | locality=0.5 | delta |
|------|--------------|--------------|-------|
| 0    | 0.923        | 0.923        | +0.000 |
| 1    | **0.788**    | **0.827**    | **+0.038** |
| 42   | 0.942        | 0.942        | +0.000 |

seed=1's locality=0.0 run is the outlier — it DEGRADED from the
flat baseline (0.923) to 0.788 at alpha=0.30. The other two seeds
didn't degrade. With locality ON, seed=1 recovers partially (0.827).

So locality's role on retrieval appears to be **downside protection
at high alpha**: it prevents a seed-specific failure mode where
graph re-rank pulls rankings away from the cosine-optimum. At low
alpha, locality doesn't matter because the graph contribution is
small relative to cosine. At high alpha, locality prevents rare
catastrophic interactions.

This is qualitatively different from the v0.5 finding, where
locality was a STRONG POSITIVE mechanism (7-8/8 wins across all
seeds, 10× MSE reductions). On retrieval it's barely detectable
without cherry-picking alpha.

## Why the v0.5 → retrieval transfer is weak

The two workloads stress the graph differently:

**v0.5 (synthetic prediction):**
- The graph's job: learn temporal dependencies via edge weights
  that pass activation.
- Every edge participates in every forward pass.
- Harmful edges create noise throughout the prediction pipeline.
- Locality-filtered edges are structurally coherent → less noise
  → cleaner gradient signal.

**Retrieval (synthetic 50-fact):**
- The graph's job: bias a cosine similarity ranking via
  activation-distance re-rank.
- Graph's contribution is WEIGHTED at alpha (0.0-0.5).
- Per-query, only a small subset of edges actually affects the
  top-K ranking.
- Cosine similarity on sbert embeddings is already strong (0.923
  baseline); graph adjustments matter only when they would CHANGE
  which items enter top-K.

On a 50-fact / 26-query dataset, most queries have well-separated
top-K scores — graph signal doesn't change the ranking. The v0.5
locality benefit doesn't have room to manifest.

## What this means for the product story

**Revised claim (honest)**: Locality filter is **safe** to enable
for retrieval workloads — it doesn't hurt on any seed/alpha tested,
and helps recover from rare alpha=0.30 failure modes.

**What it doesn't do**: improve peak retrieval accuracy over
cosine baseline on this dataset.

**What to verify before making product claims**:
1. Longer retrieval task (LoCoMo: 5000+ turns, 1986 queries) —
   does locality help on a real-world workload?
2. Continual-learning style workload (not just store + query) where
   the graph evolves over many interactions — locality's v0.5
   benefit was specifically about stable adaptation across non-
   stationary regimes.
3. Adversarial/corner-case queries where cosine alone fails — graph
   signal (with locality) might be the only thing that saves them.

## Suggested messaging

Until LoCoMo or similar validates the transfer:

- **NOT**: "SOMA with locality delivers better retrieval than vector-DB." (overclaim)
- **YES**: "SOMA's locality filter prevents graph re-rank from
  degrading retrieval at high alpha — safe to enable as default
  for agent-memory workloads where plasticity is active."

## Next steps in priority

1. **Run LoCoMo with locality on/off** (the real benchmark).
   Longer runtime (~30-60 min) but 50× more queries for clearer signal.
2. **Position-scramble** on v0.5 (already running in background):
   does the SPECIFIC initial position carry information, or any
   coherent metric works? Informs whether the v0.5 finding is
   about spatial structure or pattern-consistency.
3. **Consider DIFFERENT product use cases** where locality might
   matter more — continual learning, long-running deployments,
   post-hoc consolidation.

## Files

- `benchmarks/run_locality_ablation.py` — paired-seed runner
- `benchmarks/reports/locality_ablation_active_growth.md` — results
