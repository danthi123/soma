# Developmental SOMA: Findings and Direction

**Date:** 2026-04-18
**Status:** Ceiling-confirmation phase complete; strategic decision pending

---

## Executive Summary

Over the past ~72 hours of autonomous research, we've run 14 distinct
experiments probing whether SOMA's graph memory contributes measurable
retrieval signal beyond a pre-trained embedding baseline (all-MiniLM-
L6-v2 + cosine). The answer: **yes, but the signal is too small and
too slice-sensitive to ship as a retrieval enhancement.** The ceiling
is architectural — any retrieval signal derived from "which nodes
fire together" plateaus around +2 hits per 500 queries on LoCoMo
(~0.4% absolute improvement over VecDB-only).

What this means:
- The MemoryLayer + agent-memory pivot (2026-04-15) is not the right
  product angle for SOMA. We can't beat VecDB meaningfully on the
  canonical benchmarks.
- SOMA's brain-inspired mechanisms (neurogenesis, consolidation,
  Hebbian learning, structural plasticity) were not designed for
  static retrieval; they were designed for developmental adaptation.
  The retrieval evaluation was a pragmatic fit that doesn't play to
  their strengths.
- The 2026-04-17 strategic review
  ([soma-direction-review.md](soma-direction-review.md)) proposed four
  options. The retrieval-focused Option C implicitly won by default
  over the past week, but the data now says it's a dead end in its
  current form.

---

## What We Tested (and Ruled Out)

### Ceiling-confirmation experiments (2026-04-18)

| Phase | Experiment | Result |
|-------|-----------|--------|
| 3 | Confidence-gated hybrid (baseline) | **+3 hits/100 on tuned slice** (breakthrough) |
| 4 | Encoder fine-tuning via SOMA loss | 0 grad updates (SOMA detaches) |
| 4b | Contrastive encoder FT via graph topology | Catastrophic: -13 hits (encoder destroyed) |
| 5 | Semantic lens (learned fp mapping) | Neutral |
| 6 | Adaptive gating (scale w by fp quality) | -2 wins vs fixed |
| 7 | min_active sweep | min=3 Pareto optimal |
| 8 | Learnable diversification projections | +1 (within noise) |
| 9 | Associator count scaling {8, 32, 64} | **n=8 Pareto optimal** (scaling hurts) |
| 10 | Multi-slice held-out validation (500 queries) | **+2 total**, wins/losses 32/34 — coin flip |
| 11 | Shuffle diagnostic (real vs randomized fingerprints) | Real +2 > max shuffled +1 (signal real but weak) |
| 12 | Per-query category attribution | No clean category signal |
| 13 | Topology signal (Jaccard over node overlap) | Functionally equivalent to fingerprint |
| 14 | LongMemEval gated-hybrid validation | **-1 delta, 3W/6L** — slightly hurts; ceiling confirmed cross-benchmark |

### Key conclusions from the data

1. **Signal is real but tiny.** Shuffle diagnostic: real total delta +2
   exceeds all 5 shuffled samples (max +1). The graph's structural
   binding of memory to fingerprint does carry information.
   **But it doesn't generalize:** on LongMemEval, the gated hybrid
   scores -1 vs VecDB (3W/6L), confirming the mechanism has no
   cross-benchmark advantage.

2. **Signal ceiling is architectural.** Both fingerprint similarity
   and topology Jaccard plateau at the same +2/500 delta. Both derive
   from the same 3 lateral-inhibition-winner nodes. Scaling the graph
   (more associators) *hurts* rather than helps — extra confidence
   fires on mismatches.

3. **No shippable subset.** Per-query attribution across 6 categories
   shows cross_entity and counting have 2-3:1 win ratios as expected,
   but absolute N is tiny (31, 36) and absolute delta near zero. The
   biggest contribution (+5) is in "other," a 44% catch-all with no
   semantic pattern.

4. **Bottleneck is signal quality, not vocabulary.** Random-projection
   fingerprints are structurally diverse but semantically arbitrary
   (consistent with Phase 4b contrastive-FT failure: training the
   encoder toward graph topology catastrophically destroys it).

### What still works

- **Consolidation helps QA** (`c553a66`, +45% on synthetic)
- **Multi-session development** (`6a0a822`, +275% graph growth across 3 sessions)
- **Phase 1.3 graph reranking on LongMemEval** (alpha=1.0 > alpha=0 by +15% F1 on 20 items) — though unvalidated at larger scale

These are not retrieval-layer wins; they're demonstrations that the
developmental mechanisms work as designed on tasks they're suited for.

---

## Updated Strategic Options

Revisiting the 2026-04-17 four-option review with current data:

### Option A: Open-Ended Learning Environment (grid world / survival)
**Weight: HIGH**

Now the strongest candidate. SOMA's mechanisms were literally
designed for this — neurogenesis for new concepts, consolidation for
overnight reorganization, pruning for outdated strategies, critical
periods for staged learning.

- **Pros:** Directly tests what SOMA is built for. Clear metrics
  (survival time, adaptation speed after rule changes). No LLM
  dependency. Could produce a strong, differentiated demo.
- **Cons:** Need to build an environment from scratch. New metric
  stack. Harder to "productize" in a conventional sense.
- **Effort:** Medium-high. 2-3 weeks for a minimal env + eval loop.

### Option B: Adaptive Processing Layer (anomaly detection / edge)
**Weight: LOW**

Big pivot, different data modality, no existing infrastructure. Not
clearly aligned with where we've built competence.

### Option C: Transformer Augmentation (personal AI / user model)
**Weight: LOWERED** (from implicit-default → bench-fail)

Our implicit pursuit of this over the past week. The retrieval-
augmentation framing is now disproven at current architecture. A
"user model / adaptation over time" framing might still work but
would need fundamentally different evaluation (personalization
quality over weeks, not single-session retrieval).

### Option D: Research Paper
**Weight: HIGH** (parallel to A)

The negative results are publishable. Specific claims:
1. Brain-inspired graph memory adds a weak but statistically non-
   random retrieval signal (~0.8% absolute over random mapping).
2. The signal ceiling is architectural: it derives from lateral-
   inhibition winners, not from semantic structure in the projections.
3. Training the encoder to match graph topology causes catastrophic
   forgetting (graph topology is structurally diverse but semantically
   arbitrary).
4. Consolidation improves QA on synthetic data by +45%.
5. Developmental mechanisms survive multi-session without forgetting.

The paper doesn't depend on Option A succeeding — it's worth writing
regardless.

---

## Recommendation

**Combined path: D in parallel with A.**

- **Option D (paper)** can start immediately with the data we have.
  The narrative is clean: pragmatic-pivot led us to honest ceiling-
  discovery; here's what works and what doesn't.
- **Option A (open-ended env)** is the right forward direction for
  SOMA's actual strengths. Would be a ~2-3 week bootstrap; the
  existing graph-execution core, config, and memory systems
  transfer directly. Evaluation would need design from scratch.

If that's too broad, a smaller-scope version: **D first, decide A
after D drafting clarifies what the existing code supports.**

---

## What I'd like your input on

This is an explicit strategic decision beyond my autonomy scope. The
tactical work to date is committed to main; the next commit cadence
depends on your call:

1. Do you want to pursue A (env build) + D (paper draft) combined?
2. A-only? D-only?
3. Pivot to something else (e.g., return to the agent-memory-layer
   angle and ship what we have, treating SOMA-as-developmental-
   substrate as separate research)?
4. Stop the Developmental PoC arc and fold it into the main SOMA
   agent-memory-layer roadmap?

I'll keep running any ceiling-confirmation experiments that are
cheap and diagnostic (e.g., the LongMemEval validation in flight),
and I can draft the paper outline without blocking on the above —
but I won't start building an env or restructuring code until you
pick a direction.
