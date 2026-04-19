# Direction 4a Phase 5: Multi-hop / temporal stress test

**Date:** 2026-04-19
**Parent:** `docs/plans/2026-04-19-direction-4a-design.md` §Phase 5
**Status:** draft — executable after Phase 3 produces category-level LoCoMo results

## Motivation

The core value proposition in Direction 4a's design doc is that SOMA
should add value **standard LLMs without SOMA don't**. The most
defensible version of that claim is: SOMA's graph encodes structural
associations (co-activation, temporal patterns) that cosine similarity
on single-vector embeddings cannot capture.

LoCoMo has categories that specifically stress these:
- **multi-hop**: answer requires chaining two memories
- **temporal**: answer requires understanding when events happened
- **adversarial**: answers designed to trip up naive retrieval

Phase 5 analyzes per-category breakdown to see whether SOMA-distilled
wins specifically on these "structurally-hard" categories.

## Hypothesis

For `soma-graph-distilled+local` vs `chroma-mxbai`, the per-category
delta should be:

| Category | Expected Delta | Reasoning |
|----------|----------------|-----------|
| single-hop | ≈ 0 | cosine similarity suffices; graph is redundant |
| multi-hop | **+0.03 to +0.10** | Two-step chaining needs structural assoc |
| temporal | **+0.02 to +0.08** | Graph edges encode co-occurrence by position |
| open-domain | ≈ 0 to +0.02 | General knowledge; graph may or may not help |
| adversarial | ≈ 0 to +0.03 | Depends on adversary strategy |

If multi-hop / temporal deltas are positive and meaningfully larger
than single-hop, we have the differentiation story. If deltas are
similar across categories, SOMA's advantage (if any) is embedding-
grade improvement rather than structural.

## Dependencies (from Phase 3)

- Phase 3 runner must execute with CATEGORY_NAMES iteration bug fixed
  (commit `5e8bc93`).
- Phase 3 must produce `recall_by_category` populated with non-zero
  values across all 5 categories.
- Phase 3 JSON output must serialize per-category R@k for analysis.

Both are now in place in `benchmarks/run_locomo_distill.py` commit
`ba4883d` (bug fix with regression tests).

## Analysis method

1. Load Phase 3 JSON output
2. For each category, compute (soma-distilled R@5) - (chroma-mxbai R@5)
3. Compute bootstrap CI on the deltas (100 resamples of queries)
4. Plot: bar chart of deltas per category with CI bars
5. Report: table + figure + narrative

## Decision gates

### Ship gate (primary)

- Multi-hop delta > +0.03 with CI not straddling 0
- Temporal delta > +0.02 with CI not straddling 0
- Single-hop delta approximately 0 (not negative)

→ Position: "SOMA gives structural-retrieval advantage on multi-hop
and temporal queries without hurting single-hop — the only memory
layer that preserves semantic + structural signals jointly."

### Iterate gate (weak positive)

- One of {multi-hop, temporal} delta is positive but the other is flat
- Or both deltas positive but CI straddles 0

→ Investigate: maybe alpha needs tuning, maybe locality cutoff needs
adjustment for semantic-space-distance (post-Phase 4).

### Stop gate (null)

- All category deltas within ±0.02 noise floor
- No clear pattern of structural-category advantage

→ Honest null. Distillation might improve embedding utility but doesn't
unlock structural advantage; the §5 retrieval ceiling from the paper
persists. Document, pivot to a different differentiation story (e.g.,
continual learning, not retrieval).

## Extensions (if results are positive)

- Multi-hop ablation: manually construct queries that require exactly 2
  evidence turns with known structural relationship (same topic, same
  speaker, etc.). Compare how distilled graph vs flat cosine handles
  these.
- Temporal probe: intercept the retrieval path and check whether the
  top-K chunks are the TEMPORALLY-CLOSEST vs the MOST-SEMANTICALLY-
  SIMILAR. If graph-distilled prefers temporal neighbors more than
  chroma, that's evidence of structural bias in a good way.
- Retrieval latency: graph-distilled will be slower than chroma. By how
  much? Quantify the latency tradeoff so the product framing is honest.

## Deliverables

- `research/developmental/results/env_sequence_v05_distill_phase5_analysis.md`
- Figure (PNG) of per-category delta with CIs
- Extended paper §5 (if positive) with the differentiation story
- Product positioning update (if positive): lead with multi-hop / temporal
