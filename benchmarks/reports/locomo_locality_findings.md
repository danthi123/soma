# LoCoMo locality ablation — **NULL** (graph signal dominates baseline)

**Date:** 2026-04-19
**Commit:** 09caabd (runner), bhnyouw42 (run)
**Data:** `benchmarks/reports/locomo_locality.md`
**Dataset:** LoCoMo — 10 conversations, 5882 turns, 1982 queries

## Headline

| System | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| soma-flat (cosine only)        | 0.098 | 0.238 | 0.285 |
| soma-graph-no-locality (α=0.3) | 0.098 | 0.238 | 0.285 |
| soma-graph-locality (α=0.3)    | 0.098 | 0.238 | 0.285 |
| chroma                         | 0.095 | 0.232 | 0.279 |

**All three SOMA variants produce identical Recall@k to 3 decimal
places.** The locality filter (and the graph re-rank itself) makes
zero difference on LoCoMo at the ranks measured.

## Interpretation

Both the graph re-rank AND the locality filter are effectively
vacuous on LoCoMo because:

1. **Cosine baseline is weak on LoCoMo.** R@5 of 0.238 means the
   sbert cosine already fails to retrieve evidence for 76% of
   queries. There isn't room for graph re-rank to HELP on most
   queries — the evidence isn't in the top-50 cosine-similar items
   to begin with.

2. **Graph re-rank at α=0.3 doesn't shift top-K.** Even when graph
   activations differ across locality variants, the weighted
   blending with cosine isn't aggressive enough to change WHICH
   items end up in the top-10 list.

3. **Adversarial LoCoMo structure.** Real conversations have
   repeated mentions, pronoun resolution, temporal reasoning — all
   of which the graph substrate's 32-dim TextEncoder doesn't
   capture well. The graph contribution is close to noise on this
   workload regardless of locality.

Contrast with the v0.5 prediction task: there the graph IS the
mechanism (edges pass activation directly into prediction). Here
the graph is a secondary re-ranking signal on top of a stronger
cosine baseline.

## What this means for the product

The v0.5 locality finding **does not transfer to LoCoMo retrieval
in a measurable way**.

Honest framing:
- **NOT**: "SOMA's locality filter improves agent-memory retrieval."
- **YES**: "Locality filter is neutral on agent-memory retrieval
  workloads (no harm, no improvement at measured Recall@k)."
- **STILL TRUE**: "Locality filter is the correct plasticity design
  principle on SOMA's native prediction substrate." (v0.5 finding.)

## Why the category breakdown is zeros (report bug)

The per-category R@5 table shows all zeros with integer labels (1-5)
instead of category names (single-hop, multi-hop, etc.). This is a
bug in `run_locomo_locality.py` — it iterates over `CATEGORY_NAMES`
as a dict (yielding integer keys) instead of iterating over its
values. The underlying data in `result.recall_by_category` is keyed
by category NAME (string). The headline R@k is correctly computed.

Bug fix would be to iterate `CATEGORY_NAMES.values()` or use a list
of names from the run_locomo helper. Not fixing now because the
headline null makes the category breakdown moot — all cells would
be the same across variants.

## Where graph + locality might still matter

Not tested in this run, but theoretically candidates:
- **Longer-running agents**: LoCoMo conversations are finite (~588
  turns each). A product deployment with 100K+ memories may show
  different dynamics.
- **QA accuracy** (not just retrieval recall): the LoCoMo paper
  measures answer quality via LLM-as-judge. Graph signal might
  disambiguate among retrieved items.
- **Multi-hop / temporal queries**: where graph structure should
  shine — but the broken category breakdown hides this signal.
- **Adversarial alpha**: the synthetic-retrieval study showed
  locality prevents catastrophic failure at α=0.3 on one seed.
  LoCoMo didn't surface that failure, but it's a safety margin.

## Recommendation

1. **Ship `SOMAConfig.memory_layer()` preset with locality ON**
   (done in commit pending). It's provably safe (neutral on LoCoMo,
   downside protection on synthetic retrieval, strong positive on
   v0.5). Default settings shouldn't hurt any known workload.
2. **DO NOT claim locality improves retrieval in marketing copy.**
   The data doesn't support it.
3. **Investigate Direction 4a (LLM-distilled projections)** as the
   next research arc for making graph re-rank useful on retrieval.
   Problem isn't locality — it's that the random-projection
   substrate doesn't give graph re-rank semantic signal to work
   with. Direction 4a provides that.

## Files

- `benchmarks/run_locomo_locality.py` — runner (has minor report bug)
- `benchmarks/reports/locomo_locality.md` — raw results
