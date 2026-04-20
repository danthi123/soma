# LoCoMo graph re-rank — findings (N=60)

**Status:** Graph rerank is net-negative on LoCoMo. Pure-graph
(`alpha=1.0`) scores −40% F1 relative to pure-cosine (`alpha=0.0`)
on 60 QA pairs across 4 categories.

**Date:** 2026-04-17 (re-analysed 2026-04-20).
**Runner:** `benchmarks/industry/locomo/evaluate_graph_rerank.py`.
**Raw data:** `research/audit/results/locomo_graph_rerank.json`.

## Headline

| Metric | alpha=0 (cosine) | alpha=1 (graph) | delta |
| --- | ---: | ---: | ---: |
| F1 | 0.2127 | 0.1270 | **−0.0857 (−40.3%)** |
| Wins (per item) | 27 | 12 | — |
| Ties | — | — | 21 |

At N=60 across 4 categories, pure-graph retrieval loses on 45% of
items and only wins on 20%. Even on ties, the graph-hit case returns
`I don't know` 28% of the time vs cosine's 22%.

## Per-category breakdown

| Category | N | F1 (cosine) | F1 (graph) | wins g/c/t |
| --- | ---: | ---: | ---: | ---: |
| temporal | 34 | 0.2542 | 0.1388 | 6/18/10 |
| single_hop | 20 | 0.1630 | 0.1163 | 5/8/7 |
| multi_hop | 4 | 0.0794 | 0.0286 | 0/3/1 |
| open_domain | 2 | 0.2695 | 0.2286 | 1/1/0 |

Temporal reasoning — the category where the N=20 pilot had shown a
modest alpha=1.0 > alpha=0.0 gain — reverses at N=60 with cosine
winning 18 vs 6.

## Why graph re-rank hurts (current formulation)

From `phase1_3_per_item_analysis.json`, the per-item losses on LoCoMo
come from two failure modes:

1. **Graph under-retrieves on proper nouns.** BPE sub-word tokens
   generalise too aggressively — "Caroline" and "Carlos" share
   subword fragments, so their graph activations end up nearby. On
   "What did Caroline research?" the graph pulls Caroline-adjacent
   sessions AND Carlos-adjacent sessions, diluting the top-k. Cosine
   on SBERT's full-word representation resolves this cleanly.
2. **Graph equally-weights topic bursts.** When a conversation has
   5 sessions about running and 15 about music, the graph learns
   music-heavy activations. Running-queries consult a music-biased
   graph and the few running sessions get pushed out by high-
   activation music sessions — a retrieval-neutral-but-graph-biased
   match.

## Relation to LongMemEval findings

LongMemEval's +22% F1 lift comes from the **hybrid BM25+cosine** path
(`hybrid_alpha=0.3`), which entirely bypasses the graph-rerank layer.
`MemoryLayer.retrieve()` branches: `hybrid_alpha=X` → pure hybrid;
`hybrid_alpha=None + graph_rerank_alpha=Y` → graph rerank; the two
are XOR.

So:

- **On LongMemEval**, SOMA wins via BM25 term-match promoting gold
  from rank 2-5 to rank 1-2 (see `longmemeval_rank_delta_findings.md`).
  Graph is not consulted.
- **On LoCoMo**, graph-rerank (tested in isolation with hybrid off)
  underperforms pure cosine.

Net product implication: the graph substrate ships and grows with
use, but it is not load-bearing for retrieval. That's fine given
it is off by default (`graph_rerank_alpha=0.0`). Activating it is a
research question (see `paper-draft.md` §5 and Direction 4 spatial
distillation), not a near-term product claim.

## Files

- `research/audit/results/locomo_graph_rerank.json` — raw per-
  conversation / per-QA data (N=60)
- `research/audit/results/phase1_3_graph_rerank_alpha.json` — 6-point
  alpha sweep on N=20 temporal subset
- `research/audit/results/phase1_3_per_item_analysis.json` — per-item
  wins analysis on N=20
- `benchmarks/industry/locomo/evaluate_graph_rerank.py` — runner
