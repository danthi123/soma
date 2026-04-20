# LoCoMo rank probe — SOMA hybrid vs chroma cosine at N=1974 QAs

**Status:** **Huge SOMA lift** on LoCoMo replicates the LongMemEval
result at larger magnitudes. rank1 lift +88.7%, hit@5 lift +45.4%.

**Date:** 2026-04-20.
**Runner:** `benchmarks/industry/locomo/rank_probe.py`.
**Data:** 10 LoCoMo conversations, 1974 paired QA pairs, all 5
categories. Turn-level retrieval (each conversation turn is one
memory entry; evidence dia_ids mark gold turns).

## Headline

| Mode | hit@5 | rank1_frac | mean rank |
| --- | ---: | ---: | ---: |
| chroma cosine | 0.388 | 0.171 | 2.14 |
| **SOMA hybrid** | **0.564** | **0.322** | **1.83** |
| **lift** | **+45.4%** | **+88.7%** | — |

## Per-category rank-1 fraction

| Category | N | chroma rank1 | SOMA rank1 | rel lift |
| --- | ---: | ---: | ---: | ---: |
| adversarial | 446 | 0.101 | 0.321 | **+218%** |
| temporal | 321 | 0.218 | 0.393 | +80% |
| open_domain | 829 | 0.210 | 0.358 | +70% |
| single_hop | 282 | 0.138 | 0.209 | +51% |
| multi_hop | 96 | 0.094 | 0.115 | +22% |

All categories benefit; the largest gains are on **adversarial**
(+218%) and **temporal** (+80%).

## Per-category hit@5

| Category | N | chroma hit@5 | SOMA hit@5 | rel lift |
| --- | ---: | ---: | ---: | ---: |
| adversarial | 446 | 0.291 | 0.574 | +97% |
| temporal | 321 | 0.449 | 0.632 | +41% |
| open_domain | 829 | 0.438 | 0.620 | +42% |
| single_hop | 282 | 0.387 | 0.401 | +3% |
| multi_hop | 96 | 0.208 | 0.292 | +40% |

## Why adversarial and temporal spike

LoCoMo's **adversarial** category has questions with no direct
answer in the context (gold answer is often "I don't know" or
specific negative). Gold evidence is sparse — often just a single
turn. With sparse evidence, ranking matters more than recall; BM25
promotes the right keyword-match turn into the top-k where cosine
loses it among many similar-topic turns.

**Temporal** questions like "When did X happen?" share rare date
keywords with the target turn (e.g. "on June 15"). BM25's
term-specificity matches the date token directly; cosine smears
across all time-related turns.

## Relation to LongMemEval

Both benchmarks show the same underlying mechanism: **hybrid
retrieval (BM25 + cosine) beats pure cosine when the corpus is
conversational**. The effect size varies with corpus structure:

| Benchmark | rank1 lift | hit@k lift | N |
| --- | ---: | ---: | ---: |
| LongMemEval (k=5, session-level) | +15.6% | +5.1% | 500 |
| LoCoMo (k=5, turn-level) | **+88.7%** | **+45.4%** | 1974 |
| mxbai embedder (k=5, session-level) | +45.2% | +19.0% | 50 |

LoCoMo turn-level retrieval is harder (smaller units, more
distractors, adversarial category), so BM25's keyword boost
contributes more.

## Files

- `benchmarks/industry/locomo/rank_probe.py` — runner
- `research/audit/results/locomo_rank_probe_chroma_cosine_n10.jsonl`
- `research/audit/results/locomo_rank_probe_soma_hybrid_n10.jsonl`
- `research/audit/results/locomo_rank_probe_chroma_cosine_n10_summary.json`
- `research/audit/results/locomo_rank_probe_soma_hybrid_n10_summary.json`

## Caveats

- N=10 conversations (LoCoMo is 10-conv dataset; this IS full
  LoCoMo). 1974 paired QA pairs.
- Retrieval-only measurement — no LLM in this test. End-to-end QA
  on LoCoMo not yet measured (needs LLM runs).
- LoCoMo dataset is smaller and more adversarial than LongMemEval;
  effect sizes are not directly comparable, but the *direction* is
  consistent.
