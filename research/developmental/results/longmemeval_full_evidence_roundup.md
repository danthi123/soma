# LongMemEval — full evidence roundup

**Date:** 2026-04-20.
**Status:** SOMA hybrid retrieval wins on LongMemEval under every
metric we've measured. Lift is cross-model and cross-embedder.

## Headline

| Metric | SOMA hybrid | baseline | Relative lift |
| --- | ---: | ---: | ---: |
| F1 (qwen4b strict vs chroma_cosine, N=500) | 0.368 | 0.299 | **+22.8%** |
| F1 (qwen9b strict vs chroma_cosine, N=500) | — | — | **+22.0%** |
| **F1 (Claude strict vs chroma_RERANK, N=500)** | **0.416** | **0.363** | **+14.7%** |
| EM (qwen4b strict, N=500) | 0.242 | 0.196 | **+23.5%** |
| EM (Claude strict, N=500) | 0.304 | 0.262 | **+16.0%** |
| Judge-acc (qwen4b preds by qwen4b, N=500) | 0.440 | 0.360 | **+22.2%** |
| **Judge-acc (qwen4b preds by Claude, N=500)** | **0.418** | **0.338** | **+23.7%** |
| Judge-acc (qwen9b preds by qwen4b, N=500) | 0.434 | 0.360 | **+20.6%** |
| hit@5 (rank probe, N=500) | 0.980 | 0.932 | +5.1% |
| **rank1_frac** (rank probe, N=500) | **0.963** | **0.833** | **+15.6%** |
| mean rank (rank probe, N=500) | 1.16 | 1.32 | −12.1% |

Note: the Claude row's baseline is `chroma_rerank` (cosine +
cross-encoder reranker) — a harder baseline than the `chroma_cosine`
used in the qwen rows. The rerank closes most of the ranking gap
so SOMA's lift is smaller but still positive; the retrieval
mechanism is the same.

## Mechanism decomposition (N=500 qwen4b)

Partition by (chroma_hit, soma_hit) on F1:

| Cell | N | sum(F1 delta) | contribution |
| --- | ---: | ---: | ---: |
| (0,1) SOMA rescues | 30 | +11.49 | **34% (recall)** |
| (1,1) both hit | 460 | +22.55 | **66% (ranking)** |
| (1,0) chroma alone | 6 | +0.11 | ~0% |
| (0,0) both miss | 4 | 0 | 0% |

- **34% of lift is recall**: 30 items where only SOMA finds gold.
  BM25 catches keyword-match queries cosine misses.
- **66% of lift is ranking**: 460 items where both systems find
  gold in top-5, but SOMA promotes it to rank 1-2 (truncation-
  safe) while chroma leaves it at rank 4-5 (truncated out of 3.8K
  context).

Truncation is direct: chroma's IDK rate climbs with gold rank
(rank 1 → 17% IDK, rank 3 → 100% IDK).

## Direct rank evidence (N=500 paired)

Per-type rank-1 lift (fraction of items where gold is at rank 1):

| Type | N | chroma rank1 | SOMA rank1 | Δ |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 0.543 | 0.871 | **+0.329** |
| knowledge-update | 78 | 0.808 | 0.962 | +0.154 |
| temporal-reasoning | 133 | 0.752 | 0.850 | +0.098 |
| multi-session | 133 | 0.865 | 0.910 | +0.045 |
| single-session-assistant | 56 | 0.982 | 1.000 | +0.018 |
| single-session-preference | 30 | 0.567 | 0.567 | +0.000 |

Pair-level distribution: SOMA rescues **58/78** (74%) of chroma's
rank-2-5 items into rank 1-2. Net items where SOMA has gold
strictly earlier in context: **+71**.

## Cross-model validation (qwen9b)

Re-running the same 500 items with qwen3.5:9b-q8_0 instead of 4b
gives essentially identical lifts:

- F1: +22.0% (vs +22.8% at 4b)
- EM: +19.4% (vs +23.5% at 4b)
- Judge: +20.6% (vs +22.2% at 4b)

Decomposition stable at 37/63 recall/ranking (vs 34/66 at 4b). The
lift is **retrieval-mechanical, not LLM-specific**.

9b behaviour change: qwen9b IDKs more often on temporal-reasoning
under strict-prompt (70/133 vs 39/133 at 4b), so F1-lift on
temporal collapses but rank-1-lift is unchanged.

## Cross-LLM validation (Claude Sonnet as answerer, N=500)

Identical retrieval, swap the LLM. SOMA's retrieval still wins:

| Answerer | chroma baseline | SOMA hybrid | SOMA lift |
| --- | ---: | ---: | ---: |
| qwen3.5:4b (vs chroma_cosine) | F1 0.299 | F1 0.368 | **+22.8%** |
| Claude (vs chroma_rerank) | F1 0.363 | F1 0.416 | **+14.7%** |

On `single-session-user` specifically — the question type where
ranking-position matters most — Claude+SOMA shows +54% F1 lift over
Claude+chroma_rerank (0.503 → 0.775). This parallels the qwen4b
+59% lift pattern and is the sharpest per-type signal of retrieval
ranking mattering for end-to-end QA. See
`longmemeval_claude_runner_findings.md`.

Infrastructure: Claude runs via the operator's Unraid
`claude-code-runner` Docker image on the Claude Max subscription —
no per-request API charges, only consumes the subscription budget.

## Cross-embedder validation (mxbai-embed-large, N=50)

Swapping SBERT for mxbai-embed-large (via ollama):

| Embedder | Mode | rank1_frac |
| --- | --- | ---: |
| SBERT | chroma cosine | 0.833 |
| SBERT | SOMA hybrid | 0.963 |
| mxbai | chroma cosine | 0.620 |
| mxbai | SOMA hybrid | 0.900 |

SOMA's lift **widens** (45% relative) with mxbai because mxbai's
512-token input limit hurts base cosine retrieval, leaving more
for BM25's keyword contribution to rescue.

## Per-question-type F1 lift (qwen4b strict, N=500)

| Type | N | chroma F1 | SOMA F1 | rel lift |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 0.482 | 0.767 | **+59%** |
| multi-session | 133 | 0.113 | 0.154 | +36% |
| temporal-reasoning | 133 | 0.188 | 0.231 | +23% |
| knowledge-update | 78 | 0.408 | 0.451 | +11% |
| single-session-assistant | 56 | 0.772 | 0.766 | ~tie |
| single-session-preference | 30 | 0.031 | 0.031 | tie |

## What's not a SOMA win

- **Graph rerank**: off by default. LoCoMo N=60 showed pure-graph
  alpha=1.0 F1=0.127 vs cosine alpha=0.0 F1=0.213 (−40%). Hybrid
  (alpha=0.3 BM25+cosine) wins, NOT graph. See
  `locomo_graph_rerank_findings.md`.
- **Plastic graph activation**: on the synthetic 10-topic benchmark
  (v1), plastic and frozen SOMA both produced identical results in
  the default hybrid retrieve path because the graph is not read
  when `hybrid_alpha` is passed. See
  `plastic_graph_activation_findings.md`.

## Cross-reference

- `longmemeval_causation_findings.md` — partition decomposition
- `longmemeval_rank_delta_findings.md` — direct paired rank measurement
- `longmemeval_judge_findings.md` — 4b judge on 4b predictions
- `longmemeval_judge_findings_qwen9b.md` — 4b judge on 9b predictions
- `longmemeval_qa_compare_qwen9b_strict_findings.md` — 9b F1 results
- `longmemeval_temporal_deep_dive.md` — qwen9b temporal IDK investigation
- `longmemeval_mxbai_cross_embedder.md` — mxbai-embed-large replication
- `longmemeval_claude_runner_findings.md` — Claude-as-answerer N=500 validation
- `longmemeval_judge_findings_claude.md` — Claude-as-judge +23.7% at N=500
- `longmemeval_alpha_sweep_findings.md` — α=0.30 default validation
- `plastic_graph_activation_findings.md` — plastic graph closed-as-null
- `locomo_graph_rerank_findings.md` — LoCoMo graph rerank negative finding
- `locomo_direction4b_findings.md` — Direction 4b closed-as-null on LoCoMo
