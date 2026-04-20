# LongMemEval LLM-judge — Claude as judge on qwen4b predictions

**Status:** Confirmed with a stronger judge. The SOMA +22.2% judge-accuracy
lift on qwen4b predictions reproduces under Claude judge at **+23.7%**
(0.418 vs 0.338) on identical N=500 data. Per-type pattern broadly agrees
with qwen4b judge; temporal-reasoning lift widens.

**Date:** 2026-04-20.
**Judge:** Claude (via Unraid `claude-code-runner` Docker on Claude Max
OAuth token — no per-request API charges).
**Data:** both `qa_compare_{chroma_cosine,soma_hybrid}_n500_strict.jsonl`
rescored via `judge_predictions.py --judge-provider claude_runner`.
**Runtime:** ~29 min for chroma + ~28 min for soma on Unraid docker;
parallelised near-end.

## Headline

| Mode | F1 | qwen4b judge | **Claude judge** |
| --- | ---: | ---: | ---: |
| chroma_cosine | 0.2994 | 0.3600 | **0.3380** |
| **soma_hybrid** | **0.3677** | **0.4400** | **0.4180** |
| SOMA lift | **+22.8%** | **+22.2%** | **+23.7%** |

Three independent metrics, three near-identical deltas. The +22-24%
lift survives:
- Token-F1 (overlap-based, rewards extractiveness)
- qwen4b judge (weak judge, judges its own predictions)
- Claude judge (strong judge, independent of responder)

## Per-type breakdown (Claude judge)

| Type | N | F1 lift | qwen4b-judge lift | **Claude-judge lift** |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 70 | +59% | +59% | **+63%** |
| **temporal-reasoning** | 133 | +23% | +48% | **+58%** |
| knowledge-update | 78 | +11% | +11% | +12% |
| multi-session | 133 | +36% | +4% | +9% |
| single-session-assistant | 56 | ~0% | +2% | +2% |
| single-session-preference | 30 | +2% | +20% | +0% |

Notable agreements:
- **single-session-user** (the ranking-dominant type) shows +59-63% across
  all three metrics. Robust signal.
- **temporal-reasoning** F1 said +23%, qwen4b judge said +48%, Claude
  judge says **+58%**. Both judges reveal F1 was systematically under-
  scoring SOMA on temporal questions (paraphrase penalty). Claude judge
  is the strongest signal here.
- **multi-session** F1 said +36%, qwen judges say +4-9%. F1 was over-
  rewarding overlap on partial-wrong syntheses — confirmed by both
  judges.

One disagreement:
- **single-session-preference**: qwen4b judge recovered a +20% lift that
  F1 penalty was hiding; Claude judge sees neither (+0%). This category
  has verbose-paraphrase gold answers and both systems produce short
  extractive answers that don't match. Claude judge is stricter about
  "same factual answer" when gold is a complete sentence.

## Where judge and F1 disagree (N=500 Claude judge)

| Mode | judge=1 & F1<0.5 (F1 under-scored) | judge=0 & F1>0.5 (F1 over-rewarded) |
| --- | ---: | ---: |
| chroma_cosine | 39 | 7 |
| soma_hybrid | 45 | 8 |

Both systems have ~40-45 items where Claude judge credits the response
but F1 is below 0.5. The judge catches paraphrase + format variance
consistently across modes.

## Why this matters for credibility

The original +22.8% F1 claim used a lexical metric. The qwen4b judge
self-judgement already closed the "maybe F1 is misleading" escape hatch
but left open "maybe qwen4b is a weak judge that just mirrors F1". Claude
judge closes that: **a strong, independent judge sees the same lift to
within a percentage point**.

The triple agreement (F1 22.8% / qwen-judge 22.2% / Claude-judge 23.7%)
is about as robust as a retrieval-driven QA lift claim can be without
going to human annotation.

## Files

- `benchmarks/industry/longmemeval/results/qa_compare_chroma_cosine_n500_strict_judged_claude.{jsonl,_summary.json}`
- `benchmarks/industry/longmemeval/results/qa_compare_soma_hybrid_n500_strict_judged_claude.{jsonl,_summary.json}`
- `scripts/analysis/compare_judge_accuracy.py` — aggregator (used with
  `--suffix _n500_strict_judged_claude`)

## Cross-reference

- `longmemeval_judge_findings.md` — qwen4b judge (same predictions)
- `longmemeval_judge_findings_qwen9b.md` — qwen4b judge on qwen9b preds
- `longmemeval_claude_runner_findings.md` — Claude as answerer N=500
- `longmemeval_full_evidence_roundup.md` — consolidated headline
