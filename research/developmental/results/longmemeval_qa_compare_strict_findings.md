# LongMemEval QA with strict-answer prompting — clean F1 numbers

**Status:** Confirmed. With strict "answer in 1-5 words" prompting
that neutralizes the token-F1 verbosity penalty, SOMA hybrid
delivers **+23% F1 overall** on the full LongMemEval small (N=500)
— up from the +10% we saw with verbose prompting. The advantage
widens on multi-session and temporal-reasoning questions where
verbose prompting was hiding the SOMA lift.

**Date:** 2026-04-20.
**Runner:** `run_qa_compare.py --strict-prompt`.
**Scope:** LongMemEval small, full N=500, all 6 question types.
**Embedder:** sbert all-MiniLM-L6-v2 on CPU.
**LLM:** qwen3.5:4b-q8_0 via Ollama, T=0, max_context_tokens=3800.

## Headline

| Mode | F1 | EM | R@5 |
| --- | ---: | ---: | ---: |
| chroma_cosine (strict) | 0.2994 | 0.1960 | 0.932 |
| **soma_hybrid (strict)** | **0.3677** | **0.2420** | **0.980** |

**+22.8% F1 relative, +23.5% EM relative.** Roughly a quarter of
SOMA's answers are now exact matches to the gold string.

## Why strict prompting changes the story

Earlier N=500 run with verbose prompt ("Be concise and direct")
showed:
- chroma F1 = 0.148, soma F1 = 0.164 — overall SOMA lift +10%

With strict prompt ("Answer with ONLY the specific fact in 1-5
words. No explanation, no preamble."), on the same 500 items:
- chroma F1 = 0.299, soma F1 = 0.368 — overall SOMA lift **+23%**

Both systems' absolute F1 jumps ~100%. That's because token-F1
normalization penalizes the LLM adding "Based on the conversation
history, your..." preamble before the actual answer. The strict
prompt stops the LLM from doing that.

More importantly, **the SOMA advantage widens** with strict
prompting on the hardest question types:

| Type | N | verbose lift | strict lift |
| --- | ---: | ---: | ---: |
| single-session-user | 70 | +49% | **+59%** |
| multi-session | 133 | +6% | **+36%** |
| temporal-reasoning | 133 | +3% | **+23%** |
| knowledge-update | 78 | +5% | **+11%** |
| single-session-assistant | 56 | -5% | ~tie |
| single-session-preference | 30 | -1% | ~tie |

The multi-session and temporal-reasoning numbers are the big shift.
Hypothesis: with verbose prompting, a 4B LLM uses some of its output
budget on preamble and loses reasoning capacity. Strict prompting
gives the LLM more headroom to actually USE the better retrieved
evidence.

## Outlier: single-session-preference

Both systems score F1 ≈ 0.03 on this type. Inspection shows the gold
answers are verbose meta-descriptions:

| Gold | Our strict prediction |
| --- | --- |
| "The user would prefer responses that suggest resources specifically tailored to..." | "Adobe Premiere Pro's official documentation" |
| "The user would prefer suggestions of Sony-compatible accessories..." | "Gitzo GT3543LS tripod" |
| "The user would prefer suggestions of hotels in Miami that offer great views..." | "I don't know" |

The questions ask for preference synthesis; the gold is itself a
synthesized preference statement. Our prediction is a specific
recommendation that's factually derived from the retrieved text but
scores badly on token-F1 against a sentence-long gold.

This is a benchmark-mismatch issue, not a retrieval or reasoning
failure. A prompt per question type, or an LLM-judge metric, would
give a fairer score here. Out of scope for current run; N=30 is a
small slice of the 500 items so the overall average isn't much
affected.

## How to reproduce

```bash
python -m benchmarks.industry.longmemeval.run_qa_compare \
  --variant small \
  --modes chroma_cosine soma_hybrid \
  --model qwen3.5:4b-q8_0 \
  --out-suffix=_n500_strict \
  --sbert-device cpu \
  --strict-prompt
```

## Positioning update

Replaces the verbose-prompt +10% overall / +49% single-session-user
claim with +23% overall / +59% single-session-user / +36%
multi-session / +23% temporal-reasoning. All numbers on the SAME
500 items, SAME LLM, SAME context budget — the only variable is
retrieval strategy.

## Cross-reference

- **Retrieval-only benchmark** (`longmemeval_retrieval_findings.md`):
  SOMA hybrid R@5 = 0.980 vs chroma+rerank 0.936. We're now measuring
  the cost of each R@5 miss in QA terms.
- **Model-size invariance** (`longmemeval_qa_compare_qwen9b_findings.md`):
  same 200 items on 4b vs 9b showed identical +25-27% SOMA lift,
  confirming this isn't a model-specific effect.
- **Verbose-prompt baseline** (`longmemeval_qa_compare_findings.md`):
  the earlier analysis that showed only +10% overall F1 — kept for
  honesty about the prompting artifact.

## Files

- `benchmarks/industry/longmemeval/results/qa_compare_{chroma_cosine,soma_hybrid}_n500_strict.{json,jsonl}`
- `benchmarks/industry/longmemeval/results/qa_compare_n500_strict.log`
- `benchmarks/industry/longmemeval/results/qa_compare_summary_n500_strict.md`
