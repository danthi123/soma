# qwen3.5:9b strict N=500 — SOMA lift is LLM-size-invariant at scale

**Status:** Confirmed at scale. With strict prompting and full
N=500, SOMA hybrid delivers **+22% F1** on qwen3.5:9b vs **+23% F1**
on qwen3.5:4b on the SAME items. The retrieval advantage is a
property of retrieval mechanics; swapping to a bigger responder
LLM shifts absolute F1 but not the delta.

**Date:** 2026-04-20.
**Runner:** `run_qa_compare.py --strict-prompt --model qwen3.5:9b-q8_0`.
**Scope:** LongMemEval small, full N=500.
**Embedder:** sbert all-MiniLM-L6-v2 on CPU.
**LLM:** qwen3.5:9b-q8_0 via Ollama, T=0, max_context_tokens=3800.

## Headline

| Mode | F1 | EM | R@5 |
| --- | ---: | ---: | ---: |
| chroma_cosine (strict) | 0.2847 | 0.1860 | 0.9320 |
| **soma_hybrid (strict)** | **0.3472** | **0.2220** | **0.9800** |

**+22% F1, +19% EM overall.** Identical +5pp R@5 to qwen4b run (same
retrieval). Absolute F1 is lower than qwen4b (0.285 vs 0.299 chroma,
0.347 vs 0.368 SOMA), consistent with qwen9b being more verbose even
under strict prompting — the verbosity effect documented in the
earlier `longmemeval_qa_compare_qwen9b_findings.md` persists at full
scale.

## Cross-LLM comparison (strict prompting, N=500)

| LLM | chroma F1 | SOMA F1 | SOMA lift | chroma EM | SOMA EM | EM lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.5:4b | 0.2994 | 0.3677 | **+22.8%** | 0.1960 | 0.2420 | **+23.5%** |
| qwen3.5:9b | 0.2847 | 0.3472 | **+22.0%** | 0.1860 | 0.2220 | **+19.4%** |

F1 lift sits at **+22.0% on 9b vs +22.8% on 4b** — within noise. The
"SOMA's retrieval advantage is model-size-invariant" finding from
the earlier N=200 verbose study now replicates at full N=500 with
strict prompting. SOMA's lift is a mechanical retrieval win, not a
model-specific artifact.

## Decomposition is stable across LLMs

Running the same partition analysis on both runs:

| Run | total lift | recall frac | ranking frac |
| --- | ---: | ---: | ---: |
| qwen3.5:4b strict | +34.15 | 34% | 66% |
| qwen3.5:9b strict | +31.27 | 37% | 63% |

The 34/66 recall/ranking split holds at 37/63 on 9b — essentially
unchanged. The ranking mechanism (SOMA places gold at rank 1-2 where
it fits in the 3.8K budget; chroma buries it at rank 4-5 where
truncation drops it) is LLM-independent.

## IDK rates — 9b more honest, asymmetry persists

| LLM | chroma IDK | SOMA IDK | chroma/SOMA ratio |
| --- | ---: | ---: | ---: |
| qwen3.5:4b strict | 30.4% | 18.2% | 1.67× |
| qwen3.5:9b strict | 45.0% | 35.8% | 1.26× |

Both systems' absolute IDK rates go UP on 9b — the stronger reasoner
is more confident about "this context doesn't contain the answer"
and less willing to guess. This is a good-calibration signal.

The SOMA-less-IDK asymmetry shrinks from 1.67× (4b) to 1.26× (9b):
9b closes some of the gap because it can squeeze answers out of
suboptimal-rank context that 4b just IDKs on. But SOMA still IDKs
20% less often than chroma in the (1,1) both-retrieve cell on 9b —
the mechanism is still visible.

## Per-type win distribution (qwen9b strict N=500)

| Type | N | chroma F1 | SOMA F1 | 4b lift | 9b lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 0.455 | 0.746 | +59% | **+64%** |
| multi-session | 133 | 0.113 | 0.144 | +36% | +27% |
| temporal-reasoning | 133 | 0.166 | 0.169 | +23% | **+2%** |
| knowledge-update | 78 | 0.380 | 0.460 | +11% | **+21%** |
| single-session-assistant | 56 | 0.766 | 0.771 | ~tie | ~tie |
| single-session-preference | 30 | 0.027 | 0.024 | ~tie | ~tie |

**single-session-user clean sweep persists and gets cleaner:** 23
SOMA F1 wins, 0 chroma wins, 47 tied on 9b (vs 22/0/48 on 4b). On
this question type the bigger LLM cannot rescue a suboptimal
retrieval — retrieval is the binding constraint.

**Notable shifts:**
- **temporal-reasoning lift drops from +23% to +2%**: likely because
  9b can reason from adjacent turns even when the specific gold turn
  is at suboptimal rank. A bigger LLM partially compensates for
  ranking imperfection on a type where temporal context spans many
  turns.
- **knowledge-update lift doubles from +11% to +21%**: 9b uses the
  better-ranked SOMA context more effectively on multi-fact-update
  questions. Bigger LLM benefits more from better retrieval here.

Net: reshuffling within per-type lifts, but overall lift unchanged.

## What this replicates / extends

Replicates the earlier N=200 verbose-prompt finding that retrieval
advantage transfers across LLM sizes:
- N=200 verbose: 4b +27%, 9b +25%
- **N=500 strict (this run): 4b +23%, 9b +22%**

Extends by:
- **Running at full N=500** instead of biased first-200 sample
- **Using strict prompting** (no verbosity penalty)
- **Adding the ranking-vs-recall decomposition** (stable at ~35/65)
- **Measuring IDK asymmetry** (persists but shrinks on 9b)

## How to reproduce

```bash
python -m benchmarks.industry.longmemeval.run_qa_compare \
  --variant small \
  --modes chroma_cosine soma_hybrid \
  --model qwen3.5:9b-q8_0 \
  --out-suffix=_n500_qwen9b_strict \
  --sbert-device cpu \
  --strict-prompt
```

Run takes ~3-4 hours on a 24GB GPU with sbert on CPU for VRAM
safety. Resumes cleanly via per-item jsonl if interrupted.

## Cross-reference

- qwen4b strict findings: `longmemeval_qa_compare_strict_findings.md`
- Causation decomposition: `longmemeval_causation_findings.md`
- Raw partition analysis: `retrieval_causation_n500_qwen9b_strict.md`
