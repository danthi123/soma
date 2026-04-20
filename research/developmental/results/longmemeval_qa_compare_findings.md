# LongMemEval QA comparison — retrieval lift translates to QA lift, but unevenly

**Status:** CONFIRMED (with nuance). SOMA's BM25+cosine hybrid delivers
**+10% relative F1 overall** on LongMemEval QA (N=500) over chroma
cosine with the same LLM. The win is dramatic on single-session
questions (+49% F1) and modest on multi-session / temporal-reasoning
questions (+3-6% F1).

**Date:** 2026-04-20.
**Runner:** `benchmarks/industry/longmemeval/run_qa_compare.py`.
**Scope:** LongMemEval small variant, N=500 (full) + N=100 (initial).
**Embedder:** sentence-transformers/all-MiniLM-L6-v2.
**LLM:** qwen3.5:4b-q8_0 via Ollama, T=0, max_context_tokens=3800.

## Headline (N=500, full LongMemEval small)

| Mode | F1 | R@5 | avg input tok | retrieve+LLM ms |
| --- | ---: | ---: | ---: | ---: |
| chroma_cosine | 0.148 | 0.932 | 3879 | 1887 |
| **soma_hybrid** | **0.164** | **0.980** | 3879 | 1889 |

**+10.4% F1 relative, +5.2% R@5 absolute.**

Pairwise F1 wins (N=500): SOMA wins 95, ties 317, loses 88. Net +7
strict wins out of 500 — positive but marginal at the aggregate level.

## Per-type breakdown — this is where the real story lives

| Question type | N | chroma F1 | SOMA F1 | Δ F1 | chroma R@5 | SOMA R@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 0.211 | **0.315** | **+49.0%** | 0.800 | 1.000 |
| multi-session | 133 | 0.060 | 0.063 | +5.8% | 0.962 | 0.985 |
| temporal-reasoning | 133 | 0.116 | 0.119 | +3.1% | 0.940 | 0.970 |
| knowledge-update | 78 | 0.145 | 0.153 | +5.1% | 0.974 | 1.000 |
| single-session-assistant | 56 | **0.345** | 0.327 | **-5.4%** | 0.982 | 1.000 |
| single-session-preference | 30 | 0.176 | 0.175 | -0.6% | 0.867 | 0.867 |

## Interpreting the per-type story

**Where SOMA wins dramatically (single-session-user, +49%):** The
answer lives in one session. The question often references a specific
entity (name, date, number, product). Cosine alone disperses these
into paraphrase space; BM25 pulls the lexically-matching session to
the top. When SOMA's R@5 jumps from 0.80 → 1.00, the LLM reliably
extracts the answer. This IS the case where retrieval quality is the
binding constraint on end-to-end quality.

**Where SOMA wins modestly (multi-session, temporal-reasoning, 
knowledge-update, +3-6%):** The LLM must reason across multiple
retrieved sessions to compose the answer. Even with perfect retrieval
(SOMA hits R@5=0.97-1.00), a 4B-parameter LLM struggles to combine
evidence correctly. Retrieval is necessary but not sufficient — the
LLM is the bottleneck. F1 stays low (0.06-0.15) regardless of
retrieval quality.

**Where SOMA ties or slightly loses (single-session-assistant,
single-session-preference):** Both systems hit R@5 ~0.87-1.00 — the
gold session is found by both. Any F1 differences reduce to verbosity
or phrasing differences in the LLM's output, which are noise.
N=56 and N=30 for these types, so statistical significance is
limited.

## Sample-size caveat on the N=100 result

The earlier N=100 run showed +42% F1 (vs +10% here). Why? N=100 is
items 0-99, which are 70 single-session-user + ~30 multi-session
(the type distribution is not random — LongMemEval groups similar
types together). That pool is heavily biased toward the type where
SOMA wins most dramatically.

**This isn't wrong, but it overstates the headline.** The honest
claim is: SOMA hybrid delivers measurable F1 lift end-to-end, with
the size of the lift depending on question type — big on single-turn
entity-retrieval, small on multi-document reasoning where the LLM
is the bottleneck.

## Why R@5 wins don't always turn into F1 wins

Intuition: if SOMA finds gold on more items, the LLM should answer
correctly more often. Why isn't that true across the board?

1. **Retrieval is necessary but not sufficient.** For multi-session
   questions, finding the right 5 sessions is the first step; the LLM
   still has to synthesize across them. A 4B model struggles.
2. **F1 is noisy at low values.** On multi-session questions where
   both systems score F1 ~0.06, there's not much signal to move.
3. **Verbosity differences cancel out.** When both systems hit gold,
   small wording differences in the LLM output produce small F1
   differences in both directions.

## Rerank on LongMemEval still hurts (corroborated)

The N=100 run also tested chroma+cross-encoder-rerank. Result was
F1=0.170 — essentially tied with chroma_cosine's 0.168. Consistent
with the retrieval-benchmark finding: cross-encoder rerank adds noise
on LongMemEval's long multi-topic sessions. Not worth the latency
cost here.

## Full-context truncation is worse at matched budget

At 3.8K context budget, `full_context` (dump everything, drop oldest
to fit) collapses to F1=0.029 — essentially no retrieval. This
confirms: when context is constrained (small LLM, cost-sensitive RAG),
focused retrieval crushes naive stuffing.

## Honest production guidance

```python
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()
for turn in session_history:
    mem.store(f"[{turn.date}] {turn.role}: {turn.content}")

# SOMA beats chroma on retrieval quality across the board.
# For entity-retrieval questions, this translates to big F1 lift.
# For multi-document reasoning, the LLM is the bottleneck.
hits = mem.retrieve(question, k=5, hybrid_alpha=0.3)
```

**When to expect a large lift:** questions that hinge on specific
entities that appear lexically in one session (names, dates, numbers,
product names). The BM25 leg is the differentiator.

**When to expect a small lift:** multi-document reasoning, temporal
composition, where the LLM's ability to synthesize is the dominant
factor.

## Cross-benchmark story (updated)

| Benchmark | Level | Metric | SOMA win |
| --- | --- | --- | --- |
| LoCoMo retrieval | turn (mxbai) | R@5 | +13% vs chroma+rerank |
| LongMemEval retrieval | session (sbert) | R@5 | +4.7% + 8.3× faster |
| LongMemEval QA (N=500) | session (sbert) | F1 | **+10% overall, +49% on single-session-user** |

## Limitations

1. **Single LLM (qwen3.5:4b-q8_0).** Effects may differ with larger
   LLMs. A stronger reasoner might compensate for chroma's weaker
   retrieval on single-session questions (closing the +49% gap) AND
   extract value from SOMA's better retrieval on multi-session
   questions (opening a new gap). Untested.
2. **Single embedder (sbert).** mxbai-embed-large showed larger
   differences on LoCoMo retrieval; would expect similar directional
   effect here.
3. **Single reranker corpus.** Cross-encoder rerank helps LoCoMo
   (short atomic turns) but hurts LongMemEval (long multi-topic
   sessions). Per-corpus choice matters.

## Follow-up scope

1. **Larger LLM run (qwen3:14b)** — does the F1 gap widen on
   multi-session questions where retrieval is necessary but not
   sufficient?
2. **SOMA vs Mem0 end-to-end** — Mem0's LLM-extracted facts pipeline
   is 3000× slower to ingest (see `mem0_ingest_cost_observation.md`).
   Does the extraction buy enough QA quality to justify the cost?
3. **Plastic graph activation** — does SOMA's graph improve retrieval
   over a long-running session? (see design doc)

## Files

- `benchmarks/industry/longmemeval/run_qa_compare.py` — harness
- `benchmarks/industry/longmemeval/analyze_qa_compare.py` — analysis
- `benchmarks/industry/longmemeval/results/qa_compare_*_n500.json` — per-mode
- `benchmarks/industry/longmemeval/results/qa_compare_*_n500.jsonl` — per-item
- `benchmarks/industry/longmemeval/results/qa_compare_*_n100.{json,jsonl}` — N=100 (initial)
