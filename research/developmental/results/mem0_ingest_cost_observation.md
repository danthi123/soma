# Mem0 ingest cost is prohibitive on LongMemEval (partial finding)

**Status:** Observation, not a finished experiment. Mem0 + `infer=True`
attempted on LongMemEval but killed at ~10 minutes after completing
ingest on 0/10 items. The observed per-session cost is a useful data
point on its own.

**Date:** 2026-04-20.
**Attempt:** `benchmarks/industry/longmemeval/run_mem0_compare.py`,
`mem0_infer` mode, N=10 items.

## Observed cost

With Mem0 `infer=True` (default, LLM-extracted facts mode), same LLM
as SOMA QA comparison (qwen3.5:4b-q8_0 via Ollama):
- **~10 minutes of wall-clock time**
- **8 LLM calls total** (POST to `/api/chat`)
- **0 items fully ingested** (first item has 53 sessions)
- Plus spacy lemma + full model loads for entity extraction

That's ~75 seconds per LLM call. Extrapolating: one LongMemEval item
(~50 sessions) needs ~50-100 Mem0 LLM calls during ingest. Budget:
**~60-120 minutes per item**.

## Comparison to SOMA

Same corpus, same sbert embedder:
- **SOMA ingest**: 50 sessions × ~25ms/session = ~1.3 seconds per item
- **Mem0 ingest**: ~60-120 minutes per item

**Ratio: ~3000x** for ingest time with the same local LLM as the
extraction driver.

## Why so slow?

Mem0's `infer=True` does per-session:
1. LLM call to extract facts (large prompt = full session content)
2. LLM call to extract entities/relationships
3. Spacy lemma + entity processing
4. Multiple vector-store inserts (facts + entities)

Each adds latency. For LongMemEval's ~50-session haystacks per item,
this compounds to prohibitive cost on any non-cloud-API LLM.

## What this means for positioning

Mem0's extraction-first architecture has real cost when running
locally. It's designed for cloud LLMs at ~200-500ms/call (GPT-4o-mini
speed). With a local qwen3.5:4b @ 75s/call, Mem0 is not practical for
bulk session ingest — at least not in its default config.

SOMA's raw-message storage is **3000x faster to ingest** at the same
wallclock. Even if Mem0's extracted facts gave marginally better
retrieval, the time cost is a major real-world constraint:
- **SOMA**: build a 500-session brain in ~13 seconds
- **Mem0**: build the same brain in ~10 hours (local qwen3.5:4b)
- **Mem0**: ~2-5 minutes with cloud GPT-4o-mini (if API calls are fast)

## Open question: does the extraction help QA quality enough?

We don't yet have Mem0 QA quality numbers. Possible paths:
1. Run Mem0 with a SMALLER/FASTER LLM (qwen3:0.6b, tiny) for
   extraction. Trades extraction quality for speed. ~5-10x faster.
2. Run on cloud GPT-4o-mini for Mem0 extraction. API cost but fast.
3. Run Mem0 with `infer=False` (no extraction). Makes it essentially
   a chroma wrapper — not a fair comparison of its design.

## Next steps

- Defer proper SOMA-vs-Mem0 QA comparison until we have cloud API
  access or a faster extraction LLM.
- Meanwhile, document the ingest-cost differential as a positioning
  data point.

## Files

- `benchmarks/industry/longmemeval/results/mem0_compare_n10.log` —
  interrupted run log
