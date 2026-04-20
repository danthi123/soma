# Mem0 vs SOMA ingest comparison — apples vs oranges warning

**Status:** Partial observation with honest caveat.  Earlier draft
claimed "SOMA is 3000× faster than Mem0" — that framing is unfair
and has been walked back in this revision.

**Date:** 2026-04-20.

## What we actually measured

Mem0 with `infer=True` (its default) attempted on LongMemEval with
local Ollama LLM (qwen3.5:4b-q8_0):
- **~10 minutes wall-clock**
- **8 LLM calls** to ollama `/api/chat`
- **0/10 items fully ingested** (first item has 53 sessions, not one finished)
- Spacy lemma + full model loaded
- ~75 seconds per LLM call (extraction prompt is large)

Extrapolating: one LongMemEval item needs ~50-100 LLM calls during
ingest → **60-120 minutes per item**.

SOMA over the same corpus: **~1.3 seconds per item** (sbert embed +
vector-store insert, no LLM).

## Why this comparison is NOT apples-to-apples

Mem0 `infer=True` is not doing the same thing as SOMA `store()`:

| Operation | SOMA `store()` | Mem0 `infer=True` |
| --- | --- | --- |
| Accept raw text | ✅ | ✅ |
| Embed text | ✅ | ✅ |
| Insert into vector store | ✅ | ✅ |
| **LLM-extract facts** | ❌ | ✅ |
| **LLM-extract entities** | ❌ | ✅ |
| **LLM-reconcile with existing memory** | ❌ | ✅ |
| Spacy NER + lemmatization | ❌ | ✅ |

Mem0 is doing **semantic compression** — it's reading the whole
conversation, extracting atomic facts, and building a structured
knowledge base. SOMA is doing **indexing** — it's storing the raw
turns and embedding them for retrieval.

These are different product philosophies:
- **Mem0 bet**: compress at write time, query at read time over facts
- **SOMA bet**: store everything at write time, retrieve well at read time

A fair "indexing speed" comparison would be Mem0 with `infer=False`
(skip extraction, store raw) vs SOMA. We haven't run that yet — the
`infer=False` path is a ~10 lines/s throughput level comparable to
chroma.

## What the earlier "3000× faster" claim should have said

Claim: "SOMA's default store-and-index path is ~3000× faster than
Mem0's default extract-first path on local LLM."

That's literally true but framed in a way that hides what Mem0 is
buying with those LLM calls. The correct takeaway:
- **If you need semantic fact extraction at write time** and have
  a fast cloud LLM, Mem0's extraction is fine (cloud GPT-4o-mini at
  ~200-500ms/call → ~2-5 min per item, not 60-120 min).
- **If you need semantic fact extraction at write time** and only
  have a local LLM, Mem0 is impractical at LongMemEval scale on a
  consumer GPU.
- **If you don't need fact extraction at write time**, Mem0 with
  `infer=False` is roughly chroma-equivalent in speed, and SOMA's
  hybrid retrieval beats both on read-time quality (see
  `longmemeval_qa_compare_findings.md`).

## What still needs to be tested

The interesting question isn't "who ingests faster" — it's "does
Mem0's semantic extraction deliver better QA than SOMA's raw-text
hybrid retrieval, with the same LLM on the QA side?"

This requires either:
- A cloud API (Claude, OpenAI) for Mem0's extraction to run in
  reasonable wall-clock.
- A much smaller local LLM for Mem0's extraction (qwen3:0.6b?),
  which risks bad extraction quality.

Planned as a follow-up with Claude API.

## Files

- `benchmarks/industry/longmemeval/results/mem0_compare_n10.log`
  — interrupted run log
- `benchmarks/industry/longmemeval/run_mem0_compare.py` — harness
  (supports mem0_infer, mem0_raw, soma_hybrid modes)
