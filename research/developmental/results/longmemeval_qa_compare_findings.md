# LongMemEval QA comparison — does retrieval lift translate to answer lift?

**Status:** [pending — run in progress]

**Date:** 2026-04-20
**Runner:** `benchmarks/industry/longmemeval/run_qa_compare.py`
**Scope:** LongMemEval small variant, N=100 items × 4 retrieval modes.
**Embedder:** sentence-transformers/all-MiniLM-L6-v2.
**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2.
**LLM:** qwen3.5:4b-q8_0 via Ollama, temperature=0.0, max_context_tokens=3800.

## The question

The retrieval benchmark (`longmemeval_retrieval_findings.md`) showed
SOMA's BM25+cosine hybrid beats chroma+cross-encoder-rerank by +4.7%
R@5 on LongMemEval at 8.3× lower latency. But retrieval R@K is only
meaningful if it turns into better LLM answers. This comparison feeds
the retrieved context into the SAME LLM and scores the answer.

Modes compared (same LLM, same system prompt, same max context budget):
- `chroma_cosine`   : chroma cosine top-5
- `chroma_rerank`   : chroma cosine top-20, cross-encoder rerank to top-5
- `soma_hybrid`     : SOMA hybrid alpha=0.3, top-5 (winning retrieval config)
- `full_context`    : no retrieval — stuff all haystack (oldest-dropped to fit budget)

## Results

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@5 | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_cosine | [pending] | | | | | | | |
| chroma_rerank | [pending] | | | | | | | |
| soma_hybrid | [pending] | | | | | | | |
| full_context | [pending] | | | | | | | |

## Analysis

[pending — fill in after run completes]

## Key claims

[pending]

## Files

- `benchmarks/industry/longmemeval/results/qa_compare_*_n100.json` — per-mode full data
- `benchmarks/industry/longmemeval/results/qa_compare_summary_n100.md` — summary table
- `benchmarks/industry/longmemeval/results/qa_compare_n100.log` — runtime log
