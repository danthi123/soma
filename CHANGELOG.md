# Changelog

All notable changes to SOMA are documented here.

## [Unreleased] — 2026-04-16

### Added — API + perf

- `MemoryLayer.store_batch(texts, metadatas=)` — bulk ingest that calls
  the embedder once instead of per-text and invalidates the FAISS
  index once at the end.
- O(1) id lookups: added `_id_to_idx` cache so `get` / `related` /
  `forget` / `in` are constant-time instead of O(N) scans of `_ids`.
- REST: `POST /store_batch`, `GET /related/{node_id}` on default +
  `/bundles/{name}/...` routes for parity with the Python API.
- CLI: `soma forget --bundle <b> --node-id <id>` accepts a full id or
  a unique prefix, then re-saves the bundle.

### Added — project files

- `LICENSE` (MIT) at the repo root — pyproject already declared MIT
  but the file was missing.
- `CONTRIBUTING.md` — scope, setup, style, and PR expectations.

### Removed — dead weight

- All pre-pivot scripts under `scripts/` (training loop, autonomous
  tick, bootstrap verbalizer, diagnostics). 34 files gone; the
  memory-layer demos + `migrate_chroma` importer are all that remains.
- All pre-pivot reports (`reports/` dir removed; the live numbers now
  live under `benchmarks/reports/`).
- `docs/progress/{BUILD_COMPLETE,CHECKPOINT,FIRST_REAL_CORPUS_RUN,DEFERRED}.md`
  (kept `HYBRID_PIVOT.md` as the historical framing CLAUDE.md links).
- `docs/plans/2026-04-{12,13,14}-*` (12 pre-pivot design docs).
- `docs/{AUTONOMOUS_LOOP,UI_GUIDE,loop_tick_prompt,deployment}.md`.
- 15 pre-pivot test modules under `tests/test_scripts/`.

### Added — recall boosters

- **Hybrid search** via `mem.retrieve(query, k, hybrid_alpha=...)` — blends
  cosine with pure-Python BM25-Okapi. New `soma.memory.bm25.BM25Index`
  module; lazy, version-tracked rebuild; no extra deps.
- **Cross-encoder re-ranking** via `mem.retrieve(..., rerank_top_n=N)` plus
  `mem.attach_reranker(CrossEncoderReranker())`. New `soma.memory.rerank`
  module with `Reranker` protocol, `CrossEncoderReranker` (wraps
  sentence-transformers), and `StubReranker` for tests.
- **In-index metadata filtering** via `mem.retrieve(..., where={...})`.
  Chroma-compatible subset: exact match, AND across fields,
  `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`/`$in`/`$nin`.
- **LLM query expansion** via `soma.llm.QueryExpander` + RAGSession's
  `query_expander=` field. Rewrites the question into N variants,
  retrieves top-k per variant, merges with Reciprocal Rank Fusion
  (`soma.llm.rrf_merge`).

Measured lift on LoCoMo (5,882 turns / 1,982 queries, same sbert
embedder as peer DBs): baseline R@5 = 0.238 → hybrid alpha=0.3 = 0.415
(+17.7 pp) → hybrid + cross-encoder rerank = **0.450 (+21.2 pp, +89%
relative)**. R@1 nearly triples (0.098 → 0.287). Latency +34 ms.

### Added — production hooks

- `GET /health` (no-auth liveness probe, reports loaded bundle count)
- `GET /version`
- Optional `Authorization: Bearer <SOMA_API_KEY>` enforcement on all
  non-liveness endpoints.
- Multi-tenant `/bundles/{name}/...` routes — per-name `MemoryLayer`
  cache, on-disk root via `SOMA_BUNDLES_DIR`.
- Dockerfile + docker-compose gain HEALTHCHECK and commented auth env.

### Added — LLM plug-and-play

- `soma.llm` package with `LLMBackend` protocol + 5 concrete backends:
  `OllamaBackend` (local, no key), `OpenAIBackend`, `AnthropicBackend`,
  `OpenAICompatibleBackend` (vLLM / LM Studio / LiteLLM / llama.cpp),
  `HuggingFaceBackend` (wraps existing deploy factory).
- `backend_from_env()` auto-picks backend based on env vars / liveness.
- `RAGSession` glue for retrieve → prompt → generate with citations.

### Added — CLI

- New `soma` console script with `index` / `chat` / `stats` / `search` /
  `serve` / `version` subcommands. Wraps the demos so callers don't need
  to remember script paths.

### Added — integrations

- `SomaRetriever` (LangChain + LlamaIndex) now accepts `where`,
  `hybrid_alpha`, and `rerank_top_n`. Pre-existing `k`-only usage
  unchanged.

### Added — demos & tooling

- `demo_wiki_chat.py` — markdown + PDF ingest with heading-aware
  chunker, chat with citations.
- `demo_memory_inspect.py` — read-only browse / stats / filter / search /
  JSONL dump of any bundle.
- `demo_related_browser.py` — interactive graph walk via `mem.related()`.
- `demo_web_ui.py` — Gradio chat UI (optional `gradio` dep).
- `scripts/import_corpus.py` — generic JSONL / JSON / CSV importer with
  Mem0 / Letta / Zep presets.

### Added — benchmarks

- `benchmarks/run_scale_enterprise.py` — index-only (shared-embedding)
  scale harness at 5K / 20K / 100K / 1M.
- `benchmarks/run_recall_boost.py` — measures recall lift from
  hybrid + rerank on LoCoMo.

### Added — docs

- `docs/quickstart.md` — 60-second tour.
- `docs/comparison.md` — SOMA vs Chroma / Mem0 / Letta / Zep / Pinecone.
- `docs/cookbook.md` — 16 recipes.
- `docs/demos.md` — walkthrough of every demo.
- `docs/llm-backends.md` — per-backend setup.
- `docs/recall-improvements.md` — what's shipped + research agenda.

### Changed

- Paper-draft headline updated to include 100K-scale store gap
  (~3500×) and the recall-boost result (R@5 +21.2 pp).
- README feature-comparison table now includes Pinecone column and
  new rows for where-filter, hybrid, rerank, query expansion,
  LLM plug-and-play, multi-tenant REST.

### Benchmarks added this release

- Enterprise scale (index-only, shared sbert embedder):
  - 5K: soma-flat store 0.0s vs chroma 83s (~1000×)
  - 20K: soma-flat store 0.1s vs chroma 5.3 min (~3180×)
  - 100K: soma-flat store 0.4s vs chroma 23.6 min (~3535×)
- HNSW retrieve at 100K: SOMA 5.58 ms vs Chroma 28.56 ms (5.12× faster)
- LoCoMo recall with boosters: baseline 0.238 → hybrid+rerank 0.450
  (+21.2 pp, +89% relative)

### Incremental consolidation (shipped 2026-04-15)

- Cursor-based incremental consolidate — O(1) when no new entries.

---

## [Earlier]

See `docs/plans/` and git history for the pre-2026-04-16 arc, including
the memory-layer pivot, the hybrid-brain experiments, and the autonomous
loop / bootstrap trainer.
