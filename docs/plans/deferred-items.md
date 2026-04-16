# Deferred items — post-gap-closing backlog

Things that came up during the 2026-04-16 gap-closing push (Phases 1-7b) but were explicitly scoped out or flagged for later. Organized by category with source pointers, rough effort, and reasoning.

> **Triage:** post-2026-04-16 every item from the original "next-sprint set" has shipped (LoCoMo-QA eval in `065ebf7`, threshold calibration in `64dfc43`, LanceDB in Phase 6 + `36fec21`, JWT revocation in the blocklist work below). Everything below is tier-2+, ordered by rough effort.

---

## Completed (post-2026-04-16)

- **JWT revocation blocklist** — file-backed JSONL blocklist in `src/soma/auth_revocation.py`; `BlocklistBackend` Protocol + `FileBlocklist` impl; wired into `verify_token` + `serve.py`; new CLI `soma auth revoke / list-revoked / gc`; new `soma_auth_failures_total{reason="revoked_token"}` label.
- **LoCoMo-QA LLM-as-judge eval** — `benchmarks/run_locomo.py --run-qa-eval` flag; `benchmarks/harness/qa_eval.py`; post-hoc scoring across arms; DryRunBackend fallback when no live LLM available.
- **Threshold calibration sweep** — `benchmarks/run_conv_threshold_sweep.py` 4x4 grid harness.
- **LanceDB adapter** — full `VectorBackend` implementation with filter pushdown, snapshot/restore, parametrized protocol suite. `soma[lancedb]` extra.

---

## Tier 2 — conversational memory

- ✅ ~~**Async extraction mode**~~ Shipped in Phase 22 (`2271102`, `77beda7`). `extraction_mode="async"` with a `ThreadPoolExecutor(max_workers=1)`; `flush(timeout)` drains; `close()` + context-manager protocol; `clear_session()` flushes first. Cookbook §18.1 covers the low-latency-chat recipe and the GIL caveat.
- ✅ ~~**Batch extraction mode**~~ Shipped in Phase 25 (`4a2899b`, `360a743`). `extraction_mode="batch"` + `batch_size` kwarg; accumulates K turns and extracts via one LLM call (`BATCH_EXTRACT_PROMPT`); `flush()`/`close()` drain partials; `clear_session()` drops pending without extracting. Per-fact `source_turn_id` metadata landed as a side-benefit for sync mode too.
- ✅ ~~**`extractor_llm=` kwarg**~~ Shipped in Phase 11 (`34320b4`, `9ca76e1`).
- ✅ ~~**Summary re-summarization from raw turns**~~ Shipped in Phase 17 (`5124d4c`, `325029f`). New `resummarize_every` kwarg (default 5); every Mth summary re-derives from raw turns to prevent compounding drift. Cookbook §18 covers tuning.
- ✅ ~~**GDPR-grade forgetting**~~ Shipped in Phases 34-37 (`72a8f17`, `01b079b`, `874f943`, `7574d4e`, `bcbd9b7`, `2a6f943`, `f4d5218`). `ConversationalMemory.forget(text_matches=/subject=/user_id=, dry_run=, summary_strategy=)` with `ForgetPreview` inventory + `ForgetResult` delete, cascading through derived facts then summaries (regen or drop). `ForgetAuditSink` JSONL (`SOMA_FORGET_AUDIT_PATH`). `POST /forget` endpoint (extends legacy node_id form). Extraction-mode-aware: async flushes in-flight, batch drops pending. Full docs at `docs/gdpr.md`.
- ✅ ~~**Multi-user scoping**~~ Shipped in Phase 12 (`cc334d4`, `5216872`). `user_id` kwarg on ConversationalMemory with per-call override; retrieval auto-scopes; supersede enforces ownership. REST pattern via metadata field (no new endpoint).

## Tier 2 — observability

- ✅ ~~**`soma_compaction_total` / `soma_compaction_seconds` metrics.**~~ Shipped in Phase 8 (`24c0bd9`).
- ✅ ~~**`SOMA_METRICS_PUBLIC=0` gate.**~~ Shipped in Phase 8 (`99a6230`).
- ✅ ~~**`SOMA_METRICS_BUNDLE_LABEL_DISABLE=1`.**~~ Shipped in Phase 8 (`db28877` + follow-up commit routing inproc.py through the helper).
- **Histogram bucket tuning** after first real Grafana dashboard gets operator feedback. Phase 3 open question; defer until we have at least one production scrape history.

## Tier 2 — auth

- ✅ ~~**Refresh-token endpoint.**~~ Shipped in Phase 23 (`26cfa5f`, `2aaf86b`, `dc347a5`). `soma.auth.refresh_token(...)` helper + `POST /auth/refresh` + `soma auth refresh --token <jwt>` CLI. Mints a fresh `jti`/`exp` with the same `sub`/`bundles`/`aud`; `SOMA_JWT_REFRESH_TTL` overrides the default window; old token NOT auto-revoked (pinned in tests).
- ✅ ~~**Per-token rate limiting.**~~ Shipped in Phase 26 (`7bb92c0`, `548811b`, `fd3361a`). In-proc `TokenBucket` + `RateLimiter`; `SOMA_RATE_LIMIT_RPS` + `SOMA_RATE_LIMIT_BURST` + `SOMA_RATE_LIMIT_SCOPE` env; middleware wired inside `require_auth` so legacy `SOMA_API_KEY` path is also covered; `/metrics` + `/health` exempt; `soma_rate_limited_total{scope=...}` counter. Not a WAF — complementary to a reverse proxy, documented in `docs/auth.md`.
- ✅ ~~**Hashed-token store.**~~ Shipped in Phase 18 (`69d8e1a`, `632ea15`). Opt-in via `FileBlocklist(path, hashed=True)` or `SOMA_JWT_BLOCKLIST_HASHED=1` env. Dual-schema reader accepts legacy + new records so operators can flip without migration.
- ✅ ~~**Redis-backed revocation blocklist.**~~ Shipped in Phase 20 (`14e912d`, `bb88d62`, `d7d8773`). `RedisBlocklist` alongside `FileBlocklist`; `SOMA_JWT_BLOCKLIST_REDIS_URL` env dispatch; `soma[redis-revocation]` extra; hashed-mode interop verified cross-backend.
- ✅ ~~**Audience claim (`aud`)**~~ Shipped in Phase 18 (`708800b`, `40f9907`). `issue_token(..., audience=...)` + `verify_token(..., expected_audience=...)` + `soma auth issue --audience` + `SOMA_JWT_AUDIENCE` env.

## Tier 2 — backends (Phase 6 follow-ups)

- **Milvus adapter.** Build when asked. Heavier dep surface (pymilvus + server); overlaps Qdrant's niche almost entirely. ~2 d of work, but adds permanent maintenance surface (another version-compat test matrix, another filter translator). Skip unless a real Milvus shop shows up wanting SOMA on top.
- **Weaviate adapter.** Build when asked. Light client, heavy server. Low demand in the agent-memory space compared to pgvector/Chroma. ~2 d. Same reasoning as Milvus.
- ✅ ~~**pgvector adapter.**~~ Shipped in Phase 29 (`9a549e4`, `ff6be31`, `624ab8b`, `ed7ed3c`, `e150827`). `PgvectorBackend` via psycopg v3 + `pgvector>=0.3`; JSONB filter translator ($eq as `@> ::jsonb`, comparisons via `(metadata->>'f')::float`, $in as `ANY(%s)`, multi-field via AND; SQL-injection safe by parameter binding); snapshot via COPY+gzip; always-on unit tests + gated integration via testcontainers (`SOMA_PGVECTOR_INTEGRATION=1`).
- ✅ ~~**Chroma-as-backend.**~~ Shipped in Phase 27 (`d5d2c08`, `b4738c2`, `0da7d44`, `47041bb`). `ChromaBackend` adapter + `chroma_filter.to_chroma_where` translator ($eq/$ne/$gt/$gte/$lt/$lte/$in/$nin with $and wrapping for multi-field); `pip install "soma[chroma]"`; protocol contract suite row; snapshot/restore via dir copy; Windows sqlite-lock fix in close/restore.
- ✅ ~~**`backend.search_near_id(node_id, k)`**~~ Shipped in Phase 16 (`4825cd3`..`9437ff6`). Default impl delegates to `get_vectors + search`; Qdrant overrides via `recommend` API; LanceDB via Arrow-native self-join. `MemoryLayer.related()` routed through it.
- **Async Qdrant client (`AsyncQdrantClient`)** — build when asked. The real work is not the Qdrant swap — it's making every FastAPI route `async def`, which ripples through every endpoint, background task, and test. Once that lands, async Qdrant + async LLM calls let one worker serve ~10× the concurrent chats (since LLM + vector calls are I/O-bound). Cost: 1-2 weeks. Value: meaningful only for high-concurrency deploys (many simultaneous users per worker). Single-user or low-QPS doesn't benefit. No user has asked; revisit when someone's hitting the sync-worker ceiling.
- **Per-bundle vs shared Qdrant collection.** Per-bundle is v1; shared collection with bundle_name tag scales to 1000+ bundles. Decision deferred to demand.
- ✅ ~~**Snapshot version-compat tests** across Qdrant versions~~ Shipped in Phase 24 (`45c099c`, `3b5d703`, `74dcfc3`). testcontainers-driven 3×3 matrix across `1.11.3` / `1.12.4` / `1.13.5`; gated by `SOMA_QDRANT_VERSION_MATRIX=1` + `slow_qdrant` marker + `soma[qdrant-test]` extra; docs section in `docs/backends.md`. Weekly Gitea Actions workflow deferred until the runner has Docker available.

## Tier 2 — TypeScript client

- **npm scope claim** (`@soma-ai`) — operator-only manual step before first publish. Fallback scopes documented: `@soma-memory`, `@soma-ml`.
- **`soma-memory-client` unscoped alias** pointing at the scoped package via `deprecate` — captures npm-search hits. ~15 min.
- ✅ ~~**Retry middleware as optional re-exports.**~~ Shipped in `c01b4f5`. `withRetry(fetchImpl, opts)` wrapper plugs into `createClient`'s existing `fetch` option — no fork of generated code.
- **React Query integration** via `openapi-react-query` plugin. ~4 h, only if React users ask.

## Tier 2 — k8s / cloud

- **HPA templates** in the Helm chart. Build when asked. Technically unblocked now — S3/GCS bundles (Phase 30-33) + Qdrant HTTP backend together mean multi-replica is safe. Cost: 1-2 days. Value: low until someone actually runs SOMA on k8s with a Qdrant cluster — no such user today. Ships a capability nobody's asked for.
- **`SomaCluster` CRD operator.** Build when asked — and probably not even then. A k8s operator managing SOMA + Qdrant + backup cron + monitoring as one declarative resource would be a genuine enterprise selling point (declarative upgrades, DR, per-tenant isolation). But that's a different product from "local-first agent memory." Cost: ~3 weeks minimal, months for production-grade. Value: high IF we're pitching enterprise k8s; low if the product story stays local-first. Don't ship until enterprise is actually on the roadmap.
- ✅ ~~**S3/GCS bundle backend**~~ Shipped in Phases 30-33 (`e727917`, `0917c57`, `2deaf0e`, `ba0e34f`, `0f9e5db`, `aad100d`, `53721c4`, `774c444`, `25c79df`, `dc6eba9`, `5a4fba0`). `ObjectStore` Protocol + `LocalFSObjectStore` / `S3ObjectStore` (boto3 + moto tests) / `GCSObjectStore` (google-cloud-storage + gcp-storage-emulator). `MemoryLayer.save/load` accepts `file://` / `s3://` / `gs://` URLs. Staging via `local_root` temp dir. Deploy recipes in `docs/cloud.md` (Lambda + S3, Cloud Run + GCS, Fly + Cloudflare R2).
- **`soma cloud deploy` CLI** — partially obsoleted by `docs/cloud.md` (Phase 33), which documents the three deploy flows (Lambda, Cloud Run, Fly) as copy-paste recipes. A CLI wrapper would still be nice for a real managed service, but the capability is there today via the stock platform CLIs + Helm. Build when asked.
- **Helm 4 migration** — stay on 3.x for v1; migrate when the ecosystem settles. No ETA.
- **Topology spread vs anti-affinity production hardening** — build when asked. Stubs in `values.yaml`; no tested defaults. Picking defaults without a real multi-replica deploy to measure against would be guesswork.

## Tier 2 — benchmark polish

- **GPT-4-driven LoCoMo-QA parity run.** Mem0 / Zep / Letta publish headline QA accuracy numbers using GPT-4 as responder *and* judge. Our local-LLM smoke (gemma-4-26b-a4b-it + qwen3.5-27b via LM Studio, 2026-04-16) hit ~2% accuracy — consistent with the retrieval ceiling (R@5 = 0.238) × strict small-model inference, not a SOMA deficit. To produce apples-to-apples numbers against their published claims we need to run the harness with the same GPT-4 stack. Cost estimate: ~$30-60 at current OpenAI rates for a full LoCoMo-QA run across 3 arms. **Not a positioning blocker** — the infrastructure story (store/retrieve/disk/scale) stands on deterministic ms-level measurements. Worth doing if we ever need a single headline QA number for a paper or pitch deck. Source: QA eval session 2026-04-16 post-1M-benchmark investigation. Harness already supports `--run-qa-eval` + `--qa-eval-k` + `--judge-llm-name`. ~2 h setup + API wall-clock time.

## Tier 2 — docs / ops polish

- ✅ ~~**`soma chat` async streaming.**~~ Shipped in Phase 28 (`31fff9e`, `dece9ff`, `7a0a4d2`). `stream_generate(prompt)` optional Protocol method on `LLMBackend`; REPL detects via hasattr and falls back to `generate()` when missing. Streaming added for OpenAI / LM Studio (via OpenAI-compatible) / Ollama / Anthropic. Follow-up: unify `scripts/demo_wiki_chat.py`'s legacy `_chat` path.
- ✅ ~~**Prometheus sample dashboards**~~ — Shipped in Phase 9 (`3c6441c`..`fb50482`). Three RED/USE dashboards + import guide under `deploy/grafana/`.
- ✅ ~~**`soma bundle` subcommand group**~~ — Shipped in Phase 10 (`3987062`..`e61006b`). `list` / `info` / `delete` verbs; `migrate` still deferred.
- **`soma bundle migrate`** — schema upgrade verb for cross-version bundle migration. Not yet needed since bundle format is stable. Revisit if we break schema.
- ✅ ~~**Benchmark scheduler / cron**~~ Shipped in Phase 21 (`4a79903`..`c532def`). `scripts/check_bench_regressions.py` + JSON sidecar on scale/retrieval harnesses + `benchmarks/golden/*.json` snapshots + `.github/workflows/bench-regression.yml` (PR non-blocking, nightly blocking). Follow-up: dedicated `[bench]` extra in pyproject to tidy the CI install list.
- ✅ ~~**Ephemeral/in-RAM mode ergonomics.**~~ Shipped in Phase 19 (`662606e`, `aa9b052`, `2fe6fde`). `MemoryLayer.ephemeral()` classmethod; `soma chat --ephemeral` + `--save-on-exit` flags with atexit hook; `POST /snapshot` REST endpoint with cwd-escape safety.

## Pre-existing (carried forward)

- ✅ ~~**Task #173 — Lazy stable-capture.**~~ Shipped in Phase 13 (`25ffb15`, `916717c`, `7c3a4c2`). Removes O(N²/K) cost from consolidate; benchmark adapter pins eager behaviour so existing reports stay comparable. Alpha=0 default means zero user-visible change; becomes load-bearing if graph-rerank reactivates.
- **Task #174 — Enterprise scale benchmark (100K + 1M).** 1M still running at time of writing.
- ✅ ~~**Pre-existing mypy errors** in `src/soma/memory/api.py`~~ Closed in `0f5a029` (9 errors total — 5 `has_encoder` narrowing, 2 `np.ndarray` generics, 1 `_COMPARE_OPS` dict typing, 1 `with_sbert` None guard).

## Research agenda (out-of-sprint)

- **Graph re-rank re-activation.** The plastic-graph substrate ships but the growth thresholds (training-tuned) don't fire under memory-layer workloads. The research agenda in `benchmarks/reports/paper-draft.md` §5 details what would activate it.
- **Multimodal memory.** Current store is text-only. Arrow-native adapters (LanceDB) could host image/audio vectors behind the same API. No concrete plan.
- **Federated / multi-device sync.** Bundles are portable; sync across devices is a different problem (CRDT on the WAL? sync service?). No plan.

---

## How to triage this doc going forward

1. When a deferred item becomes a real user ask → promote to a dated plan under `docs/plans/`.
2. When a deferred item becomes impossible or irrelevant → delete with a one-line git commit message.
3. Review at least every quarter; items staler than two quarters with no ask are probably not happening — delete them.

### "Build-when-asked" philosophy

Several entries below carry the "build when asked" note. These are
items where the *capability* would be valuable to some audience but
*nobody has asked yet*. Shipping them speculatively adds maintenance
surface (more adapters to version-compat-test, more docs to keep in
sync, more deps to update) without a user whose concrete need
disciplines the design. The pattern is: write the capability down,
keep the rationale current, and build it when a real user's request
can shape the scope. Cost estimates stay attached so a future
prioritization pass can act without re-researching.
