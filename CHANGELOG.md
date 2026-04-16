# Changelog

All notable changes to SOMA are documented here.

## [Unreleased] — 2026-04-16

### Added — pluggable vector backends (Phase 6)

- **`VectorBackend` protocol** (`src/soma/memory/backend.py`):
  runtime-checkable interface every adapter implements. Numpy
  `float32` at the boundary (never torch), so adapters that speak
  arrow/C/HTTP stay out of torch's import graph. Raises
  `FilterPushdownUnsupported(op=?, field=?)` when a backend can't
  translate a `where` clause — MemoryLayer catches and falls back to
  its Python pre-filter + `search_subset` path.
- **`InProcBackend`** (`src/soma/memory/backends/inproc.py`): default
  adapter. Pure numpy vector matrix + lazy FAISS (flat / hnsw).
  `supports_filter_pushdown=False`. Snapshot writes
  `memory_embeddings.pt` bit-identical to pre-Phase-6 bundles so old
  bundles keep loading.
- **`QdrantBackend`** (`src/soma/memory/backends/qdrant.py`, optional
  `pip install soma[qdrant]`): three modes in one class — `memory`
  (embedded in-proc core), `local` (on-disk, warns past 20K), `http`
  (the scale path). `supports_filter_pushdown=True`; filter
  translator in `qdrant_filter.py` handles every `_COMPARE_OPS` op
  plus `$in`/`$nin`. Per-bundle collection, deterministic int64
  point-ids, `node_id` stored in payload so rehydration via `scroll`
  works.
- **`MemoryLayer.backend=`** (`src/soma/memory/api.py`): new kwarg
  routes every vector op (add / remove / search / get_vectors /
  subset) through the supplied backend. Default = `InProcBackend`
  with the same `faiss_*` kwargs so zero-change upgrades keep
  working.
- **Filter pushdown dispatch** in `retrieve(where=...)`: tries the
  backend first, falls back cleanly on `FilterPushdownUnsupported`.
  Parity tests in `tests/test_memory/test_filter_parity.py` pin
  equivalence across paths.
- **`_soma_activations` keyed by `node_id`** (was positional list):
  unlocks backends that soft-delete or reorder. `consolidate`
  cursor remains an int into the ordered text list and is clamped
  on `forget`.
- **`tests/test_memory/test_backend_protocol.py`**,
  `test_inproc_backend.py`, `test_qdrant_backend.py`,
  `test_qdrant_filter.py`, `test_qdrant_http.py`,
  `test_soma_activations_keying.py`, `test_filter_parity.py` —
  full TDD coverage for the phase. HTTP-mode tests skip unless
  `SOMA_QDRANT_TEST_URL` env is set.
- **`docs/backends.md`**: protocol overview, when-to-pick-which
  table, InProc vs Qdrant positioning, filter-pushdown semantics,
  and a "write your own backend" walkthrough.
- **`benchmarks/run_backend_matrix.py`** + smoke tests: adapter
  matrix harness (InProcFlat, InProcHNSW, QdrantLocal, QdrantHTTP
  when env set) with shared sbert cache. 1K smoke shows InProcHNSW
  at Recall@10 = 0.974 (inside the 0.02 ship-blocker threshold) and
  QdrantLocal at Recall@10 = 1.000.
- **Feature comparison (README)**: new row — *Pluggable vector
  backends: yes (InProc + Qdrant)* vs Chroma/Mem0-Zep/Pinecone which
  all say no.
- **Paper §4.3.5**: adapter-matrix table + positioning of filter
  pushdown as the no-callsite-change superpower.

### Added — TypeScript client

- **`@soma-ai/client`** (new `clients/typescript/` package): thin
  wrapper over `openapi-fetch` with types generated from the live
  SOMA `/openapi.json`. Works in Node 18+, browsers, Deno, Bun, and
  Cloudflare Workers. `createClient({ baseUrl, token })` papers over
  all three server auth modes (JWT preferred, legacy `SOMA_API_KEY`
  fallback, open-mode default). Full path / body / response /
  401-error type inference; vitest covers both compile-time type
  shapes and an msw-stubbed runtime round-trip.
- **OpenAPI snapshot** (`clients/typescript/openapi.json`) committed
  alongside the generated `src/schema.d.ts`. CI workflow
  `.github/workflows/client-ts.yml` re-snapshots on every PR to
  `src/soma/serve.py` and fails fast on drift; publishes on `v*` tag
  push with `--provenance`.
- Reference docs: [`docs/clients.md`](docs/clients.md) (install, auth
  modes, multi-tenant usage, retry middleware, browser notes, scope
  claim checklist).

### Added — conversational memory

- **`ConversationalMemory` wrapper** (`src/soma/memory/conversational.py`):
  opt-in sugar over `MemoryLayer` that adds Mem0/Zep-style LLM-driven
  fact extraction + ADD/UPDATE/SUPERSEDE/NOOP reconciliation on every
  turn. Threshold short-circuit (<0.75 ambiguous floor, >=0.92 near-dup)
  avoids an LLM round-trip on clear-cut cases; only the ambiguous band
  pays for a second LLM call. Zep-style "invalidate, don't delete"
  preserves superseded entries for audit (metadata.superseded_by
  pointer). Rolling session summaries every N turns, raw turns still
  stored so LoCoMo-style evaluation pipelines stay compatible. See
  `docs/cookbook.md` §18 and
  `docs/plans/2026-04-16-phase-2-conversational-memory.md`.
- **`MemoryLayer.update_metadata(node_id, patch)`**: metadata-only
  mutation primitive backing SUPERSEDE (old entry flagged, not deleted).
  WAL-replay compatible via a new `op="update_metadata"` record type
  that appends the patch under `metadata.patch` and replays by merging
  into the target entry's metadata. Compaction drops update records
  whose target is already in the snapshot. Zero impact on existing
  `store` / `forget` WAL semantics.
- **`_matches_where` null handling**: `{"field": {"$eq": None}}` now
  matches entries missing the field (interpreted as "no value here").
  Makes the default conversational-retrieve filter
  `where={"superseded_by": {"$eq": None}}` work as expected.
- **LoCoMo `--conversational` adapter**: `benchmarks/run_locomo.py
  --conversational` swaps in the new `ConversationalSomaAdapter` so
  the retrieval report compares raw-turn RAG against the
  extract+reconcile pipeline on the same conversations, with a
  "Facts / turns" column surfacing how much structure the LLM pulled
  out.

### Added — auth

- **Per-bundle JWT auth** replaces the single shared `SOMA_API_KEY`
  bearer model. A token now carries a claim of shape
  `soma: {v:1, bundles: {name: [read|write|admin]}}`; the REST
  dependency `require_auth(bundle_path_param, perm)` enforces the
  hierarchy (`write` implies `read`, `admin` implies everything) per
  route. HS256 default via `SOMA_JWT_SECRET`, RS256 opt-in via
  `SOMA_JWT_ALG=RS256` + `SOMA_JWT_PUBLIC_KEY_PATH` (keeps
  issuer/verifier split possible without shipping a secret to every
  REST worker).
- `SOMA_API_KEY` still works as a deprecated admin escape hatch;
  responses unlocked via that path carry `X-SOMA-Deprecated: use JWT`.
- New `src/soma/auth.py`: `Principal` dataclass + `issue_token()` +
  `verify_token()` + `generate_secret()`. Shared between `serve.py`
  and `cli.py`; no FastAPI imports so CLI usage stays light.
- New `soma auth` CLI: `issue` (mint token, repeatable
  `--bundle NAME:PERMS`, `--expires 30d|7d|24h|60m`), `verify` (decode
  + JSON-print claims), `rotate-secret` (fresh 32-byte urlsafe-b64).
- OpenAPI spec now advertises `securitySchemes.bearerAuth` with
  `bearerFormat=JWT`; every protected operation lists
  `{bearerAuth: []}`. `/health`, `/version`, `/metrics` stay public.
- `soma_auth_failures_total{reason}` counter (Phase 3 integration):
  `invalid_token | expired_token | insufficient_perm |
  missing_credentials`. Wired via a `record_auth_failure` helper in
  `serve.py` so every 401/403 path funnels through one place.
- Reference docs: `docs/auth.md` (quickstart, claim shape, HS256 vs
  RS256 tradeoffs, route perm map, CLI reference, rotation workflow,
  deprecation timeline for `SOMA_API_KEY`).
- `pyjwt[crypto]>=2.8` added to the `serve` extra.
- **JWT revocation blocklist** — file-backed JSONL store at
  `SOMA_JWT_BLOCKLIST_PATH` closes the "revoke a leaked token without
  nuking `SOMA_JWT_SECRET`" gap. New module
  `src/soma/auth_revocation.py` with `BlocklistBackend` Protocol,
  `RevocationRecord` dataclass, and `FileBlocklist` impl
  (portalocker-serialised writes + mtime-polled reads, ~30 s
  propagation across workers). `verify_token` gains an optional
  `blocklist=` kwarg; `serve.py` wires a module-level
  `_blocklist = blocklist_from_env()` so revoked tokens return 401
  `{"detail": "token revoked"}` with the new
  `soma_auth_failures_total{reason="revoked_token"}` counter. Unset
  env var = pre-revocation Phase 4 behaviour intact. New CLI:
  `soma auth revoke` (either `--token` or `--jti/--exp`),
  `soma auth list-revoked`, `soma auth gc`. Design memo:
  `docs/plans/2026-04-16-jwt-revocation.md`; operator reference:
  "Revocation" section of `docs/auth.md`. Redis-backed variant stays
  deferred (see `docs/plans/deferred-items.md`).

### Added — observability

- **Prometheus `/metrics` endpoint** (optional via
  `pip install "soma[metrics]"`). Exposes 14+ SOMA-specific
  counters/gauges/histograms (`soma_store_total`,
  `soma_retrieve_latency_seconds`, `soma_entries`, etc.) alongside
  the standard per-route HTTP metrics that
  `prometheus-fastapi-instrumentator` layers on automatically.
  When the extra isn't installed, `/metrics` 404s and the core
  `serve` path keeps its zero-observability-dep footprint.
- **Structured JSON logging** via `soma.log.JSONFormatter` +
  `configure_json_logging()`. Gated on `SOMA_LOG_JSON=1`. Zero new
  deps (stdlib only). Emits one JSON line per retrieve with a
  pinned schema (`event`, `bundle`, `query_len`, `k`, `has_where`,
  `hybrid_alpha`, `rerank_top_n`, `n_hits`, `backend`, `latency_ms`,
  `cache_miss`) — ready for Loki / Datadog / CloudWatch ingest with
  no format parsing.
- **OpenTelemetry tracing** (optional via `pip install "soma[otel]"`
  + `SOMA_OTEL_ENABLED=1`). Lazy-imports `FastAPIInstrumentor`, so
  the 5-package OTel dependency footprint stays truly optional.
  Uses standard `OTEL_EXPORTER_OTLP_*` env vars — no in-code
  collector config.
- `soma_reload_total{bundle}` counter fires on every
  `reload_if_stale()` call that actually picks up a peer-committed
  record — pairs with the Phase 1 multi-worker staleness fix.
- Reference docs: `docs/observability.md` (metric table, Grafana
  query examples, log schema, OTel collector config).

### Added — durability + concurrency

- **Write-ahead log** (`src/soma/memory/wal.py`). Every `store()` /
  `store_batch()` / `forget()` appends a CRC-framed record to a paired
  `memory_ops.wal.jsonl` + `memory_embeddings.wal.bin` sidecar before
  mutating in-memory state. Crashes after the append (no `save()` call)
  still replay on reload.
- **Atomic `save()`** — each output file writes to `<path>.tmp`, fsyncs,
  then `os.replace`s into place. Crash mid-save leaves the prior bundle
  untouched.
- **Schema v2** bundles — `memory_index.json` gains `schema_version: 2`
  and ships alongside the WAL sidecars. v1 bundles load in legacy mode
  and auto-upgrade to v2 on the next `save()`.
- **Durability knob** — `MemoryLayer(..., durability="sync"|"batch"|"async")`.
  `sync` (default) fsyncs every op; `batch` groups 32 ops; `async`
  never fsyncs until explicit `flush()`. Cookbook recipe 17 covers the
  tradeoffs.
- **Auto-compaction** — a background daemon thread rewrites the
  snapshot when the WAL exceeds `max(4 MB, 1.0x snapshot)` bytes,
  10 000 records, or 1 h since the last compaction. Only one
  compaction runs at a time; concurrent triggers are no-ops.
- **Multi-worker readers** — `MemoryLayer.reload_if_stale()` scans the
  WAL tail past a cursor and applies peer-worker appends on top of the
  in-memory state. Wired into `serve._get_mem()` so every cached
  bundle sees the freshest committed state before answering a request.
- **`portalocker.Lock` on `bundle.lock`** — serializes writes across
  processes sharing a bundle dir. Readers never block writers; writers
  serialize with each other.

### Added — k8s

- **Helm chart** at `deploy/helm/soma` (Chart v0.1.0, kubeVersion
  `>=1.28`). First-class peer of the Railway / Render / Fly / Docker
  paths — `helm install soma oci://ghcr.io/soma-ai/charts/soma` is
  the one-liner. Ships a `StatefulSet` (single-replica, LOCKED until
  Phase 6) with `volumeClaimTemplates` for `/app/data`, probes
  (`startupProbe` 150 s budget for sbert warmup), non-root
  securityContext, a `ClusterIP` Service on 8420, a headless Service
  for stable pod DNS, and a ServiceAccount.
- **Idempotent API-key Secret** — mints a 32-byte random key on fresh
  install; `lookup` helper preserves the value across
  `helm upgrade`; `helm.sh/resource-policy: keep` survives
  `helm uninstall`. Bring-your-own Secret via
  `api.existingSecret: <name>`; disable auth entirely with
  `api.enabled: false`.
- **Opt-in Ingress + HTTPRoute (Gateway API v1)**, mutually exclusive
  via a `{{ fail }}` guard in `_helpers.tpl`. Ingress supports
  cert-manager annotations out of the box; HTTPRoute is gated on
  `Capabilities.APIVersions.Has "gateway.networking.k8s.io/v1"`.
- **Opt-in ServiceMonitor + PodMonitor** gated on
  `monitoring.coreos.com/v1` — scrapes the Phase 3 `/metrics`
  endpoint. Plus pod-level `prometheus.io/scrape` annotations via
  `metrics.annotations.enabled`.
- **Schema-enforced invariants** — `values.schema.json` pins
  `replicaCount.maximum: 1` (defense against multi-writer WAL
  corruption) and requires `resources.requests.memory` (prevents
  the 1 Gi OOM-on-first-retrieve footgun).
- **CI** (`.github/workflows/helm.yml`) runs `helm lint` +
  `kubeconform -strict` across k8s 1.28 / 1.29 / 1.30 +
  `ct install` on `kind` for `default-values`, `ingress-enabled`,
  and `servicemonitor` test value files under `deploy/helm/ci/`.
- **Dual-channel release** (`.github/workflows/helm-release.yml`)
  on `chart-v*` tags — OCI push to `ghcr.io/soma-ai/charts/soma` +
  GitHub Pages index via `helm/chart-releaser-action@v1`. Fail-fast
  if either channel fails so the two never drift.
- Runbook: `docs/deployment-k8s.md` (prereqs, quickstart, values
  reference, secret management, persistence, observability,
  upgrading, multi-replica roadmap, troubleshooting).

### Added — cloud deploy

- One-click templates for **Railway** (`railway.json`), **Render**
  (`render.yaml`), and **Fly.io** (`fly.toml`) at the repo root. README
  carries Deploy-on-Railway / Deploy-to-Render buttons and a `fly
  launch` one-liner; full per-platform runbook (including
  DigitalOcean / VPS compose flow) in `docs/deployment-cloud.md`.
- **Dockerfile `$PORT` support** — shell-form `CMD` expands
  `${PORT:-8420}` at runtime so Railway / Render / Fly can inject
  their own port. Local `docker run` still defaults to 8420.
- **Pre-baked sbert model** in the Docker image — `all-MiniLM-L6-v2`
  is downloaded during `docker build` into `HF_HOME=/app/.cache/
  huggingface`, cutting cold-start from ~60 s to ~8 s at a ~90 MB
  image-size cost. Model is still overridable via `SOMA_EMBED_MODEL`.
- **Layer-cache friendly build** — `pyproject.toml` + `README.md`
  copied before source, so source-only edits don't re-run
  `pip install -e ".[serve]"`.
- `.dockerignore` at the repo root trims the build context (drops
  `artifacts/`, `benchmarks/reports/`, `data/`, `checkpoints/`,
  `tests/`, `docs/plans/`, `.git/`, caches).

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
