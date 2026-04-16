# Changelog

All notable changes to SOMA are documented here.

## [Unreleased] — 2026-04-16

### Added — benchmarks

- **LoCoMo QA eval with LLM-as-judge** — `benchmarks/run_locomo.py`
  gains `--run-qa-eval` / `--qa-eval-max-questions` (default 200) /
  `--judge-llm-name` flags. For each LoCoMo question a responder LLM
  answers from the retrieved context; a judge LLM compares the
  candidate against the gold annotation and returns a strict
  JSON verdict. Post-hoc scoring so every arm sees the same judge.
  Fallback to `DryRunBackend` with explicit warning when no live
  backend is reachable. New `benchmarks/harness/qa_eval.py`
  (`answer_question`, `judge_answer`, `evaluate_qa`) + full TDD
  coverage in `benchmarks/tests/test_qa_eval.py` — capturing-backend
  unit tests and an end-to-end smoke through `run_locomo.main`.
  Run with `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` set or Ollama
  running to produce real numbers.
- **Conversational threshold calibration sweep** —
  `benchmarks/run_conv_threshold_sweep.py` sweeps a 4x4 grid of
  `near_dup_threshold ∈ {0.88, 0.90, 0.92, 0.94}` x
  `ambiguous_threshold ∈ {0.65, 0.70, 0.75, 0.80}` on a LoCoMo
  subset (20 conversations), recording facts_stored, llm_calls,
  p50/p95 `add_message` latency, Recall@5, and optional QA
  accuracy per combo. Emits a markdown report at
  `benchmarks/reports/conv_threshold_sweep.md` with a
  recommendation banner picking the best cost-adjusted combination
  (accuracy-per-LLM-call, falling back to recall-per-call when QA
  eval is off). `_CountingBackend` wraps the LLM for per-combo
  cost measurement; DryRunBackend fallback documented in-report.
  Tests at `benchmarks/tests/test_conv_threshold_sweep.py` pin grid
  shape (16 rows), variance across combos, and recommendation
  banner presence. Run with `OPENAI_API_KEY` or running Ollama to
  produce real numbers — the shipped report is a DryRunBackend
  smoke run.
- **Paper draft §4.4 + §4.5** (`benchmarks/reports/paper-draft.md`):
  Recall vs QA accuracy distinction table (memory vs LLM concerns)
  and threshold calibration table + recommended-defaults section.

### Added — backends

- **Pluggable LanceDB backend** (`src/soma/memory/backends/lancedb.py`,
  optional `pip install soma[lancedb]`): embedded, arrow-native vector
  store (10M+ scale). Third adapter, filling the "local-first past
  Qdrant-local's 20K cap, no server" niche between InProc (RAM-bound)
  and Qdrant HTTP (requires infra). `supports_filter_pushdown=True`
  via `lancedb_filter.to_lancedb_where` (SQL-style predicates for
  `$eq` / `$ne` / `$gt` / `$gte` / `$lt` / `$lte` / `$in` / `$nin`).
  Snapshot = copy the LanceDB table directory into the bundle;
  restore = inverse. Schema-error-on-filter path converts to
  `FilterPushdownUnsupported` so MemoryLayer cleanly falls back to
  its Python pre-filter + `search_subset` path when the schema
  doesn't carry the referenced column.
- **Parametrized protocol contract suite**
  (`tests/test_memory/test_backend_protocol.py`): every shipped
  adapter is forced through the same basic-invariants gate (ntotal /
  dim / add / search / get_vectors / remove / clear / snapshot /
  restore / protocol isinstance). Missing optional deps skip the row.
  New adapters land with one factory entry.
- **`LanceDBBackend` adapter tests**
  (`tests/test_memory/test_lancedb_backend.py`): 30 tests covering
  identity, round-trips, filter pushdown (per operator + unsupported
  op), snapshot/restore, directory persistence, recreate semantics,
  ImportError path, and a MemoryLayer end-to-end round-trip.
- **`lancedb_filter.to_lancedb_where`** translator + 18 unit tests
  covering every operator, literal rendering (strings / numbers /
  booleans / None), apostrophe escaping, empty-list edge cases, and
  unsupported-op rejection.
- **Benchmark matrix extension**: `LanceDBFlat` and `LanceDBHNSW`
  rows added to `benchmarks/run_backend_matrix.py` and regenerated
  at 100K. Numbers land in `benchmarks/reports/backend_matrix.md`.
- **Docs**: extended `docs/backends.md` with a LanceDB section
  between InProc and Qdrant — pitch, index-type choice table,
  bundle layout, filter-pushdown semantics + fallback, tradeoffs.
- **README feature comparison**: Pluggable-vector-backends row now
  reads *yes (InProc + Qdrant + LanceDB)*.

### Added — conversational-memory ergonomics (Phase 11)

- **`extractor_llm=` kwarg on `ConversationalMemory`**: lets users
  pin a stronger model for the structured-output steps (fact
  extraction + reconcile ADD/UPDATE/SUPERSEDE/NOOP) while keeping
  chat + summary on a smaller model. Targets 3B-local users
  (local-first positioning) whose chat LLM can't reliably produce
  strict JSON. Defaults to `None` — falls back to `self._llm` with
  `self._extractor_llm = extractor_llm or llm` so call sites stay
  free of None-checks. Summary stays on `self._llm` because
  free-form prose handles small-model variance fine.
  (`src/soma/memory/conversational.py`, 8 new tests.)
- **`ConversationalSomaAdapter.__init__` forwards `extractor_llm`**
  so benchmarks can pin a stronger extractor without changing the
  harness protocol.

### Added — multi-user scoping (Phase 12)

- **`user_id=` on `ConversationalMemory`**: constructor-level default
  user + per-call override. Every turn / fact / summary write stamps
  `metadata["user_id"]` when set. Unblocks multi-tenant deploys
  (shared bundle, isolated per-user state) without a new endpoint
  — REST callers pass `user_id` in the metadata field.
- **`retrieve()` scopes by default**: when constructed with
  `user_id="alice"`, retrieval filters to alice's scope. Pass
  `user_id=None` explicitly to unscope for admin drill-downs.
  New bonus `where=` kwarg on `retrieve()` composes with the
  internal user/session/superseded filter via dict merge
  (caller keys win).
- **Safety invariants**: `supersede()` verifies the target fact's
  `user_id` matches before mutating — raises `PermissionError` on
  cross-user supersede attempts. `clear_session()` filters by the
  active user_id when set. Reconcile's candidate search scopes
  to the same user so Alice's new fact can't silently merge into
  Bob's existing fact.
- **Backward compat**: when `user_id` is unset, metadata stays
  byte-identical to pre-Phase-12. No migration needed on existing
  bundles. Localised via a `_stamp_user_id` static helper.
- **Documentation**: new §18.1 "Multi-user scoping" under the
  ConversationalMemory cookbook recipe, with the REST pattern
  (`POST /store {"metadata": {"user_id": "alice"}}`).

### Added — lazy stable-capture (Phase 13, closes Task #173)

- **`_stable_capture_dirty` flag + lazy capture**: `consolidate()`
  no longer runs the O(N_so_far) stable-capture pass eagerly. It
  flips the dirty flag and returns. The first subsequent
  `retrieve()` with `graph_rerank_alpha > 0` triggers the capture
  lazily; alpha=0 paths skip the cost entirely. Mutations
  (`store`, `store_batch`, `forget`, `clear`) re-dirty the flag so
  intervening changes get picked up on the next retrieve.
- **`MemoryLayer.stable_capture()` public API**: eager hook for
  benchmarks, warmup passes, and callers who want the cost paid
  predictably. No-op when SOMA isn't attached or the feature is
  disabled, so safe to call unconditionally.
- **`eager_stable_capture=True` on `SomaAdapter`**: default pins
  the pre-Phase-13 behaviour for existing benchmark reports
  (plasticity / longitudinal-drift / enterprise-scale) so numbers
  stay comparable commit-to-commit. New benchmark runs can opt
  into the lazy path by flipping the flag.
- **Cost impact**: removes O(N²/K) stable-capture work from
  consolidate sessions. At N=10K, K=100, that's ~1M stable steps
  per session → 0 under the default alpha=0 config. Becomes
  load-bearing if/when graph-rerank reactivates (research agenda
  §5 of paper-draft).
- **No signal regression**: bit-identical retrieval scores between
  eager and lazy paths under the same deterministic seed. 11 new
  tests pin every dirty/clean transition + the mutation-triggers.

### Added — observability loose ends (Phase 8)

- **`soma_compaction_total{bundle, outcome}`** Counter +
  **`soma_compaction_seconds{bundle}`** Histogram: consolidation
  cycles now record success/error outcome and wall-clock duration
  (buckets 0.1s → 5min). Closes the explicit Phase 3 Risks §1 gap.
  Wrapped around `MemoryLayer.consolidate()` with `try/finally` so
  exceptions still propagate unchanged while metrics still fire.
- **`SOMA_METRICS_PUBLIC=0` env gate**: when set, `/metrics` is
  wrapped with `Depends(require_auth(None, "read"))` so sensitive
  operator deploys can keep metrics behind bearer auth while
  retaining the zero-friction public-scrape default. If no JWT
  secret or API key is configured, the gate still returns 200
  (matches `require_auth`'s fast-path — operators can't
  accidentally lock themselves out).
- **`SOMA_METRICS_BUNDLE_LABEL_DISABLE=1` cardinality escape**:
  routes every `.labels(bundle=...)` call site through a
  centralised `_bundle_label()` helper that returns `"_disabled"`
  when the env is set. Lets multi-tenant deploys with 10K+ bundles
  keep the same metric surface without exploding Prometheus
  series counts. Applied across `MemoryLayer`, REST wrappers,
  and the InProc backend's FAISS index gauge.

### Added — Grafana dashboards (Phase 9)

- **Three importable dashboards** under `deploy/grafana/`:
  `soma-overview.json` (8 RED panels — request rate, error rate,
  p50/p95/p99 retrieve latency, per-route 5xx, slowest-routes
  table), `soma-auth.json` (5 panels — failures by reason,
  revoked-token hits, success rate, per-reason breakdown),
  `soma-bundle-health.json` (6 USE panels — WAL append rate,
  flush p95, consolidation p95 with dual `soma_consolidate_seconds`
  + `soma_compaction_seconds` targets, entry/FAISS size timeseries,
  peer-reload rate, retrieve latency heatmap).
- **Template variables** wired via `${DS_PROMETHEUS}` datasource
  placeholder plus `$bundle` and `$instance` drop-downs (both
  driven by `label_values` Prometheus queries).
- **Import guide** (`deploy/grafana/README.md`): four paths —
  Grafana web UI, `grafana-cli admin` + HTTP API fallback,
  docker-compose provisioning, and a Kubernetes ConfigMap
  pattern — plus smoke-test + UI-round-trip editing workflow.
- **Validation tests** (`tests/test_deploy/test_grafana_dashboards.py`):
  26 parametrized runs confirm JSON parses, `schemaVersion >= 36`,
  every target has a non-empty `expr`, every target uses the
  `${DS_PROMETHEUS}` placeholder, required `$bundle` / `$instance`
  template variables exist, panel counts match the plan.
- **Methodology**: dashboards follow the RED (Rate / Errors /
  Duration) and USE (Utilization / Saturation / Errors) patterns
  popularised in the wshobson/agents public skill; structural
  conventions adapted to SOMA's Prometheus metric names.

### Added — CLI bundle management (Phase 10)

- **New `soma bundle` subcommand group**: `list`, `info`, `delete`.
  Fills the obvious gap — the CLI had per-entry verbs (`search`,
  `forget`) and session verbs (`chat`, `index`) but nothing for
  whole-bundle lifecycle.
- **`soma bundle list [root]`**: depth-3 scan from the given root
  (default `.`), prints an aligned table with path / entries /
  embed-dim / backend / last-modified / WAL size. Detects and
  badge-marks corrupt bundles (missing `memory_index.json`, bad
  JSON, truncated) with a footnote explaining each.
- **`soma bundle info <path>`**: per-bundle detail view including
  total disk bytes, WAL state, snapshot generation timestamp.
- **`soma bundle delete <path> [--yes]`**: interactive y/N prompt
  by default (case-insensitive y|yes accepted); `--yes` skips
  the prompt. **Safety invariant**: always calls `is_bundle_dir()`
  first and exits 2 on non-bundle paths — even with `--yes`, so
  `soma bundle delete ~` is blocked.
- **New `src/soma/bundle.py` module**: stdlib-only
  `is_bundle_dir()`, `BundleInfo` frozen dataclass, `load_info()`,
  `list_bundles()` — no `MemoryLayer` / `torch` imports so
  `bundle list` stays fast across directories with hundreds of
  bundles. Reads `memory_index.json` + `backend.json` directly.
- **Tests**: +63 tests (20 in `tests/test_bundle.py` covering
  introspection helpers + corrupt-bundle handling; 43 in
  `tests/test_cli.py` covering each verb end-to-end, interactive
  prompts, and safety checks).

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
