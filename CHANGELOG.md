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

### Added — ephemeral / RAM-only mode (Phase 19)

- **`MemoryLayer.ephemeral()` classmethod**: first-class idiom for
  "RAM-only, no WAL, persist only on `.save()`". Right for notebooks,
  REPLs, and short agent runs where durability is paid at session
  end. Supports either a custom `embed_fn` + `embed_dim` or an
  `sbert_model` shorthand. WAL stays the default for long-lived
  services via the normal `MemoryLayer(...)` constructor.
- **`soma chat --ephemeral` + `--save-on-exit PATH`** CLI flags.
  `--ephemeral` starts a fresh in-RAM MemoryLayer (mutually exclusive
  with `--bundle`). `--save-on-exit` registers an `atexit` handler
  that calls `.save(path)` when the process exits — works with or
  without `--ephemeral`. Exceptions in the save hook are swallowed
  with a warning so they don't mask the underlying exit cause.
- **`POST /snapshot` REST endpoint**: one-shot bundle dump from a
  running server. Body `{"path": "..."}` → `200 {"saved": true,
  "path": "...", "entries": N}`. Rejects paths that escape the
  server's cwd (safety belt). Requires `write` perm when JWT auth
  is on; matches existing endpoint pattern.
- **+16 tests** across `test_ephemeral.py`, `test_cli.py`,
  `test_serve_smoke.py`.

### Added — Redis revocation blocklist (Phase 20)

- **`RedisBlocklist`** alongside `FileBlocklist` for multi-host /
  k8s deploys where the file-backed store's 30s poll lag is too
  slow. `setex(key, ttl=exp-now, reason)` gives instant cross-worker
  propagation and automatic expiry without a manual `gc_expired`.
  Same `BlocklistBackend` Protocol so existing callers don't change.
- **Optional extra**: `pip install "soma[redis-revocation]"` pulls
  `redis>=5.0`. Missing dep raises `ImportError` with a clear
  install hint at construction time.
- **`SOMA_JWT_BLOCKLIST_REDIS_URL`** env var wires Redis into
  `blocklist_from_env`. Resolution order: Redis URL > file path >
  null (behaviour unchanged when env vars unset). Warning logged if
  both URL and file-path env vars are set (Redis wins).
- **Hashed-mode interop verified**: `FileBlocklist(hashed=True)` and
  `RedisBlocklist(hashed=True)` produce the same lookup key for the
  same jti — operators can migrate file→Redis with a straight re-key.
- **Tests via `fakeredis`**: module-level `pytest.importorskip` so
  CI without the dev dep skips cleanly. +14 tests (11 Redis unit,
  3 env-dispatch).
- **Docs**: new Redis section in `docs/auth.md` with when-to-use
  table, env vars, minimal docker-compose snippet.

### Added — benchmark regression CI (Phase 21)

- **`scripts/check_bench_regressions.py`**: CLI that diffs a
  `--current` JSON against a committed `--golden` JSON with
  per-metric tolerances (±20% latency, ±5% recall by default). Exits
  1 with a diff table on regression, 0 on pass. Auto-creates the
  golden on first run if missing (bootstrap).
- **JSON sidecar output** on `run_scale_vs_chroma.py` and
  `run_retrieval.py` — matches the existing pattern in
  `run_backend_matrix.py`. New `--json-out` flag and `--lite` mode
  for CI-sized runs. Markdown output byte-identical in the default
  path.
- **`benchmarks/golden/` directory** with initial snapshots captured
  from live harness runs against `main` at `d54621e`. Three JSONs
  (scale_vs_chroma, retrieval, backend_matrix) + README documenting
  provenance and regeneration steps.
- **Gitea Actions workflow** (`.gitea/workflows/bench-regression.yml`):
  triggers on PRs touching `src/soma/memory/**`, `src/soma/io/**`,
  `benchmarks/**`, or the workflow itself; nightly cron at 08:00
  UTC; manual dispatch. Non-blocking on PRs (posts comment on
  regression); blocking on nightly so main gets fail-loud surfaces.
- **+5 tests** (4 plan-required + 1 Diff round-trip). Test count
  in `tests/test_scripts/` now 39 pass.
- **Known follow-up**: the CI workflow installs `[dev,metrics,ann,sbert]`
  + `chromadb` — a dedicated `[bench]` extra in `pyproject.toml`
  would tidy that. Parked since it didn't block anything and would
  have conflicted with Phase 20's concurrent `pyproject.toml` edit.

### Added — async extraction mode (Phase 22)

- **`ConversationalMemory(extraction_mode="async")`**: `add_message`
  returns as soon as the raw turn is persisted; LLM extraction +
  reconcile runs on a `ThreadPoolExecutor(max_workers=1)` owned by
  the instance. Unlocks low-latency chat paths where the caller
  doesn't need facts ready synchronously. Default stays `"sync"` —
  existing behaviour unchanged.
- **`flush(timeout=None)`** drains all pending futures and re-raises
  any exceptions from the executor thread. **`close()`** flushes
  then shuts the executor down (idempotent). Context-manager
  protocol (`__enter__` / `__exit__`) wired to `close()`.
- **`clear_session()` now flushes first** so no late facts land
  after a wipe. Summary rollovers stay synchronous (they read
  accumulated memory state, which may lag behind in async mode —
  documented tradeoff).
- **`max_workers=1`** preserves monotonic extraction order within a
  session. Tests pin start/done interleaving (no out-of-order
  completion). Python GIL means the win is I/O overlap with the
  LLM call, not CPU parallelism — called out in the recipe.
- **Docs**: new §18.1 in `docs/cookbook.md` covers when to reach
  for it (low-latency chat), the context-manager idiom, and the
  flush-exception surfacing semantics.
- **+8 tests** in `tests/test_memory/test_conversational_async_extraction.py`
  covering fast return, flush drain, context-manager close,
  in-order extraction, clear_session flush-first, exception
  surfacing, default-sync unchanged, double-close safety.

### Added — refresh-token endpoint (Phase 23)

- **`soma.auth.refresh_token(current_token, ...)`**: verifies the
  presented token end-to-end (signature, exp, revocation, audience)
  and mints a fresh one with the same `sub` / `bundles` / `aud` but
  a new `jti` and `exp`. Default new-TTL reuses the original
  `exp − iat` window; override via `new_expires_in=`. Raises the
  same exceptions as `verify_token` on bad inputs.
- **`POST /auth/refresh`** endpoint in `src/soma/serve.py`. Reads
  the current bearer, calls `refresh_token`, returns `{"token": ...,
  "exp": <unix>}`. `SOMA_JWT_REFRESH_TTL` (`30d`/`24h`/`60m`)
  overrides the default window; `SOMA_JWT_MAX_TTL` caps it. All
  failure modes return 401 with `WWW-Authenticate: Bearer` and
  route through `record_auth_failure()` so the existing counter
  reasons (`expired_token` / `revoked_token` / `invalid_token`)
  keep their semantics.
- **Old token NOT auto-revoked** — deliberate; rotation is a
  separate operator choice via `POST /auth/revoke`. Pinned in a
  test so this stays explicit.
- **`soma auth refresh --token <jwt> [--expires 60m]`** CLI verb
  mirrors the endpoint. HS256 reuses `SOMA_JWT_SECRET`; RS256 needs
  both the public and private key paths (verification-only
  deployments can't sign a new token and get a clear error).
- **`soma.auth.parse_ttl_spec`** helper extracted so the CLI,
  server, and refresh_token all parse `30d|24h|60m` the same way.
- **+21 tests** — 10 auth-core (round-trip, expired, revoked, aud,
  default TTL, no-auto-revoke), 7 serve (happy path, 401 paths,
  TTL env, aud preservation), 4 CLI (round-trip, expired, missing
  secret, --expires override).

### Added — Qdrant cross-version snapshot tests (Phase 24)

- **`tests/test_memory/test_qdrant_version_compat.py`**:
  testcontainers-driven matrix spinning up Qdrant `1.11.3`, `1.12.4`,
  `1.13.5`, taking a snapshot against one version and restoring into
  another (3×3 = 9 cross-version cases + 3 smoke = 12 total).
  Validates that `backend.json` + snapshot-recover round-trip stays
  consistent across supported versions; retrieval on the target
  returns the expected top-1 id with a deterministic hash embed.
- **Gated by `SOMA_QDRANT_VERSION_MATRIX=1`** and the
  `slow_qdrant` pytest marker so the default `pytest` run stays
  unchanged (the matrix takes 6–10 minutes and needs Docker).
  Module-level skip fires cleanly when `testcontainers` /
  `qdrant-client` / `requests` are missing or the gate is unset.
- **`pip install soma[qdrant-test]`** — new optional extra pulls
  `testcontainers>=4`. `slow_qdrant` marker registered under
  `[tool.pytest.ini_options].markers`.
- **Docs**: new "Cross-version snapshot testing" section in
  `docs/backends.md` documenting the rationale, matrix scope, 1.11+
  version floor, and how to run it locally.
- **CI workflow intentionally deferred** — test infrastructure is
  in place; wiring a weekly Gitea Actions job can follow once the
  Unraid runner has Docker-in-Docker (or we stand up a dedicated
  runner with Docker available).

### Added — batch extraction mode (Phase 25)

- **`ConversationalMemory(extraction_mode="batch", batch_size=K)`**:
  accumulates K consecutive turns in a pending buffer, then extracts
  facts from all of them in a single LLM call. Cuts extract-path LLM
  cost ~K× for workloads that tolerate fact-availability lagging by
  up to K turns. Default `batch_size=8`. Synchronous but aggregated;
  not combined with `"async"` in this phase.
- **`_flush_batch()` fires inline** when the buffer reaches
  `batch_size`; partial buffers drain via `flush()`, `close()`, and
  `__exit__`. `clear_session()` *drops* the pending buffer without
  extracting — wiping means wiping, don't write facts the user is
  trying to forget.
- **`BATCH_EXTRACT_PROMPT`** in `conversational_prompts.py`: numbered
  turn list in, JSON array of `{turn_index, category, text}` out.
  Per-fact `source_turn_id` metadata stamped from the original
  batch tuple so downstream consumers can trace a fact back to its
  source turn. Missing / out-of-range `turn_index` falls back to
  the last turn + WARNING log (conservative, never lossy).
- **Sync mode gets `source_turn_id` for free**: the same plumbing
  change that routes turn ids through the batch path also stamps
  them on sync-mode facts. Backward-compatible — `supersede()` and
  older callers don't pass it and their metadata is byte-identical
  to before.
- **+10 tests** in `tests/test_memory/test_conversational_batch_extraction.py`
  covering accumulate-until-K, flush drains partial, context-manager
  flush-on-exit, clear-drops-pending, turn-order preserved in facts,
  exceptions-surface-on-trigger-call, sync-and-async-unchanged,
  batch_size validation, routing with explicit turn_index, fallback
  on missing turn_index.

### Added — per-token rate limiting (Phase 26)

- **`src/soma/rate_limit.py`**: in-process token-bucket rate limiter
  for authenticated requests. `TokenBucket` + `RateLimiter` primitives
  with per-key isolation and monotonic-clock math (NTP slew / DST /
  manual wall-clock rewinds can't corrupt accounting).
- **`SOMA_RATE_LIMIT_RPS`** (float) enables the limiter.
  `SOMA_RATE_LIMIT_BURST` (int, defaults to `ceil(rps)`) caps burst.
  `SOMA_RATE_LIMIT_SCOPE=per-token` (default) or `per-subject`
  decides whether refreshes of the same token share a bucket. All
  unset → limiter disabled, behaviour unchanged.
- **Middleware wired inside `require_auth`** — not a per-route
  decorator. Every authenticated path (including the legacy
  `SOMA_API_KEY` escape hatch, which gets its own jti=`legacy`
  bucket) is covered automatically. `/metrics` and `/health` are
  exempt so ops tooling can't lock itself out.
- **`soma_rate_limited_total`** counter (label: `scope`). No
  per-jti / per-subject label — cardinality would explode. The
  actionable dashboard query is "total 429s per minute".
- **Idle eviction** after `idle_evict_seconds` (default 300s) of
  silence on a key, so dead tokens don't leak memory.
- **+23 tests** (15 in `tests/test_rate_limit.py` for primitives,
  8 in `tests/test_serve/test_rate_limit_serve.py` for HTTP
  integration). Zero real sleep — all time-sensitive tests inject
  `now=` explicitly.
- **Docs**: new "Rate limiting" section in `docs/auth.md` — env
  vars, when to prefer the in-proc limiter vs a reverse proxy,
  per-token vs per-subject tradeoff.

### Changed — Gitea Actions migration

- **Moved `.github/workflows/*` → `.gitea/workflows/*`**
  (`bench-regression.yml`, `client-ts.yml`, `helm.yml`,
  `helm-release.yml`). SOMA repos are hosted on Gitea; keeping the
  workflows in the Gitea-native location avoids ambiguity and lets
  Gitea Actions pick them up directly.
- **`bench-regression.yml`**: latency tolerance widened to ±50%
  temporarily. Goldens were captured on a Windows 3090 host but the
  Gitea runner is CPU-only Intel on Unraid — tighten back after
  re-capturing goldens on the runner.
- **`client-ts.yml`**: dropped `npm publish --provenance` (GitHub
  OIDC only) in favour of the existing `NPM_TOKEN` secret. Package
  still publishes to npmjs.org on tag push; signed-provenance badge
  is no longer emitted.
- **`helm-release.yml`**: tag-push trigger disabled. Downstream
  publish steps still target `ghcr.io` + GitHub Pages via
  `helm/chart-releaser-action`; re-enable after reworking to
  Gitea's OCI package registry when we cut the first chart tag.

### Added — backend perf (Phase 16)

- **`VectorBackend.search_near_id(node_id, k, exclude_self=True)`**:
  new Protocol method for "given a stored node_id, find its k
  nearest neighbours" in one round-trip. Replaces the pre-Phase-16
  pattern where `MemoryLayer.related()` did `get_vectors([id]) →
  search(vector, k+1) → strip self`, paying two HTTP round-trips on
  remote backends.
- **Default impl** (`_default_search_near_id` in `backend.py`)
  preserves today's behaviour — adapters with no fast path delegate
  and the Protocol surface stays pure. InProc uses the default
  (in-RAM; round-trip is meaningless).
- **Qdrant override** calls `client.recommend(positive=[point_id],
  limit=k)` — server-native for this exact operation. The recommend
  API always omits the positive points from results, so
  `exclude_self=True` is the cheap default path; `exclude_self=False`
  synthesises a top self-hit with score 1.0 for contract uniformity
  with InProc.
- **LanceDB override** pulls the pivot row via `to_arrow` (stays in
  Arrow, no Python ser/deser) then feeds the FixedSizeList straight
  into `table.search`. Self-exclusion pushed down as `WHERE id != ?`
  so the planner skips the pivot before distance computation.
- **`MemoryLayer.related()` routed through the new method** — one-line
  change because the default impl matches today's behaviour.
- **+21 tests** covering the contract across all three shipped
  adapters (via the parametrized `shipped_backend` fixture) plus
  Qdrant-specific mock-based fast-path verification (proves the
  override takes exactly one `recommend` call, zero `retrieve`/`search`).

### Added — ConversationalMemory drift control (Phase 17)

- **`resummarize_every` kwarg** on `ConversationalMemory` (default 5):
  every Mth summary is re-derived from raw turns (via new
  `RESUMMARY_PROMPT`) instead of chaining off the previous summary.
  Prevents compounding hallucinations and omissions on long-lived
  sessions. `resummarize_every=0` disables (always chain — today's
  behaviour).
- **New `RESUMMARY_PROMPT`** (`src/soma/memory/conversational_prompts.py`)
  explicitly instructs the LLM not to reference prior summaries,
  focuses on stable facts (names, locations, preferences, goals) +
  decisions / commitments.
- **`metadata["resummary"] = bool`** on every stored summary so the
  audit trail distinguishes chained vs re-derived rolls.
- **Cookbook recipe** in `docs/cookbook.md` §18 on when to tune the
  cadence (shorter M for chatty agents; 0 for cost-constrained
  local-LLM deploys).
- **+5 tests**. Backward-compat: existing
  `ConversationalMemory(memory=..., llm=...)` callers gain the new
  kwarg with a sensible default; no adapter changes needed.

### Added — auth hardening (Phase 18)

- **Hashed-token blocklist** — `FileBlocklist(path, hashed=True)`
  stores `sha256(jti)` instead of the plain jti. The revocation
  file becomes safe to exfiltrate: it leaks *whether* a jti is
  revoked, not *which*. Default remains `hashed=False` (byte-
  identical output to pre-Phase-18). Dual-schema reader accepts
  both legacy `{"jti": ...}` and new `{"jti_key": ...}` records on
  load so operators can flip the flag without migrating existing
  files. `gc_expired` preserves whichever key scheme each record
  originally used.
- **`SOMA_JWT_BLOCKLIST_HASHED=1`** env var wires the hashed flag
  through `blocklist_from_env`.
- **Audience claim (`aud`)**: `issue_token(..., audience="svc-A")`
  populates the claim; `verify_token(..., expected_audience="svc-A")`
  enforces it. Both kwargs default to `None` so existing callers
  behave identically. Unlocks multi-server fleets sharing one JWT
  signing secret without cross-service token replay.
- **pyjwt pin**: pyjwt 2.x actively *rejects* tokens carrying `aud`
  when the verifier passes `audience=None` (raises `InvalidAudienceError`).
  Worked around by forcing `options={"verify_aud": expected_audience
  is not None}` so unset-on-verify means "skip the check", matching
  the invariant the rest of the code contract expects.
- **`soma auth issue --audience svc-A`** CLI flag.
- **`SOMA_JWT_AUDIENCE=svc-A`** env var on the server, plumbed into
  `verify_token` via `require_auth`.
- **+12 tests** across revocation / auth-core / CLI / serve.

### Changed — docs refresh (Phase 14)

- **README rewrite** for the post-push feature set. Replaced the
  stale "drop-in vector DB replacement" pitch with the local-first
  agent-memory framing from `docs/positioning.md`. Feature
  comparison table now includes ConversationalMemory, multi-user
  scoping, JWT auth + revocation, Grafana dashboards, LanceDB
  backend, and the `soma bundle` CLI group. Install extras list
  updated (`metrics`, `otel`, `qdrant`, `lancedb`); deduplicated
  legacy entries. Trimmed from 305 to 181 lines.
- **Quickstart rewrite** (`docs/quickstart.md`) — end-to-end
  agent-memory flow in 10 copy-paste-runnable steps: install
  extras, start the server, mint a JWT, create a bundle, use
  ConversationalMemory with optional `extractor_llm=`, pass
  `user_id` through metadata for multi-user scoping, retrieve
  with filters, check bundle state, revoke a leaked token,
  import a Grafana dashboard.
- **Cookbook +2 recipes**: §19 LanceDB backend (with correct
  `LanceDBBackend(path=, dim=, index_type=)` ctor + MemoryLayer
  wiring), §20 Prometheus + Grafana operational recipe (dashboard
  list, PromQL reference, JSON-logs env var).
- **Positioning refresh** (`docs/positioning.md`): swapped stale
  benchmark bullets (2.8× store / 85 KB vs 1920 KB) for current
  numbers from `scale_enterprise_100k.md` and `backend_matrix.md`.
  Deleted the Stage-2/3/4 roadmap block (every item shipped);
  added five new feature-comparison-matrix rows (conversational
  extract/reconcile, multi-user scoping, JWT auth/revocation,
  Grafana, pluggable backends).

### Changed — polish audit + small cleanup (Phase 15)

- **mypy clean on `src/soma/memory/api.py`**: resolved 9 pre-existing
  errors concentrated on the `has_encoder` Optional-narrowing
  pattern. Bound `encoder = self._encoder` local variables so mypy
  narrows the union, replaced boolean `has_encoder` guards with
  `is not None` checks on the local, annotated `_COMPARE_OPS` as
  `dict[str, Callable[[Any, Any], bool]]`, added explicit `None`
  guard on `model.get_sentence_embedding_dimension()` (sbert's
  declared Optional return). Zero behavioural change.
- **TypeScript retry middleware** (`clients/typescript/src/retry.ts`)
  — standalone `withRetry(fetchImpl, opts)` wrapper around any
  fetch implementation, re-exported from `@soma-ai/client`'s index.
  Options: `maxRetries` (default 3), `backoff` (`"linear"` |
  `"exponential"`, default exponential), `initialDelayMs` (default
  200), `retryOn` (predicate, default retries on 502/503/504).
  Does NOT retry 4xx by default. 11 new tests covering retry
  behaviour, give-up-after-max, custom predicates, 4xx skip.
- **Audit pass — nothing else actionable.** AST-scanned `src/`
  for duplicate method defs (only false-positive in
  `llamaindex.py`'s mutually-exclusive `_HAS_LLAMAINDEX` branches),
  grepped for stale metric name refs post-Phase 9 (all clean in
  `src/` / `tests/` / `deploy/grafana/`), checked new files
  (`bundle.py`, `auth_revocation.py`, `lancedb.py`) for
  error-handling consistency with surrounding code (all match).
  Reassuring signal that the parallel-agent push didn't leave
  major artifacts.

### Added — 1M enterprise-scale benchmark row (Task #174)

- **`benchmarks/reports/scale_enterprise_1000000.md`**: index-only
  methodology at 1M entries. Headline — SOMA 5.0s store vs Chroma
  4.2 hrs (~3000× faster); SOMA-HNSW 4.44 ms retrieve vs Chroma
  127 ms (28.6× faster); 1.69 GB vs 2.37 GB on disk (1.41× smaller).
  Recall@5 actually favours SOMA-flat (0.270) over Chroma (0.230)
  at this scale — more topic-coherent ordering under linear scan.
- **`paper-draft.md` §3.3 extended** with the 1M row + findings
  paragraph tying the mechanics delta to real-time chat-turn
  budgets (4.44 ms retrieve at 1M is still well under a frame
  budget).

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
