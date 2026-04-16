# Phase 3 — Observability (Prometheus + structured JSON logs)

> **For Claude:** Execute via TDD after Phase 1 (WAL) lands. Each task has failing tests first, then minimal impl, then commit.

**Goal:** `GET /metrics` exposes Prometheus counters/gauges/histograms matching the 2026 vector-DB norm (Qdrant native, Chroma via OTel, Pinecone via Prom HTTP SD). One structured JSON log line per store/retrieve/forget. OTel spans as opt-in extra.

**Architecture:**
- `prometheus-fastapi-instrumentator` as **optional extra** (`pip install soma[metrics]`). Hand-rolled `prometheus-client` primitives for memory-layer-specific counters/histograms.
- Stdlib `logging` + ~30-line `JSONFormatter` — zero new hard deps.
- OTel via `opentelemetry-instrumentation-fastapi` behind `soma[otel]` extra + `SOMA_OTEL_ENABLED=1` gate.
- All instrumentation imports behind `try/except ImportError`; core `serve` extra works without observability deps.

**Tech stack:** `prometheus-fastapi-instrumentator>=7.1` + `prometheus-client`, stdlib `logging`, optional `opentelemetry-instrumentation-fastapi`.

---

### Task 1: JSONFormatter (stdlib, ~30 LOC)

**Files:**
- Create: `src/soma/log.py`
- Test: `tests/test_log.py`

**Step 1:** failing tests:
- `test_formatter_emits_valid_json_per_line` — `json.loads(formatter.format(record))` succeeds.
- `test_standard_fields_present` — ts, level, name, message always emitted.
- `test_extra_fields_merged` — `logger.info("event", extra={"k": "v"})` yields `{"k": "v"}` in output.
- `test_exc_info_included` — exception + traceback in output on `logger.exception(...)`.
- `test_ts_is_iso8601_utc` — ISO-8601 with `Z` suffix.

**Step 2:** implement `JSONFormatter(logging.Formatter)` that emits `{ts, level, name, message, **extra}` JSON per log line. Add module-level `configure_json_logging()` that swaps root logger's formatter when `SOMA_LOG_JSON=1`.

**Step 3:** commit `feat(log): stdlib JSONFormatter with extras + exc_info`.

### Task 2: prometheus-client primitives + metrics module

**Files:**
- Create: `src/soma/metrics.py`
- Modify: `pyproject.toml` — add `metrics = ["prometheus-fastapi-instrumentator>=7.1", "prometheus-client"]`
- Test: `tests/test_metrics.py`

**Step 1:** failing tests (skip if prometheus-client not installed):
- `test_counters_increment` — `STORE_TOTAL.labels(bundle="x").inc()` works.
- `test_histograms_observe` — `RETRIEVE_LATENCY.observe(0.1)` works.
- `test_gauges_set` — `ENTRIES.labels(bundle="x").set(42)`.
- `test_noop_when_prometheus_missing` — ImportError path returns sentinel objects with `.inc()/.observe()/.set()` that do nothing (no user crash if `soma[metrics]` not installed).

**Step 2:** implement 14 metric objects per research report §4:
- Counters: `soma_store_total`, `soma_store_batch_total`, `soma_retrieve_total{bundle,backend}`, `soma_forget_total`, `soma_faiss_rebuild_total{index_type}`, `soma_bm25_rebuild_total`, `soma_consolidate_total`, `soma_wal_append_total{op}` (Phase 1 integration).
- Gauges: `soma_entries{bundle}`, `soma_loaded_bundles`, `soma_faiss_index_size{bundle}`.
- Histograms: `soma_retrieve_latency_seconds{bundle,backend}`, `soma_embed_latency_seconds{batch_size_bucket}`, `soma_faiss_rebuild_seconds`, `soma_bm25_rebuild_seconds`, `soma_consolidate_seconds`, `soma_wal_flush_seconds`.
- Sentinel classes `NoopCounter`/`NoopHistogram`/`NoopGauge` returned when prometheus-client unavailable.

**Step 3:** commit `feat(metrics): prometheus-client primitives + noop fallback`.

### Task 3: Wire metrics into MemoryLayer

**Files:**
- Modify: `src/soma/memory/api.py` (instrument store/store_batch/retrieve/forget/_maybe_build_faiss/_maybe_build_bm25/consolidate; see research §5 for exact line pointers)
- Test: `tests/test_memory/test_metrics_integration.py`

**Step 1:** failing tests:
- `test_store_increments_counter` — after `mem.store(...)`, `soma_store_total` ≥ 1.
- `test_retrieve_observes_latency` — `soma_retrieve_latency_seconds.count` ≥ 1 after a retrieve.
- `test_faiss_rebuild_counter_fires_on_threshold_crossing` — store past `faiss_threshold`, counter increments.
- `test_retrieve_backend_label_matches_path` — linear vs faiss label differs appropriately.

**Step 2:** wrap each hot path in context managers:
```python
with soma.metrics.RETRIEVE_LATENCY.labels(bundle=self._bundle_name, backend=backend_name).time():
    ...
```
For embed latency: `soma_embed_latency_seconds.labels(batch_size_bucket=bucket(len(texts))).observe(dt)` with buckets {1, 10, 100, 1000+}.

**Step 3:** commit `feat(memory): instrument store/retrieve/forget with Prometheus metrics`.

### Task 4: JSON log line per retrieve

**Files:**
- Modify: `src/soma/memory/api.py` — add one `logger.info` at the tail of `retrieve()`
- Test: `tests/test_memory/test_retrieve_log_line.py`

**Step 1:** failing test:
- `test_retrieve_emits_single_json_line_with_schema` — capture with `caplog`; assert `json.loads(record.message)` has exactly the expected keys (`event, bundle, query_len, k, has_where, hybrid_alpha, rerank_top_n, n_hits, backend, latency_ms, cache_miss`).

**Step 2:** single `logger.info("retrieve", extra={...})` at `return candidates[:k]`. Schema per research §2.

**Step 3:** commit `feat(memory): structured JSON log line per retrieve`.

### Task 5: /metrics endpoint + instrumentator

**Files:**
- Modify: `src/soma/serve.py`
- Test: `tests/test_serve/test_metrics_endpoint.py`

**Step 1:** failing tests:
- `test_metrics_endpoint_returns_200_without_auth` — /metrics bypasses `require_api_key`.
- `test_metrics_advertises_soma_counters` — response contains `soma_store_total`, `soma_retrieve_latency_seconds`.
- `test_metrics_endpoint_tagged_system` — OpenAPI spec has `tags=["system"]`.
- `test_metrics_no_ops_when_instrumentator_missing` — monkeypatch ImportError, `/metrics` returns 501 or 404 with clear error body.

**Step 2:** in `serve.py`, after app creation:
```python
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", tags=["system"])
except ImportError:
    pass  # metrics extra not installed; /metrics 404s
```
Also call `configure_json_logging()` if `SOMA_LOG_JSON=1`.

**Step 3:** commit `feat(serve): /metrics endpoint + JSON logging init`.

### Task 6: OTel as opt-in extra

**Files:**
- Modify: `pyproject.toml` — add `otel = ["opentelemetry-instrumentation-fastapi", "opentelemetry-sdk", "opentelemetry-exporter-otlp"]`
- Modify: `src/soma/serve.py` — gated OTel init on `SOMA_OTEL_ENABLED=1`
- Test: `tests/test_serve/test_otel_optional.py` (skip when extra not installed)

**Step 1:** failing test:
- `test_otel_not_initialized_by_default` — instrumentation count stays 0 without env var.
- `test_otel_initializes_when_enabled` — `SOMA_OTEL_ENABLED=1` triggers FastAPIInstrumentor.instrument_app(app).

**Step 2:** implement the env-gated initializer. Use lazy imports.

**Step 3:** commit `feat(serve): optional OpenTelemetry tracing behind SOMA_OTEL_ENABLED`.

### Task 7: Docs

**Files:**
- Create: `docs/observability.md`
- Modify: `README.md` — add `/metrics` to the REST endpoint list
- Modify: `CHANGELOG.md`

**Step 1:** docs sections:
- Install: `pip install soma[metrics]` / `soma[otel]`.
- Metrics reference: table of all 14 metrics, labels, meanings, Grafana query examples.
- Logs: `SOMA_LOG_JSON=1` flag, sample log line, pipeline recommendations (Loki/Datadog/CloudWatch).
- OTel: env vars, collector setup, sample Jaeger trace.

**Step 2:** commit `docs: Phase 3 — observability reference`.

---

## Risks

1. **Metric cardinality explosion with many bundles.** `bundle` label on everything; 10K bundles = 10K series per metric. Mitigation: document the `SOMA_METRICS_BUNDLE_LABEL_DISABLE=1` env to drop the label for high-cardinality deploys.
2. **/metrics exposed without auth.** Standard norm (Prom scrapers don't auth), but sensitive deploys may not want bundle names leaking. `SOMA_METRICS_PUBLIC=0` could gate it behind the bearer scheme in a future stage.
3. **OTel dep footprint.** ~5 packages pulled by the extra. Keep it truly optional, never hard-dep.
4. **Race between WAL append and metric observe.** Both happen under the bundle lock; order doesn't matter but double-counting must not. Test: metric count should equal WAL record count exactly.

## Open questions

- **Histogram bucket choices.** Defaults for `soma_retrieve_latency_seconds`: {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0}. Good spread for MiniLM; may want a sub-ms floor for in-proc FAISS. Revisit after first Grafana dashboard.
- **Process-level vs per-request metrics.** FastAPI Instrumentator gives per-route HTTP metrics (counter, latency, in-flight); we layer memory-layer-specific ones on top. Document the overlap.

## Related plans

- Phase 1 — `docs/plans/2026-04-16-phase-1-wal-autosave.md` (`soma_wal_append_total` + `soma_wal_flush_seconds` depend on Phase 1 instrumentation hooks)
- Phase 4 — `docs/plans/2026-04-16-phase-4-jwt-auth.md` (Phase 4 may add `soma_auth_failures_total{reason}`)
