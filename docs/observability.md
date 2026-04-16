# Observability — Prometheus metrics, structured logs, OpenTelemetry

> For the design rationale, see `docs/plans/2026-04-16-phase-3-observability.md`.

SOMA ships three layers of observability, each opt-in:

1. **Prometheus `/metrics`** — 18+ counters, gauges, histograms covering
   every MemoryLayer hot path plus standard FastAPI per-route timings.
2. **Structured JSON logs** — one JSON line per retrieve (and via stdlib
   `logger.info(..., extra={...})` from anywhere else in SOMA).
3. **OpenTelemetry spans** — full HTTP trace-ability when the collector
   side is set up. Opt-in, disabled by default.

Everything is gated behind optional extras — the default `pip install
soma` pulls zero observability dependencies, and the core `serve` path
works with none of them installed.

## Install

```bash
pip install "soma[metrics]"           # /metrics endpoint + prometheus-client
pip install "soma[otel]"              # OpenTelemetry spans
pip install "soma[metrics,otel]"      # both
```

Enable the JSON log formatter at runtime:

```bash
export SOMA_LOG_JSON=1
uvicorn soma.serve:app --port 8420
```

## Prometheus metrics

The `GET /metrics` endpoint is exposed by
`prometheus-fastapi-instrumentator` when `soma[metrics]` is installed.
By default `/metrics` is public (standard Prom scraper practice). Set
`SOMA_METRICS_PUBLIC=0` to put it behind `Depends(require_auth)` with
a `"read"` scope requirement — useful for deploys on shared networks
where metrics are sensitive. When no JWT secret or API key is
configured, the gate still returns 200 (operators can't accidentally
lock themselves out). When the `soma[metrics]` extra is absent,
`/metrics` simply 404s.

### Metric reference

| Metric                            | Type      | Labels                   | What it means                                                                 |
| --------------------------------- | --------- | ------------------------ | ----------------------------------------------------------------------------- |
| `soma_store_total`                | counter   | `bundle`                 | `MemoryLayer.store()` calls                                                   |
| `soma_store_batch_total`          | counter   | `bundle`                 | `MemoryLayer.store_batch()` calls (one per batch, not per item)               |
| `soma_retrieve_total`             | counter   | `bundle`, `backend`      | `MemoryLayer.retrieve()` calls. `backend` ∈ {`linear`, `faiss`}               |
| `soma_forget_total`               | counter   | `bundle`                 | `forget()` calls that actually removed an entry                               |
| `soma_faiss_rebuild_total`        | counter   | `index_type`             | FAISS rebuilds, `index_type` ∈ {`flat`, `hnsw`}                               |
| `soma_bm25_rebuild_total`         | counter   | —                        | BM25 index rebuilds                                                           |
| `soma_consolidate_total`          | counter   | —                        | `consolidate()` invocations                                                   |
| `soma_wal_append_total`           | counter   | `op`                     | WAL records appended. `op` ∈ {`store`, `forget`}                              |
| `soma_reload_total`               | counter   | `bundle`                 | `reload_if_stale()` calls that picked up ≥1 peer-committed record             |
| `soma_entries`                    | gauge     | `bundle`                 | Live entries in the MemoryLayer                                               |
| `soma_loaded_bundles`             | gauge     | —                        | Count of MemoryLayer instances currently cached by the REST server            |
| `soma_faiss_index_size`           | gauge     | `bundle`                 | Entries in the active FAISS index (0 when store is on the linear path)        |
| `soma_retrieve_latency_seconds`   | histogram | `bundle`, `backend`      | Wall-clock latency of `retrieve()`                                            |
| `soma_embed_latency_seconds`      | histogram | `batch_size_bucket`      | Embed latency. Bucket ∈ {`1`, `10`, `100`, `1000`, `1000+`}                   |
| `soma_faiss_rebuild_seconds`      | histogram | —                        | Time to rebuild the FAISS index                                               |
| `soma_bm25_rebuild_seconds`       | histogram | —                        | Time to rebuild the BM25 index                                                |
| `soma_consolidate_seconds`        | histogram | —                        | Per-call consolidate duration                                                 |
| `soma_compaction_total`           | counter   | `bundle`, `outcome`      | Consolidation cycles. `outcome` ∈ {`ok`, `error`}                             |
| `soma_compaction_seconds`         | histogram | `bundle`                 | Wall-clock duration of `consolidate()` (buckets 0.1s→5min)                    |
| `soma_wal_flush_seconds`          | histogram | —                        | WAL flush duration (fsync cost)                                               |

Histogram latency buckets are hard-coded (see `src/soma/metrics.py` —
`_DEFAULT_LATENCY_BUCKETS`): `[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1,
0.5, 1.0, 5.0, +Inf]`. Tune by editing that list if your workload
spends most of its time outside that spread.

### Grafana query examples

P95 retrieve latency per bundle:

```promql
histogram_quantile(
  0.95,
  sum(rate(soma_retrieve_latency_seconds_bucket[5m])) by (bundle, le)
)
```

Store rate per bundle:

```promql
sum(rate(soma_store_total[1m])) by (bundle)
```

Live entry count:

```promql
soma_entries
```

FAISS rebuild frequency (flat vs hnsw):

```promql
sum(rate(soma_faiss_rebuild_total[10m])) by (index_type)
```

WAL append ops rate:

```promql
sum(rate(soma_wal_append_total[1m])) by (op)
```

### High-cardinality considerations

Every per-bundle metric carries a `bundle` label. If you run a
multi-tenant deployment with thousands of bundles, that produces
thousands of series per metric.

**Escape hatch:** set `SOMA_METRICS_BUNDLE_LABEL_DISABLE=1` to
collapse every `bundle` label to the literal string `"_disabled"`
across all SOMA metrics. You keep the same metric names (so dashboards
don't break) but Prometheus sees a single series per metric. Trade off
per-bundle drill-down for cardinality control.

## Structured JSON logs

`soma.log.configure_json_logging()` swaps the root logger's handler
formatters to `soma.log.JSONFormatter` when `SOMA_LOG_JSON=1`. The
`soma.serve` module calls this at import time, so `uvicorn
soma.serve:app` picks up the formatter before the first request.

### One log line per retrieve

Every `MemoryLayer.retrieve()` call emits exactly one `INFO` log
record on the `soma.memory` logger with this schema:

```json
{
  "ts": "2026-04-16T14:22:07.811234Z",
  "level": "INFO",
  "name": "soma.memory",
  "message": "retrieve",
  "event": "retrieve",
  "bundle": "acme_prod",
  "query_len": 42,
  "k": 5,
  "has_where": true,
  "hybrid_alpha": 0.3,
  "rerank_top_n": null,
  "n_hits": 5,
  "backend": "faiss",
  "latency_ms": 4.127,
  "cache_miss": false
}
```

**Schema is stable.** Adding keys is safe; renaming or removing keys
is a breaking change (locked to
`tests/test_memory/test_retrieve_log_line.py`).

### Pipeline recommendations

- **Loki**: set the uvicorn systemd unit's stdout to a journald pipe;
  Promtail's `json` stage auto-parses each line into structured
  labels. Query:
  `{job="soma"} | json | event = "retrieve" | latency_ms > 100`.
- **Datadog**: the official Python log integration picks up the
  formatter's output verbatim. `ts` maps to Datadog's `@timestamp`
  automatically.
- **CloudWatch**: ship via the CloudWatch agent's `file` input; CW
  Logs Insights `@message` will already be structured so filter
  with `fields @timestamp, @message | filter event = "retrieve"`.
- **stdlib**: `python -m json.tool` on the live log stream is the
  zero-dep local dev story.

### Adding your own structured fields

Call stdlib logging from anywhere; `extra` kwargs are merged straight
into the JSON object:

```python
import logging
logger = logging.getLogger("soma.user")
logger.info("ingest_batch_done", extra={"n": 1000, "src": "mem0"})
```

Output when `SOMA_LOG_JSON=1`:

```json
{"ts": "...", "level": "INFO", "name": "soma.user",
 "message": "ingest_batch_done", "n": 1000, "src": "mem0"}
```

## OpenTelemetry (opt-in)

Install the extra and flip the env var:

```bash
pip install "soma[otel]"
export SOMA_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
export OTEL_SERVICE_NAME=soma
uvicorn soma.serve:app --port 8420
```

SOMA uses `opentelemetry-instrumentation-fastapi`'s
`FastAPIInstrumentor.instrument_app(app)` at startup, so every HTTP
request automatically produces a span tagged with route, status, and
duration. The OTLP exporter is configured purely from standard
`OTEL_*` env vars — SOMA does not wrap or override them.

### Collector setup (minimal)

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  otlphttp/jaeger:
    endpoint: http://jaeger:4318

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlphttp/jaeger]
```

Point Jaeger (or Tempo, Datadog APM, etc.) at the collector. Each
request shows up as a span with attributes
`http.method`, `http.route`, `http.status_code` and timing.

### Overlap with Prometheus

FastAPI Instrumentator also exposes per-route HTTP metrics
(`http_requests_total{method,handler,status}`, `http_request_duration_seconds`).
Those are separate from SOMA's `soma_*` metrics — the HTTP series
tell you *route-level* health, the `soma_*` series tell you
*memory-layer* health. Both are scraped from `/metrics` in one shot.

## Troubleshooting

- **`/metrics` 404s despite `soma[metrics]` install** — the import
  happens at module load. Restart your `uvicorn` worker after
  installing the extra.
- **No JSON lines in stdout** — confirm `SOMA_LOG_JSON=1` is in the
  process env before `configure_json_logging()` runs. In systemd
  unit files, put it in `Environment=`; in docker, in the compose
  `environment:` block.
- **OTel warnings but no spans** — the env var SWITCH flips
  `_otel_enabled`, but the *exporter* still needs a collector URL.
  Set `OTEL_EXPORTER_OTLP_ENDPOINT` or the SDK silently drops
  spans.
- **High cardinality on `soma_entries`** — one series per bundle
  is the design; see §High-cardinality considerations above.

## Sample Grafana dashboards

Three importable dashboards ship under `deploy/grafana/`:

- **`soma-overview.json`** — RED (Rate / Errors / Duration) across
  the REST surface. Stats for total requests, p95 retrieve latency,
  auth failure rate; time series for per-route request rate,
  p50/p95/p99 retrieve latency, per-bundle store rate, 5xx rate;
  slowest-routes table.
- **`soma-auth.json`** — auth-focused. Failures by reason (stacked),
  revoked-token hits, blended success rate, per-reason table. Pairs
  with the Phase 4 JWT + Phase 4-followup revocation metrics.
- **`soma-bundle-health.json`** — USE (Utilization / Saturation /
  Errors) for the memory layer. WAL append rate, flush p95,
  consolidation p95 (dual-sourced from `soma_consolidate_seconds`
  and `soma_compaction_seconds`), live-entries / FAISS-index-size
  timeseries, peer-reload rate, retrieve-latency heatmap.

All three declare `${DS_PROMETHEUS}` as their datasource placeholder
and templated `$bundle` / `$instance` variables. Import via the
Grafana web UI, `grafana-cli admin`, docker-compose provisioning, or
a Kubernetes ConfigMap — see `deploy/grafana/README.md` for each
path and a smoke-test procedure.

## Related plans

- Phase 1 — `docs/plans/2026-04-16-phase-1-wal-autosave.md`
  (`soma_wal_append_total`, `soma_wal_flush_seconds`, `soma_reload_total`)
- Phase 4 — `docs/plans/2026-04-16-phase-4-jwt-auth.md`
  (`soma_auth_failures_total{reason}`)
- Phase 8 — `docs/plans/2026-04-16-phase-8-observability-loose-ends.md`
  (`soma_compaction_total`, `soma_compaction_seconds`,
  `SOMA_METRICS_PUBLIC`, `SOMA_METRICS_BUNDLE_LABEL_DISABLE`)
- Phase 9 — `docs/plans/2026-04-16-phase-9-grafana-dashboards.md`
  (sample dashboards under `deploy/grafana/`)
