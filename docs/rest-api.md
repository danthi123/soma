# SOMA REST API reference

> **Live spec:** `GET /openapi.json` on any running SOMA server returns
> the full OpenAPI 3.1 schema. A committed snapshot lives at
> [`clients/typescript/openapi.json`](../clients/typescript/openapi.json)
> — regenerated in CI from a stub-embedder server on every PR to
> `src/soma/serve.py`. Treat the live spec (or the committed snapshot)
> as authoritative for request/response shapes; this page summarises
> the surface.
>
> **Run it:** `soma serve --port 8420` (or `uvicorn soma.serve:app --port 8420`).

## Base URL

`http://{host}:{port}` — port defaults to `8420`. Multi-tenant routes
are prefixed with `/bundles/{name}` and route to per-bundle
`MemoryLayer` instances under `$SOMA_BUNDLES_DIR`. Default-bundle
routes (no `/bundles/{name}` prefix) operate on a single bundle at
`$SOMA_BUNDLE_PATH`.

## Authentication

See [`auth.md`](auth.md) for the full flow. Short version:

| Mode | Trigger | Header to send |
| --- | --- | --- |
| **Open (dev)** | no env vars set | *(omit Authorization)* |
| **JWT (recommended)** | `SOMA_JWT_SECRET` (HS256) or `SOMA_JWT_PUBLIC_KEY_PATH` (RS256) | `Authorization: Bearer <JWT>` |
| **Legacy (deprecated)** | `SOMA_API_KEY` | `Authorization: Bearer <key>` |

The server accepts a JWT and the legacy key in parallel. A valid JWT
always wins; an invalid JWT falls through to the legacy check before
returning 401. Responses from the legacy path carry an
`X-SOMA-Deprecated: use JWT` header — watch for it in logs.

**Permissions** (JWT claim `soma.bundles[name]`):

- `read` — `GET /get/{id}`, `GET /recent`, `GET /related/{id}`, `GET /status`, `POST /retrieve`.
- `write` — everything under `read`, plus `POST /store`, `POST /store_batch`, `POST /forget`, `POST /consolidate`, `POST /save`, `POST /snapshot`.
- `admin` — reserved (currently equivalent to `write` on every bundle in the claim).

Routes not tenant-scoped (`GET /status` on the default bundle,
`POST /retrieve`, etc.) require the perm on *any* bundle in the claim —
this keeps the token model uniform without a "default" special case.

## Route map

Every route below is implemented by a decorator in
[`src/soma/serve.py`](../src/soma/serve.py); the `operation_id`
matches the OpenAPI `operationId` for generated clients.

### Liveness / metadata (public)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe. Returns `{"ok": true, "loaded_bundles": int}`. |
| `GET` | `/version` | Package version string. |
| `GET` | `/metrics` | Prometheus scrape. Only mounted when `soma-memory[metrics]` is installed; set `SOMA_METRICS_PUBLIC=0` to gate on auth. |
| `GET` | `/openapi.json` | FastAPI-generated OpenAPI 3.1 schema. |
| `GET` | `/docs` | Swagger UI (FastAPI default). |
| `GET` | `/redoc` | ReDoc UI (FastAPI default). |

### Auth

| Method | Path | Perm | Purpose |
| --- | --- | --- | --- |
| `POST` | `/auth/refresh` | valid JWT | Mint a refreshed token preserving `soma` claims. Body carries the old JWT; response carries the new one. See `docs/auth.md` "Refresh" section. |

### Default bundle (single-tenant at `$SOMA_BUNDLE_PATH`)

| Method | Path | Perm | Purpose |
| --- | --- | --- | --- |
| `GET` | `/status` | `read` | Entry count, disk bytes, backend name. |
| `POST` | `/store` | `write` | Store one entry. |
| `POST` | `/store_batch` | `write` | Store N entries in one call (amortises embed cost). |
| `POST` | `/retrieve` | `read` | Hybrid retrieval with optional `where`, `hybrid_alpha`, `rerank_top_n`. |
| `GET` | `/get/{node_id}` | `read` | Fetch one entry by id. 404 if missing. |
| `GET` | `/related/{node_id}` | `read` | Graph-adjacent neighbours (uses the plastic-graph substrate). |
| `GET` | `/recent` | `read` | Last N entries in insertion order. |
| `POST` | `/forget` | `write` | Delete by id or by criteria (GDPR cascade, see `docs/gdpr.md`). |
| `POST` | `/consolidate` | `write` | Trigger one consolidation cycle. |
| `POST` | `/save` | `write` | Snapshot bundle to disk. |
| `POST` | `/snapshot` | `write` | Named snapshot (blue/green rollout pattern). |

### Multi-tenant (per-bundle, at `$SOMA_BUNDLES_DIR/{name}/`)

Every default-bundle route above is also exposed at
`/bundles/{name}/<same path>` with the per-bundle permission check.
The bundle name must match `[A-Za-z0-9_.-]{1,64}`. A single `soma
serve` process hosts many bundles — they load lazily on first request
and cache in memory.

| Method | Path | Perm | Purpose |
| --- | --- | --- | --- |
| `GET` | `/bundles/{name}/status` | `read` | — |
| `POST` | `/bundles/{name}/store` | `write` | — |
| `POST` | `/bundles/{name}/store_batch` | `write` | — |
| `POST` | `/bundles/{name}/retrieve` | `read` | — |
| `GET` | `/bundles/{name}/get/{node_id}` | `read` | — |
| `GET` | `/bundles/{name}/related/{node_id}` | `read` | — |
| `GET` | `/bundles/{name}/recent` | `read` | — |
| `POST` | `/bundles/{name}/forget` | `write` | — |
| `POST` | `/bundles/{name}/consolidate` | `write` | — |
| `POST` | `/bundles/{name}/save` | `write` | — |
| `POST` | `/bundles/{name}/snapshot` | `write` | — |

## Request / response shapes

The Pydantic models in `src/soma/serve.py` are the source of truth;
the OpenAPI spec reflects them verbatim. Highlights:

- **`POST /store`**: body `{"text": str, "metadata": dict}` →
  response `{"node_id": str}`.
- **`POST /store_batch`**: body `{"texts": list[str], "metadatas":
  list[dict] | null}` (metadatas is optional) → response
  `{"node_ids": list[str]}`.
- **`POST /retrieve`**: body `{"query": str, "k": int, "where":
  dict | null, "hybrid_alpha": float | null, "rerank_top_n":
  int | null}` → response `{"hits": list[Hit]}` where each `Hit` is
  `{"node_id": str, "text": str, "score": float, "metadata": dict,
  "timestamp_step": int}`.
- **`POST /forget`**: body accepts either `{"node_id": "<id>"}` for
  a single-id delete OR a criteria object for a GDPR cascade
  delete. Criteria fields (any one or more; mutually exclusive with
  `node_id`): `text_matches: str`, `subject: str`, `user_id: str`,
  plus modifiers `case_sensitive: bool` (default `false`),
  `dry_run: bool` (default `false` — when `true`, returns a
  `ForgetPreview` without deleting), and
  `summary_strategy: "regen" | "drop"` (default `"regen"`
  rewrites partially-covered summaries from surviving turns;
  `"drop"` deletes them without calling the LLM). See
  `docs/gdpr.md` for the cascade semantics.
- **`GET /status`**: response `{"num_entries": int, "bundle_path":
  str, "embed_model": str}`.
- **`GET /health`**: response `{"ok": bool, "loaded_bundles": int}`.
- **`GET /version`**: response `{"version": str}`.

For the full field-by-field schema, use:

```bash
soma serve --port 8420 &
curl -s http://localhost:8420/openapi.json | python -m json.tool
```

Or read the committed snapshot at
[`clients/typescript/openapi.json`](../clients/typescript/openapi.json).

## Error model

Every 4xx / 5xx response body is a JSON object:

```json
{"detail": "<human-readable reason>"}
```

(Pydantic 422 validation errors carry a list under `detail` — same shape as FastAPI's default.)

Common branches:

| Status | Meaning |
| --- | --- |
| `400` | Bad bundle name / malformed query. |
| `401` | Missing or bad `Authorization` header (when auth is enabled). |
| `403` | Valid token but wrong permission for the route. |
| `404` | `node_id` or bundle not found. |
| `422` | Pydantic request-body validation error. |
| `429` | Rate limiter tripped (only when `SOMA_RATE_LIMIT_RPS` is set). Response includes `Retry-After`. |
| `501` | Criteria-based `/forget` with no `ConversationalMemory` wired (see `docs/gdpr.md`). |

Standard headers:

| Header | When |
| --- | --- |
| `X-SOMA-Deprecated: use JWT` | The request authenticated with `SOMA_API_KEY` (legacy). |
| `Retry-After` | 429 response. |

## Generated clients

- **TypeScript (`npm install soma-memory`):** see
  [`docs/clients.md`](clients.md). Types regenerate from
  `/openapi.json` on every PR.
- **Other languages:** point any OpenAPI generator at
  `GET /openapi.json` or the committed snapshot. Options at
  [openapi.tools](https://openapi.tools/).

## Configuration reference

See the module docstring at the top of
[`src/soma/serve.py`](../src/soma/serve.py) for the full env-var list.
Highlights:

| Variable | Purpose |
| --- | --- |
| `SOMA_BUNDLE_PATH` | Default-bundle directory. |
| `SOMA_BUNDLES_DIR` | Root for multi-tenant `/bundles/{name}/`. |
| `SOMA_EMBED_MODEL` | sentence-transformers model name. `stub` for CI. |
| `SOMA_JWT_SECRET` | HS256 shared secret. |
| `SOMA_JWT_PUBLIC_KEY_PATH` / `SOMA_JWT_PRIVATE_KEY_PATH` | RS256 keys. |
| `SOMA_JWT_ALG` | `HS256` (default) or `RS256`. |
| `SOMA_JWT_LEEWAY` | Clock-skew tolerance, seconds (default 60). |
| `SOMA_JWT_BLOCKLIST_PATH` / `SOMA_JWT_BLOCKLIST_REDIS_URL` | Revocation backend. See `docs/auth.md`. |
| `SOMA_API_KEY` | Legacy admin escape hatch (deprecated). |
| `SOMA_RATE_LIMIT_RPS` / `_BURST` / `_SCOPE` | In-process token-bucket rate limiter. |
| `SOMA_CORS_ORIGINS` | Browser CORS allow-list. Default `http://localhost:*`. |
| `SOMA_LOG_JSON` | `1` to emit one JSON line per retrieve. |
| `SOMA_OTEL_ENABLED` | `1` to enable OpenTelemetry spans (requires `[otel]` extra). |

---

For operator-facing Prometheus metrics + Grafana dashboards, see
[`observability.md`](observability.md). For the multi-tenant JWT auth
flow end-to-end, see [`auth.md`](auth.md).
