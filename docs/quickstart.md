# SOMA Quickstart — End-to-End Agent Memory

A walkthrough of the full stack every SOMA user eventually lives on: REST
server, per-bundle JWT auth, conversational-memory fact extraction,
multi-user scoping, token revocation, Grafana dashboards. Runnable top to
bottom in ~10 minutes on any laptop.

If you only want the 10-line Python API tour, [the README](../README.md)
has it. This doc is for the first agent you actually deploy.

## 1. Install with the extras you'll need

```bash
pip install "soma-memory[sbert,serve,metrics]"
```

> **Developing on SOMA?** From a clone, use the editable variant:
> `pip install -e ".[sbert,serve,metrics]"`.

- `sbert` — `sentence-transformers` for quality retrieval.
- `serve` — `fastapi`, `uvicorn`, `pyjwt[crypto]` for the REST server.
- `metrics` — `prometheus-fastapi-instrumentator` + `prometheus-client`
  so the `/metrics` endpoint lights up.

Add `[ann]` for FAISS once you cross ~10K entries, `[otel]` for
OpenTelemetry spans, `[qdrant]` or `[lancedb]` for alternate backends.
See [`backends.md`](backends.md) for the when-to-pick-which.

## 2. Start the server

```bash
export SOMA_BUNDLES_DIR=./data/bundles
soma serve --port 8420
```

On an unconfigured host this runs in **open mode** — every route is
reachable without a bearer token. Fine for a local dev loop; not fine for
anything anyone else can hit. The next step closes that.

## 3. Mint a per-bundle JWT

HS256 is the one-host default; for split issuer/verifier deploys see the
RS256 recipe in [`auth.md`](auth.md).

```bash
# Generate a shared secret once; store it like any credential.
export SOMA_JWT_SECRET=$(soma auth rotate-secret)

# Issue a token scoped to bundle "alice" with read+write for 30 days.
export ALICE_TOKEN=$(
  soma auth issue --sub alice --bundle alice:read,write --expires 30d
)

# Restart the server so it picks up SOMA_JWT_SECRET from env.
soma serve --port 8420
```

The server now accepts the JWT on `/bundles/alice/*` and returns 401 on
anything unsigned. `/health`, `/version`, and `/metrics` stay public —
they have to, so liveness probes and Prometheus scrapers keep working.

## 4. Store via REST

```bash
curl -X POST http://localhost:8420/bundles/alice/store \
  -H "Authorization: Bearer $ALICE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text": "Alice lives in Portland, OR", "metadata": {"source": "intake"}}'
```

The response carries `{"node_id": "<uuid>"}`. That id is also what
`/forget`, `/get/{id}`, and `/related/{id}` take.

Each `/bundles/{name}` path maps to `$SOMA_BUNDLES_DIR/{name}/` on disk;
bundles load lazily and cache in memory. A single `soma serve` process
can host many bundles — see [`cookbook.md`](cookbook.md) §15 for the
full multi-tenant pattern.

## 5. ConversationalMemory — fact extraction + reconcile

Raw `/store` is fine for pre-chunked documents. For chat turns, wrap the
bundle in `ConversationalMemory` so the LLM extracts atomic facts,
de-duplicates them, and supersedes older claims when they conflict. This
is the Mem0/Zep-style pipeline — see [`cookbook.md`](cookbook.md) §18
for the design notes.

```python
from soma.llm    import backend_from_env
from soma.memory import ConversationalMemory, MemoryLayer

mem = MemoryLayer.load("./data/bundles/alice")   # the bundle you just wrote to
llm = backend_from_env()                         # Ollama > OpenAI > Anthropic > HF
cm  = ConversationalMemory(memory=mem, llm=llm, session_id="chat-1")

cm.add_message("user", "I just moved to Boston from Portland")
# -> facts extracted (location), reconciled against existing entries.
#    The old Portland fact gets metadata.superseded_by pointing at the
#    new Boston fact; both stay in the bundle for audit.
```

If your chat LLM is a small (3B-ish) local model that struggles to emit
strict JSON, pin a stronger one for the structured steps without
upgrading the whole chat path:

```python
from soma.llm import OllamaBackend

chat_llm = OllamaBackend(model="llama3.2:3b")         # 3B chat
extract  = OllamaBackend(model="qwen2.5:7b-instruct") # 7B JSON emitter
cm = ConversationalMemory(memory=mem, llm=chat_llm, extractor_llm=extract)
```

## 6. Multi-user scoping on a shared bundle

One bundle, many end-users. Pass `user_id=` and every write stamps
`metadata.user_id`; retrieval, clear, and supersede auto-scope to that
user without a new endpoint.

```python
cm_alice = ConversationalMemory(memory=mem, llm=llm, session_id="s1", user_id="alice")
cm_bob   = ConversationalMemory(memory=mem, llm=llm, session_id="s1", user_id="bob")

cm_alice.add_message("user", "my dog is Rex")
cm_bob.add_message("user",   "my cat is Mia")

cm_alice.retrieve("pets")       # -> only Rex
cm_bob.retrieve("pets")         # -> only Mia
cm_alice.retrieve("pets", user_id=None)  # admin drill-down sees both
```

`supersede()` on a cross-user fact raises `PermissionError`. Backward
compat: leave `user_id` unset and metadata stays byte-identical to
pre-Phase-12 bundles.

Over REST, the same pattern works without new routes — pass `user_id` in
the request body's `metadata` field and filter with the same key in
`where`:

```bash
curl -X POST http://localhost:8420/bundles/alice/store \
  -H "Authorization: Bearer $ALICE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text": "Alice likes pho", "metadata": {"user_id": "alice"}}'
```

## 7. Retrieve with a metadata filter

```bash
curl -X POST http://localhost:8420/bundles/alice/retrieve \
  -H "Authorization: Bearer $ALICE_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "query": "where does alice live?",
        "k": 3,
        "where": {"user_id": "alice"}
      }'
```

Supported `where` operators: exact match, `$eq`, `$ne`, `$in`, `$nin`,
`$gt`, `$gte`, `$lt`, `$lte`. Multiple fields AND together. Matches
Chroma's semantics for the subset people actually use.

For the hybrid BM25 + cosine blend and cross-encoder rerank knobs, see
[`cookbook.md`](cookbook.md) §12–14. They layer on top of the same
endpoint via `hybrid_alpha` and `rerank_top_n` body params.

## 8. Check bundle state

```bash
soma bundle list ./data/bundles         # one row per bundle, WAL size + entry count
soma bundle info ./data/bundles/alice   # detail view: disk bytes, last-modified, backend
```

Use for health checks, disk attribution, post-incident forensics. The
`list` command marks corrupt bundles with a footnote pointing at the
broken file. Deleting is `soma bundle delete <path>` with a y/N prompt
unless `--yes`.

## 9. Revoke a leaked token

You don't need to rotate `SOMA_JWT_SECRET` to kill one token — point the
server at a blocklist file and every worker picks up revocations on the
next mtime poll (~30 s).

```bash
export SOMA_JWT_BLOCKLIST_PATH=./data/jwt-blocklist.jsonl
# restart soma serve so it starts reading the blocklist
```

Revoke the leaked token:

```bash
soma auth revoke --token "$LEAKED_TOKEN" --reason "leaked on slack 2026-04-16"
```

The next request that token makes returns HTTP 401 with
`{"detail": "token revoked"}` and
`soma_auth_failures_total{reason="revoked_token"}` advances. Garbage-
collect past-exp entries on a daily cron:

```bash
soma auth gc
```

Full lifecycle + RS256 split-host setup: [`auth.md`](auth.md).

## 10. Import the Grafana dashboards

SOMA ships three importable dashboards under `deploy/grafana/`:

- `soma-overview.json` — RED (Rate / Errors / Duration) across the REST
  surface.
- `soma-auth.json` — `soma_auth_failures_total` by reason, revoked-token
  hits, success rate.
- `soma-bundle-health.json` — USE (Utilization / Saturation / Errors):
  WAL throughput, consolidation p95, live-entries gauge, peer-reload
  rate, retrieve-latency heatmap.

Import via Grafana UI (Dashboards → New → Import → upload JSON), pick
the Prometheus datasource for the `${DS_PROMETHEUS}` placeholder, done.
For `grafana-cli admin`, docker-compose provisioning, or a Kubernetes
`ConfigMap` pattern — plus the smoke-test procedure — see
[`deploy/grafana/README.md`](../deploy/grafana/README.md).

The dashboards expect metrics emitted by `pip install "soma-memory[metrics]"`.
Every PromQL query maps 1:1 to a metric documented in
[`observability.md`](observability.md).

---

## Where to go next

- **More recipes** — [`cookbook.md`](cookbook.md) has 26, including
  hybrid retrieval, cross-encoder rerank, streaming JSONL ingest,
  Chroma migration, cloud bundles (S3/GCS), typed schemas, and
  context packing for LLM prompts.
- **Different LLM** — [`llm-backends.md`](llm-backends.md) covers
  Ollama, OpenAI, Anthropic, vLLM / LM Studio, local HuggingFace.
- **Alternative storage** — [`backends.md`](backends.md) for the
  InProc / Qdrant / LanceDB tradeoffs.
- **Why SOMA over X?** — [`comparison.md`](comparison.md) walks the
  Chroma / Mem0 / Zep / Letta / Pinecone decision matrix.
- **Benchmarks** — [`../benchmarks/reports/paper-draft.md`](../benchmarks/reports/paper-draft.md)
  wires every published number back to its committed script + report.
