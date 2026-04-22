# SOMA Cookbook — Recipes for Common Patterns

Drop-in code snippets for the patterns most agent-memory callers
need. All recipes use only the public API; copy them and adapt.

## 1. Ingest a wiki and chat

```python
from pathlib import Path
from soma.llm    import OllamaBackend, RAGSession
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()
paras: list[str] = []
metas: list[dict] = []
for md in Path("wiki/").rglob("*.md"):
    for para in md.read_text(encoding="utf-8").split("\n\n"):
        if para.strip():
            paras.append(para.strip())
            metas.append({"path": str(md)})
mem.store_batch(paras, metadatas=metas)  # one embed call per batch, not per para
mem.save("brain/")

chat = RAGSession(memory=mem, llm=OllamaBackend(model="llama3.2"))
print(chat.ask("how do we deploy?").text)
```

(For real use, prefer `soma index --wiki ... --bundle ...` from the
CLI — it does heading-aware chunking with overlap.)

## 2. Persistent chat session

```python
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()  # or .load_with_sbert("brain/") to resume

# Each turn, store both sides so future turns can recall them.
def turn(user_msg: str, llm) -> str:
    hits = mem.retrieve(user_msg, k=5)
    ctx  = "\n".join(f"- {h.text}" for h in hits)
    reply = llm.generate(
        f"Context from prior turns:\n{ctx}\n\nUser: {user_msg}\nAssistant:"
    )
    mem.store(user_msg, metadata={"role": "user"})
    mem.store(reply,    metadata={"role": "assistant"})
    return reply

mem.save("brain/")  # at end of session
```

## 3. Per-user memory (multi-tenancy by metadata)

```python
mem = MemoryLayer.with_sbert()  # one shared store

mem.store("Alex prefers vegetarian recipes",  metadata={"user_id": "alex"})
mem.store("Bobbi prefers seafood",            metadata={"user_id": "bobbi"})

# In-index pre-filter (same semantics as Chroma's `where`):
alex_hits = mem.retrieve("food preferences?", k=5, where={"user_id": "alex"})

# Multi-field = AND; value-list via $in; numeric comparisons:
hits = mem.retrieve(
    "q", k=5,
    where={"user_id": {"$in": ["alex", "bobbi"]}, "priority": {"$gte": 3}},
)
```

If you'd rather isolate physically, use one bundle per user:

```python
alex_mem  = MemoryLayer.load("brains/alex/")
bobbi_mem = MemoryLayer.load("brains/bobbi/")
```

## 4. Soft-forget by metadata (GDPR / right-to-be-forgotten)

```python
to_forget = [
    h.node_id for h in mem.get_recent(len(mem))  # full scan
    if h.metadata.get("user_id") == "alex"
]
for node_id in to_forget:
    mem.forget(node_id)
mem.save("brain/")
```

## 5. Importance-weighted store (mark some entries critical)

```python
# Pre-store, weight by source confidence (only metadata; retrieval
# still ranks by cosine — use this on the read side):
mem.store(text, metadata={"source": "policy-doc", "priority": "high"})
mem.store(other_text, metadata={"source": "user-chat", "priority": "low"})

# At retrieve time, re-rank by prior:
def boosted(query: str, k: int = 5):
    hits = mem.retrieve(query, k=k * 3)  # over-fetch
    boost = {"high": 1.2, "low": 0.9}
    for h in hits:
        h_score = h.score * boost.get(h.metadata.get("priority", ""), 1.0)
        h.metadata["_boosted_score"] = h_score
    return sorted(
        hits, key=lambda h: h.metadata["_boosted_score"], reverse=True
    )[:k]
```

## 6. Hybrid retrieval — vector + recency

```python
def vector_plus_recent(query: str, k: int = 5, recency_weight: float = 0.3):
    semantic = mem.retrieve(query, k=k * 2)
    recent   = mem.get_recent(k * 2)
    seen, out = set(), []
    for h in semantic:
        out.append(h)
        seen.add(h.node_id)
        if h.score < recency_weight:
            break
    for h in recent:
        if h.node_id not in seen:
            out.append(h)
        if len(out) >= k:
            break
    return out[:k]
```

## 7. Streaming ingest from JSONL

```python
import json
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()
with open("export.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        mem.store(rec["text"], metadata=rec.get("metadata", {}))
        if mem._step % 1000 == 0:  # snapshot every 1K
            mem.save("brain/")
mem.save("brain/")
```

## 8. Use any OpenAI-compatible local server (vLLM / LM Studio / llama.cpp)

```python
from soma.llm import OpenAICompatibleBackend, RAGSession

backend = OpenAICompatibleBackend(
    model="qwen2.5-7b-instruct",
    base_url="http://localhost:8000/v1",  # vLLM
)
chat = RAGSession(memory=mem, llm=backend)
```

## 9. Custom prompt template

```python
from soma.llm import RAGSession

PROMPT = """\
You are a code-aware assistant. The user's repo has been indexed
into memory. Use only the snippets below; cite [n] inline. If the
snippets don't cover the question, say so and suggest what to grep.

Snippets:
{context}

Question: {question}

Reply (cite like [1]):"""

chat = RAGSession(memory=mem, llm=backend, prompt_template=PROMPT)
```

## 10. Migrate from Chroma

```bash
python scripts/migrate_chroma.py \
  --chroma-dir path/to/chroma-store/ \
  --soma-dir   path/to/new-soma-bundle/
```

The script reads docs + metadata + embeddings from Chroma's
PersistentClient, writes them into a fresh MemoryLayer bundle. Same
embedder = same retrieval results, just a different store.

## 11. Run as a background REST API

```bash
soma serve --port 8420
# then
curl -X POST http://localhost:8420/store \
  -H 'Content-Type: application/json' \
  -d '{"text": "user lives in Portland", "metadata": {"src":"chat"}}'
curl -X POST http://localhost:8420/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "where does the user live?", "k": 3}'
```

## 12. Hybrid lexical + vector retrieval

When queries hinge on specific terminology (proper names, domain
jargon, numeric IDs) that sbert's sub-word tokenizer smears into a
broader semantic neighbourhood, add BM25 alongside cosine:

```python
# alpha=0 → pure BM25; alpha=1 → pure cosine; 0.3–0.5 is typical.
hits = mem.retrieve("reading list for Kubernetes RBAC", k=5, hybrid_alpha=0.3)
```

BM25 index is built lazily and invalidated when new entries are
stored. No extra deps.

## 13. LLM query expansion

Let the LLM rewrite each question into N variants + sub-questions,
retrieve per variant, merge with Reciprocal Rank Fusion. Cheap +3-8
pp Recall@5 on under-specified queries:

```python
from soma.llm import QueryExpander, RAGSession, backend_from_env

backend = backend_from_env()
chat = RAGSession(
    memory=mem, llm=backend,
    query_expander=QueryExpander(llm=backend, n_variants=3),
)
chat.ask("what was that thing I mentioned about dinner?")
# Expands -> retrieves per variant -> RRF-merges -> final answer
```

## 14. Cross-encoder re-ranking

Over-fetch cosine candidates, re-rank them with a small
cross-encoder (~5–10 ms/candidate on CPU). Usually +5–15% Recall@5
on real queries:

```python
from soma.memory.rerank import CrossEncoderReranker

mem.attach_reranker(CrossEncoderReranker())  # lazy model load
hits = mem.retrieve("...", k=5, rerank_top_n=20)

# Combine with hybrid for best of both:
hits = mem.retrieve("...", k=5, hybrid_alpha=0.3, rerank_top_n=20)
```

The pre-rerank score is kept in `h.metadata["_pre_rerank_score"]` so
you can compare.

## 15. Multi-tenant REST server + JWT auth

One server, many brains. Tenant routes under `/bundles/{name}`. Each
caller gets a JWT scoped to the bundles they can read or write. See
`docs/auth.md` for the full reference.

```bash
# 1. Generate a shared HS256 secret and mint per-caller tokens:
export SOMA_JWT_SECRET=$(soma auth rotate-secret)
export ALEX_TOKEN=$(
  soma auth issue --sub alex --bundle alex:read,write --expires 30d
)
export BOBBI_TOKEN=$(
  soma auth issue --sub bobbi --bundle bobbi:read,write --expires 30d
)

# 2. Start server — it picks up SOMA_JWT_SECRET from env:
SOMA_BUNDLES_DIR=./data/bundles soma serve --port 8420

# 3. Each caller hits their own bundle:
curl -X POST http://localhost:8420/bundles/alex/store \
  -H "Authorization: Bearer $ALEX_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text": "alex prefers vegetarian"}'

curl -X POST http://localhost:8420/bundles/bobbi/store \
  -H "Authorization: Bearer $BOBBI_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text": "bobbi prefers seafood"}'

# ALEX_TOKEN on bobbi's bundle → 403 (bundle mismatch).
# Health check (always no-auth):
curl http://localhost:8420/health
```

Each `/bundles/{name}` path maps to `$SOMA_BUNDLES_DIR/{name}/` on
disk; bundles are loaded lazily and cached in memory.

**Legacy** `SOMA_API_KEY` is still supported as a deprecated admin
escape hatch — responses carry `X-SOMA-Deprecated: use JWT`.

## 16. Inspect a bundle without loading the LLM

```bash
soma stats  --bundle brain/
soma search --bundle brain/ --query "deployment notes" --k 5
python scripts/demo_memory_inspect.py recent --bundle brain/ --n 50
python scripts/demo_memory_inspect.py dump --bundle brain/ > snapshot.jsonl
```

## 17. Durability — crash-safe persistence (WAL)

```python
from soma.memory import MemoryLayer

# durability="sync" (default): fsync after every store — zero loss on
# kernel panic, ~1 ms/op overhead. Good for "I pressed Ctrl-C" safety.
mem = MemoryLayer.with_sbert()
# persist later via mem.save("brain/") — for an auto-persisting layer,
# use MemoryLayer(embed_fn=..., embed_dim=..., bundle_path="brain/") instead.

# "batch": fsync every 32 ops — 10-20x throughput at the cost of losing
# up to the last batch on crash. Good for bulk ingest.
mem = MemoryLayer(
    embed_fn=my_embed, embed_dim=384,
    bundle_path="brain/", durability="batch",
)

# "async": never fsync in hot path; caller flushes on demand. Fastest,
# loses any in-flight records on an unclean shutdown.
mem = MemoryLayer(
    embed_fn=my_embed, embed_dim=384,
    bundle_path="brain/", durability="async",
)
mem.store("fact")
mem.flush()  # force-sync before planned shutdown
```

A WAL + snapshot layout lets the bundle reload after a crash WITHOUT
a prior `save()` call — every `store()` / `forget()` is recoverable.
Multi-worker uvicorn on one bundle is safe; each worker catches the
others' WAL tail before each retrieve.

## 18. Conversational memory — Mem0/Zep-style extraction + reconcile

`ConversationalMemory` wraps a `MemoryLayer` with LLM-driven fact
extraction and reconciliation, borrowing Mem0's two-phase pipeline
(arXiv 2504.19413; +26% LoCoMo QA accuracy over raw RAG at 91% lower
latency) and Zep's "invalidate, don't delete" SUPERSEDE semantics.

```python
from soma.llm    import backend_from_env
from soma.memory import ConversationalMemory, MemoryLayer

mem = MemoryLayer.with_sbert()   # or .load_with_sbert("brain/")
llm = backend_from_env()         # picks Ollama / OpenAI / Anthropic / HF
cm  = ConversationalMemory(memory=mem, llm=llm, session_id="alex")

cm.add_message("user", "I just moved to Boston from Portland")
# -> atomic facts extracted (location) + reconciled against existing
#    facts. If "User lives in Portland" was already stored, the LLM
#    decides SUPERSEDE: the Portland fact gets metadata.superseded_by
#    pointing at the new Boston fact; old fact stays for audit.

for hit in cm.retrieve("where does the user live?"):
    print(hit.text)   # "User lives in Boston"   (Portland filtered by default)

# Inspect history / roll back
cm.retrieve("where does the user live?", include_superseded=True)  # both

# Session lifecycle
cm.get_summary()              # most-recent rolled summary (every 20 turns)
cm.list_facts()               # live facts only
cm.supersede(old_id, "new text")  # explicit invalidation
cm.clear_session()            # wipes turns+facts, keeps summaries
```

Design notes:

- **Threshold short-circuit.** Extracted facts with top-candidate
  cosine ≥ 0.92 are treated as duplicates (no LLM call). Facts with
  top-cosine < 0.75 are ADDed unconditionally (no LLM call either).
  Only the 0.75–0.92 ambiguous band pays for an LLM round-trip
  returning `ADD` / `UPDATE` / `SUPERSEDE` / `NOOP`. Thresholds are
  kwargs — calibrate per embedder.
- **Raw turns still land.** Every call to `add_message` stores the
  verbatim turn with `metadata.type="turn"` so LoCoMo-style eval
  pipelines that expect raw turns keep working unchanged.
- **Rolling summaries.** Every `summary_every` turns (default 20) the
  wrapper asks the LLM for a 3–5-sentence recap and stores it with
  `metadata.type="summary"`.
- **Re-summarization (anti-drift).** Chained summaries compound
  hallucinations over long sessions: each new summary is built from
  the previous summary plus recent turns, so mistakes stick. Every
  `resummarize_every` summaries (default 5) the wrapper instead
  re-derives a fresh summary from the last
  `resummarize_every × summary_every` raw turns only, bypassing the
  previous summary. Tune lower (e.g. 3) for noisy extractors where
  drift accumulates fast, higher (e.g. 10) when LLM calls are
  expensive and turns are short. Set `resummarize_every=0` to
  disable re-summarization and keep the pre-Phase-17 chained-only
  behaviour. Re-summary entries are marked with
  `metadata.resummary=True` so the audit trail distinguishes chained
  from re-derived summaries.
- **SUPERSEDE ≠ delete.** Old entries stay in the bundle with
  `metadata.superseded_by = new_id`. `retrieve()` filters them out by
  default; pass `include_superseded=True` to see the audit trail.

### 18.1 Async extraction for low-latency chat

For interactive chat paths where the UI is waiting on the turn to
persist but doesn't need extracted facts synchronously available,
pass `extraction_mode="async"`. `add_message()` returns as soon as the
raw turn is stored; extract + reconcile run on a single background
thread (`ThreadPoolExecutor(max_workers=1)`, so within-session
extraction order is preserved). Call `flush()` — or use the
context-manager protocol — before reading extracted facts.

```python
with ConversationalMemory(
    memory=mem, llm=llm, session_id="alex",
    extraction_mode="async",
) as cm:
    cm.add_message("user", "I moved to Boston")   # returns immediately
    cm.add_message("user", "I work at Acme Corp") # returns immediately
    # ...respond to user while extraction runs in the background...
    cm.flush()                                    # facts now queryable
    hits = cm.retrieve("where does the user live?")
# __exit__ drains pending futures + shuts down the executor.
```

`close()` is idempotent; `clear_session()` flushes first so in-flight
facts land and are then wiped rather than leaking in after the reset.
Exceptions raised on the executor thread surface on the next `flush()`
call (they are not silently swallowed). Python's GIL means the async
win is I/O overlap with the LLM network call — local CPU-bound
backends will not see a speedup. Prefer `"sync"` for batch ingest
where strict per-turn ordering matters and latency is a non-concern.

### 18.2 Multi-user scoping on a shared bundle

One bundle, many users (e.g. a multi-tenant chat app). Pass `user_id=`
to `ConversationalMemory` and every stored turn / fact / summary is
tagged with `metadata.user_id`; `retrieve()`, `clear_session()`, and
`supersede()` auto-scope to that user.

```python
cm_alice = ConversationalMemory(
    memory=mem, llm=llm, session_id="chat-1", user_id="alice",
)
cm_bob   = ConversationalMemory(
    memory=mem, llm=llm, session_id="chat-1", user_id="bob",
)

cm_alice.add_message("user", "my dog is Rex")  # stored as alice
cm_bob.add_message("user",   "my cat is Mia")  # stored as bob

# retrieve() scopes to the constructor's user_id by default
cm_alice.retrieve("pets")   # -> only Alice's Rex entry
cm_bob.retrieve("pets")     # -> only Bob's Mia entry

# Per-call override; user_id=None is the admin drill-down
cm_alice.add_message("user", "typing on behalf of Bob", user_id="bob")
admin_hits = cm_alice.retrieve("pets", user_id=None)  # sees both

# supersede() refuses cross-user mutation on a shared bundle
cm_alice.supersede(bob_fact_id, "…")  # raises PermissionError
```

Non-breaking: leaving `user_id` unset (single-tenant deploys,
pre-Phase-12 callers) preserves the old behaviour and writes no
`user_id` key into metadata.

**REST pattern.** The REST surface (`POST /store`, `/retrieve`, …)
already accepts arbitrary `metadata`, so no new endpoints are needed.
Clients pass `{"metadata": {"user_id": "alice"}}` in request bodies
and filter with the same key in `where` on retrieval. A JWT claim
layer that auto-routes the `user_id` from the token is Phase 13+
territory; the core plumbing ships here.

## 19. Scale past 20K with an embedded LanceDB backend

Qdrant-local warns past ~20K entries because its embedded index wasn't
designed for that scale. The `LanceDBBackend` adapter fills the "local-
first, no server, 10M+ scale" slot between `InProc` and a Qdrant HTTP
deployment. Install the extra and swap the backend at construction
time:

```bash
pip install -e ".[lancedb,sbert]"
```

```python
from sentence_transformers import SentenceTransformer
import torch

from soma.memory import MemoryLayer
from soma.memory.backends.lancedb import LanceDBBackend

model = SentenceTransformer("all-MiniLM-L6-v2")
def embed(text: str) -> torch.Tensor:
    return torch.tensor(model.encode(text, convert_to_numpy=True))

backend = LanceDBBackend(path="./data/lance-alice", dim=384, index_type="hnsw")
mem = MemoryLayer(
    embed_fn=embed,
    embed_dim=384,
    backend=backend,
    bundle_path="brains/alice",
)

mem.store("alice prefers vegetarian", metadata={"user_id": "alice"})
# Filter pushdown — LanceDB evaluates the where clause in the engine
# instead of Python. Supports $eq / $ne / $gt / $gte / $lt / $lte / $in / $nin.
hits = mem.retrieve("food preferences?", k=5, where={"user_id": "alice"})
```

Bundle layout: the LanceDB table directory sits inside the bundle, so
`mem.save()` / `MemoryLayer.load()` still works unchanged — you're not
managing a separate datastore. See [`backends.md`](backends.md) for the
InProc / Qdrant / LanceDB decision matrix and the filter-pushdown
fallback semantics.

Benchmarked numbers (`benchmarks/reports/backend_matrix.md`, N=100K):

| Backend      | Store total | Retrieve p50 (ms) | Retrieve p95 (ms) | Recall@10 |
|--------------|------------:|------------------:|------------------:|----------:|
| InProcFlat   | 6.7 s       | 7.71              | 12.91             | 1.000     |
| InProcHNSW   | 6.6 s       | 0.45              | 0.80              | 0.714     |
| LanceDBFlat  | 16.6 s      | 39.90             | 44.26             | 1.000     |
| LanceDBHNSW  | 17.9 s      | 6.54              | 9.33              | 0.676     |

Pick `InProcHNSW` for raw latency when the dataset fits in RAM;
`LanceDBFlat` when you want exact recall plus filter pushdown and
on-disk storage past RAM size.

## 20. Operate with Prometheus + Grafana

`pip install -e ".[metrics]"` exposes 18+ counters / gauges /
histograms on `GET /metrics` — every MemoryLayer hot path plus FastAPI
per-route timings. Schema and labels are stable across minor versions.

```bash
pip install -e ".[metrics]"
soma serve --port 8420                 # exposes /metrics
curl -s http://localhost:8420/metrics | grep '^soma_'
```

Three importable dashboards ship under `deploy/grafana/`:

- `soma-overview.json` — RED (Rate / Errors / Duration) across the REST surface.
- `soma-auth.json` — `soma_auth_failures_total{reason}` + revoked-token hits + success rate.
- `soma-bundle-health.json` — USE (Utilization / Saturation / Errors):
  WAL throughput, consolidation p95, live-entries gauge, peer-reload
  rate, retrieve-latency heatmap.

Import via the Grafana UI, `grafana-cli admin`, docker-compose
provisioning, or a Kubernetes `ConfigMap` — the step-by-step recipes
(plus the smoke-test procedure) live in
[`../deploy/grafana/README.md`](../deploy/grafana/README.md). Full
metric table and PromQL examples: [`observability.md`](observability.md).

Multi-tenant deploys with thousands of bundles can collapse the
`bundle` label to a single series per metric via
`SOMA_METRICS_BUNDLE_LABEL_DISABLE=1`; the metric names stay the same
so dashboards don't break.

Switch stdout to structured JSON for Loki / Datadog / CloudWatch
ingestion:

```bash
export SOMA_LOG_JSON=1
uvicorn soma.serve:app --port 8420
```

Every `retrieve()` call emits one JSON line tagged `event=retrieve`
with `bundle`, `backend`, `latency_ms`, `n_hits`, `hybrid_alpha`,
`rerank_top_n`, and `cache_miss`. Schema is stable —
`tests/test_memory/test_retrieve_log_line.py` pins it.

## 21. Stream tokens live in `soma chat`

`soma chat` streams the LLM reply token-by-token automatically whenever
the resolved backend supports it — `OpenAIBackend`, `OllamaBackend`,
`AnthropicBackend`, and any `OpenAICompatibleBackend` (LM Studio, vLLM,
LiteLLM, llama.cpp's built-in server) all ship streaming adapters out
of the box. Backends without it (dry-run, local HuggingFace) fall back
to the existing blocking `generate()` path with no visible change.

Detection is a single `hasattr(backend, "stream_generate")` check at
REPL start, so adding streaming to a custom backend is a one-method
add:

```python
from collections.abc import Iterator

class MyBackend:
    name = "my-llm"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        return self._client.complete(prompt, max_tokens=max_tokens)

    # Optional — implement only if your transport supports streaming.
    def stream_generate(
        self, prompt: str, *, max_tokens: int = 256
    ) -> Iterator[str]:
        for chunk in self._client.stream(prompt, max_tokens=max_tokens):
            if chunk.text:
                yield chunk.text
```

Notes:

- The REPL accumulates the full reply from the stream and stores it as
  one assistant turn — token-by-token memory extraction is a non-goal.
- Ctrl-C during a stream drops the partial turn and returns you to the
  prompt with the terminal left in a usable state.
- The `/chat` REST endpoint stays one-shot for now; SSE / WebSocket
  streaming there is a separate phase.

---

## 22. Bundles on S3 or GCS (scale-to-zero deploys)

Point `MemoryLayer.save` / `load` at a URL and the bundle lives in
object storage instead of local disk. Unlocks Cloud Run / AWS Lambda
/ Fly Machines / any platform where local disk is ephemeral.

```python
from soma.memory import MemoryLayer

mem = MemoryLayer(...)
mem.add(["hello"], embedding)

mem.save("s3://my-bucket/soma/bundle")   # S3 (boto3 + AWS creds)
mem.save("gs://my-bucket/soma/bundle")   # GCS (ADC chain)
mem.save("file:///abs/local/bundle")     # explicit local URL
mem.save("/abs/local/bundle")            # plain path still works

restored = MemoryLayer.load("s3://my-bucket/soma/bundle")
```

S3 endpoint override (MinIO, Cloudflare R2, DigitalOcean Spaces):

```python
from soma.storage import S3ObjectStore
store = S3ObjectStore(
    bucket="b", prefix="p",
    endpoint_url="https://<account>.r2.cloudflarestorage.com",
)
mem.save(store)
```

Install `pip install "soma-memory[s3]"` or `pip install "soma-memory[gcs]"`.
Full deploy recipes (Dockerfile, IAM, env vars) in [`docs/cloud.md`](cloud.md).

## 23. GDPR-grade forgetting with audit trail

Beyond `clear_session`: cascade-delete raw turns + derived facts +
summaries (regenerated from surviving turns when possible, else dropped).

```python
from soma.memory.conversational import ConversationalMemory
from soma.forget_audit import ForgetAuditSink

audit = ForgetAuditSink.from_env()   # reads SOMA_FORGET_AUDIT_PATH
cm = ConversationalMemory(memory=mem, llm=llm, audit_sink=audit)

# Preview first — always a safe dry-run
preview = cm.forget(text_matches="gardening", dry_run=True)
print(f"Would delete {preview.total_vectors} vectors "
      f"({len(preview.raw_turns)} turns, "
      f"{len(preview.derived_facts)} facts, "
      f"{len(preview.summaries)} summaries)")

# Commit the delete
result = cm.forget(text_matches="gardening")
print(f"Deleted {result.total_deleted}, regenerated "
      f"{len(result.regenerated_summaries)} summaries")

# Conservative: always drop partial-coverage summaries (no LLM call)
result = cm.forget(text_matches="gardening", summary_strategy="drop")
```

Over REST: `POST /forget` with `{"text_matches": "gardening",
"dry_run": true}` — requires `write` scope. Full docs including the
compliance posture at [`docs/gdpr.md`](gdpr.md).

## 24. Agent workflow with typed schemas

Use built-in schemas to store structured agent state alongside
free-text memory. `store_typed` validates fields and embeds
searchable text automatically; `retrieve_typed` reconstructs typed
instances with filter safety.

```python
from soma.memory import MemoryLayer
from soma.schemas.builtin.agent import Decision, Observation, TaskState

mem = MemoryLayer.with_sbert()

# Track task lifecycle
task = TaskState(
    task_id="deploy-v2",
    status="active",
    step=1,
    plan_summary="migrate DB, deploy backend, run smoke tests",
)
mem.store_typed(task)

# Record observations from tool calls
mem.store_typed(Observation(
    task_id="deploy-v2",
    source="tool",
    content="migration completed in 12s, 3 tables altered",
))

# Record a decision
mem.store_typed(Decision(
    task_id="deploy-v2",
    choice="blue-green deploy",
    rationale="zero-downtime requirement from SLA",
    alternatives="rolling, canary",
))

# Retrieve all active tasks
active = mem.retrieve_typed(
    TaskState,
    query="deploy",
    k=10,
    status="active",
)

# Retrieve decisions for a specific task
decisions = mem.retrieve_typed(
    Decision,
    query="deploy strategy",
    k=5,
    task_id="deploy-v2",
)
```

Mix typed and untyped entries freely -- `retrieve()` still works on
everything in the store.

## 25. Custom schema extension

Define domain-specific schemas in your package. Registration happens
at import time -- no config, no plugin system.

```python
# myagent/schemas.py
from soma.schemas import schema, field

@schema("devops.deploy")
class Deploy:
    """Track a deployment with rollback info."""
    service: str = field(filterable=True, searchable=True)
    version: str = field(filterable=True)
    environment: str = field(
        filterable=True,
        choices=["dev", "staging", "prod"],
    )
    status: str = field(
        filterable=True,
        choices=["pending", "rolling", "live", "rolled_back"],
        default="pending",
    )
    rollback_to: str | None = field(default=None)
    notes: str = field(searchable=True, default="")

    class Meta:
        context_priority = 0.9  # high priority in context packing
        ttl_seconds = 86400 * 7  # expire after 1 week
```

```python
# myagent/main.py
import myagent.schemas  # auto-registers Deploy

from soma.memory import MemoryLayer
from soma.schemas import get_schema, list_schemas

mem = MemoryLayer.with_sbert()

# Verify registration
assert "devops.deploy" in list_schemas()
assert get_schema("devops.deploy") is myagent.schemas.Deploy

# Use it
from myagent.schemas import Deploy

mem.store_typed(Deploy(
    service="api-gateway",
    version="2.4.1",
    environment="prod",
    notes="includes fix for auth timeout",
))

deploys = mem.retrieve_typed(
    Deploy,
    query="auth fix",
    k=5,
    environment="prod",
    status="live",
)
```

For third-party packages: put `import .schemas` in your package's
`__init__.py` so callers just `import mypackage` and schemas
register.

## 26. Context packing for LLM prompts

`pack_context` assembles a token-budgeted context string from memory,
mixing recent entries, semantically relevant hits, active tasks,
decisions, and preferences.

```python
from soma.memory import MemoryLayer
from soma.schemas.packing import pack_context

mem = MemoryLayer.with_sbert()
# ... store some typed and untyped entries ...

# Default mix (50% relevant, 15% recent, 10% tasks, 10% decisions, 15% prefs)
context = pack_context(mem, query="what should I deploy next?")

# Custom mix -- shift budget toward decisions and code incidents
context = pack_context(
    mem,
    query="what went wrong last week?",
    max_tokens=2000,
    mix={
        "relevant": 0.4,
        "recency": 0.1,
        "decisions": 0.2,
        "code.incident": 0.3,  # any schema type name works as a slot
    },
)

# Restrict to specific domains
context = pack_context(
    mem,
    query="customer complaints",
    types=["customer.*", "collab.*"],
)

# Use in an LLM prompt
prompt = f"""\
You are a helpful assistant. Use the context below to answer.

Context:
{context}

Question: What should I deploy next?
"""
```

Each entry is formatted as `[type_name] text`, one per line. The
packer deduplicates across slots so no entry appears twice.

---

Missing a recipe you want? Open an issue with the use case — most
agent-memory patterns are 5–20 lines of glue around `store` /
`retrieve` / `related` / `get_recent` / `forget`.
