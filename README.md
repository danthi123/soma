# SOMA

**Local-first agent-memory layer.** A drop-in replacement for `vector-store + RAG` where the store is a plastic graph that grows and prunes with use. Store text, retrieve by meaning, reconcile conversational facts, and let the structure reshape itself over time. Everything local, everything on your disk, LLM-agnostic.

> **M1 — Hybrid retrieval validated** (2026-04-20): **+22.8 % F1** and **+15.6 % rank-1** over a Chroma-cosine baseline on **LongMemEval N=500**, same embedder, same LLM, matched context budgets. Triple-cross-validated (direct Token-F1 +22.8 %, qwen-as-judge +22.2 %, Claude-as-judge +23.7 %) and reproducible across six axes — cross-LLM, cross-embedder, cross-benchmark, cross-judge, α-sweep. Milestone doc: [`docs/milestones/2026-04-20-hybrid-retrieval-validated.md`](docs/milestones/2026-04-20-hybrid-retrieval-validated.md). Evidence: [`research/developmental/results/longmemeval_full_evidence_roundup.md`](research/developmental/results/longmemeval_full_evidence_roundup.md).

> **60-second tour:** install, store a fact, retrieve it — see the [Quick start](#quick-start) below or the full end-to-end flow in [`docs/quickstart.md`](docs/quickstart.md). Picking SOMA over Mem0/Letta/Zep/Chroma? [`docs/comparison.md`](docs/comparison.md). Patterns + recipes: [`docs/cookbook.md`](docs/cookbook.md). Positioning: [`docs/positioning.md`](docs/positioning.md).

## Install

```bash
# Minimal (torch + tokenizers only):
pip install -e .

# Quality retrieval (sentence-transformers):
pip install -e ".[sbert]"

# REST API server + JWT auth:
pip install -e ".[serve]"

# FAISS ANN (>10K entries):
pip install -e ".[ann]"

# Prometheus /metrics + OpenTelemetry tracing:
pip install -e ".[metrics]"
pip install -e ".[otel]"

# Alternative vector backends:
pip install -e ".[qdrant]"    # Qdrant (local file or HTTP)
pip install -e ".[lancedb]"   # embedded arrow-native (10M+ scale)
pip install -e ".[chroma]"    # drop-in for existing Chroma users
pip install -e ".[pgvector]"  # Postgres + pgvector

# Cloud object-store bundles:
pip install -e ".[s3]"        # s3:// URLs on save/load
pip install -e ".[gcs]"       # gs:// URLs on save/load

# Framework adapters:
pip install -e ".[langchain]"
pip install -e ".[llamaindex]"

# Everything runtime-useful:
pip install -e ".[sbert,ann,serve,metrics,otel,qdrant,lancedb,chroma,pgvector,s3,gcs,langchain,llamaindex]"
```

> Installing from PyPI? Replace `pip install -e "."` with `pip install "soma-memory"` (and the same for every `[extra]` variant above — the distribution name is `soma-memory`, the import name stays `soma`).

## Quick start

```python
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()                        # all-MiniLM-L6-v2
mem.store("user lives in Portland, OR", metadata={"user": "alex"})
mem.store("user is vegetarian",         metadata={"user": "alex"})
mem.store("user's dog is named Luna",   metadata={"user": "alex"})

hits = mem.retrieve("dietary restrictions", k=3, where={"user": "alex"})
mem.save("my-brain/")                                 # portable bundle
mem = MemoryLayer.load("my-brain/")                   # resume anywhere
```

For the end-to-end agent flow — `soma serve`, JWT issue + revoke, `ConversationalMemory` fact extraction, multi-user scoping, Grafana dashboard import — see [`docs/quickstart.md`](docs/quickstart.md).

## Runnable examples

Self-contained scripts under [`examples/`](examples/) that exercise the core API end-to-end:

- [`01_quickstart.py`](examples/01_quickstart.py) — the 10-line Python API tour (store, retrieve, save/load round-trip with metadata filters).
- [`02_persistent_chat_agent.py`](examples/02_persistent_chat_agent.py) — chat agent whose memory survives process restarts. Stub LLM inline; hook your own with ~5 lines.
- [`03_multi_tenant_bundle.py`](examples/03_multi_tenant_bundle.py) — one process, many isolated per-tenant bundles, with a cross-tenant-leak check.
- [`cloud_s3_demo.py`](examples/cloud_s3_demo.py) — round-trip a bundle through `s3://` object storage.

Run any of them with `python examples/<name>.py` after `pip install -e ".[sbert]"`.

## How it compares

| Capability                                     | Chroma | Mem0 / Zep | Pinecone | **SOMA** |
|------------------------------------------------|:------:|:----------:|:--------:|:--------:|
| Vector retrieval                               | yes    | yes        | yes      | yes      |
| Local-first, zero cloud deps                   | yes    | partial    | no       | yes      |
| Metadata `where` filter at retrieve            | yes    | yes        | yes      | yes      |
| Hybrid BM25 + vector (built-in)                | no     | partial    | partial  | **yes**  |
| Cross-encoder rerank (built-in)                | no     | no         | partial  | **yes**  |
| LLM query expansion (built-in)                 | no     | partial    | no       | **yes**  |
| Conversational extract + reconcile (built-in)  | no     | yes        | no       | **yes**  |
| Multi-user scoping on a shared bundle          | no     | partial    | no       | **yes**  |
| Plug-and-play LLM backends                     | no     | partial    | no       | **yes** (5 shipped) |
| Plastic graph substrate                        | no     | no         | no       | **yes**\* |
| Single-directory brain portability             | partial| no         | no       | **yes**  |
| Multi-tenant REST (`bundles/{name}`)           | no     | yes        | yes      | **yes**  |
| Per-bundle JWT auth + revocation blocklist     | no     | partial    | yes      | **yes**  |
| Crash-safe WAL + auto-compaction               | partial| yes        | yes      | **yes**  |
| Prometheus metrics + importable Grafana dashboards | no | no         | partial  | **yes**  |
| Pluggable vector backends (adapter protocol)   | no     | no         | no       | **yes** (InProc + Qdrant + LanceDB + Chroma + pgvector) |
| Bundles on S3 / GCS (scale-to-zero ready)      | no     | no         | no       | **yes** (`s3://` / `gs://` URLs) |
| GDPR-grade forgetting with audit trail         | no     | no         | no       | **yes** (`POST /forget` + `docs/gdpr.md`) |
| Typed schemas (31 built-in, extensible)        | no     | no         | no       | **yes** (8 domains, context packer) |

\* substrate ships; current memory workload doesn't trigger growth/pruning thresholds — see `benchmarks/reports/paper-draft.md` §5 for the research agenda to activate it.

Full comparison + migration notes: [`docs/comparison.md`](docs/comparison.md).

**Benchmark (same sbert embedder, measured vs Chroma, reports in `benchmarks/reports/`):**

- Quality parity: identical Recall@3 / MRR@3 / NDCG@3 at same embedder (by construction).
- Disk: 22.6× smaller at 50 facts, narrowing to 1.4× at 20K and 1.42× at 100K.
- Store (full pipeline 1K–20K): 3.2–3.6× faster per op; index-only 100K ingest takes **0.4 s vs Chroma's 23.6 min** because SOMA's store is a tensor append while Chroma pays ~14 ms/op for SQLite+HNSW metadata (`scale_enterprise_100k.md`).
- Retrieve HNSW backend: 1.18–1.25× faster at 1K–20K, growing to **5.12× at 100K** while preserving identical recall.
- Drift: 30-day simulation, old-fact Recall@3 = 0.883 ≈ recent 0.938 (memory doesn't rot).

**Recall boosters — SOMA goes beyond the same-embedder ceiling:**

Peer vector DBs all tie SOMA on recall when using the same embedder (identical cosine over identical vectors). To beat them, SOMA ships three opt-in boosters:

| Retrieval strategy             | R@1   | R@5   | Lift R@5 vs cosine |
|--------------------------------|------:|------:|-------------------:|
| Pure cosine (peer DB ceiling)  | 0.098 | 0.238 | —                  |
| Hybrid BM25+cosine             | 0.207 | 0.415 | +17.7 pp (+74%)    |
| Cross-encoder rerank           | 0.203 | 0.309 | +7.1 pp            |
| **Hybrid + rerank**            | 0.287 | 0.450 | **+21.2 pp (+89%)** |

Measured on LoCoMo (5,882 turns, 1,982 questions). Both knobs on triples R@1 and adds ~34 ms on top of baseline 13 ms. Full suite lives under `benchmarks/reports/` with the `paper-draft.md` aggregator wiring every number back to its script + report.

## REST API + Docker

```bash
# Local:
soma serve --port 8420

# Docker:
docker compose up
```

Endpoints: `/health`, `/version`, `/status`, `/store`, `/store_batch`, `/retrieve`, `/get/{id}`, `/related/{id}`, `/recent`, `/forget`, `/consolidate`, `/save`, plus `/bundles/{name}/...` multi-tenant variants under per-bundle JWT auth.

**Auth** (`pip install "soma-memory[serve]"`): per-bundle JWTs with `read`/`write`/`admin` scopes, HS256 or RS256, rotation via `soma auth rotate-secret`, single-token revocation via a file-backed blocklist (`SOMA_JWT_BLOCKLIST_PATH`). Full reference: [`docs/auth.md`](docs/auth.md).

**Observability** (`pip install "soma-memory[metrics]"`): `GET /metrics` exposes 18+ Prometheus counters/gauges/histograms covering every MemoryLayer hot path plus per-route HTTP timings. Three importable Grafana dashboards ship under [`deploy/grafana/`](deploy/grafana/) (RED overview, auth, USE bundle-health). Set `SOMA_LOG_JSON=1` for Loki/Datadog-ready structured logs. OpenTelemetry spans via `[otel]` + `SOMA_OTEL_ENABLED=1`. Metric reference: [`docs/observability.md`](docs/observability.md).

### TypeScript client

```bash
npm install @soma-ai/client
```

```ts
import { createClient } from "@soma-ai/client";
const soma = createClient({ baseUrl: "http://localhost:8420", token: process.env.SOMA_TOKEN });
await soma.POST("/store", { body: { text: "Paris is the capital of France." } });
const { data } = await soma.POST("/retrieve", { body: { query: "capital?", k: 3 } });
```

Works in Node 18+, browsers, Deno, Bun, Cloudflare Workers. Types regenerate from the live `/openapi.json` on every PR — see [`docs/clients.md`](docs/clients.md).

## CLI

```bash
soma index   --wiki path/to/docs --bundle my-brain/   # ingest folder
soma chat    --bundle my-brain/                       # auto-picks LLM backend
soma stats   --bundle my-brain/                       # entry count, disk
soma search  --bundle my-brain/ --query "..."         # vector search, no LLM
soma serve   --port 8420                              # REST API
soma bundle  list ./data/bundles                      # lifecycle: list | info | delete
soma auth    issue --sub alex --bundle alex:read,write --expires 30d
soma auth    revoke --token $LEAKED --reason "leaked on slack"
```

`soma chat` auto-detects a backend: Ollama if running, OpenAI/Anthropic if `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` is set, otherwise local HuggingFace. Override with `--backend`. See [`docs/llm-backends.md`](docs/llm-backends.md).

## Cloud deploy

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/new/template?template=https://github.com/danthi123/soma&envs=SOMA_API_KEY)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/danthi123/soma)

Fly.io: `fly launch --from https://github.com/danthi123/soma --copy-config`. Kubernetes (Helm 3.14+): a Helm chart ships under `deploy/helm/` — `helm install soma deploy/helm/soma` — runbook in [`docs/deployment-k8s.md`](docs/deployment-k8s.md). Per-platform notes: [`docs/deployment-cloud.md`](docs/deployment-cloud.md). Minimum tier: 2 GB RAM.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
mypy src/soma/
```

## Docs

- **[Quickstart](docs/quickstart.md)** — end-to-end agent-memory flow (install → serve → JWT → ConversationalMemory → Grafana).
- **[Comparison](docs/comparison.md)** — SOMA vs Chroma / Mem0 / Letta / Zep / Pinecone.
- **[Cookbook](docs/cookbook.md)** — recipes for hybrid retrieval, rerank, multi-tenant REST, ConversationalMemory (sync/async/batch), multi-user, migrations, streaming chat, cloud bundles, typed schemas, context packing.
- **[Typed Schemas](docs/schemas.md)** — define, store, retrieve, extend, and pack typed memory entries (31 built-in schemas across 8 domains).
- **[Auth](docs/auth.md)** — per-bundle JWTs, RS256 split, revocation, rotation.
- **[Observability](docs/observability.md)** — Prometheus metrics, JSON logs, OTel, Grafana dashboards.
- **[Backends](docs/backends.md)** — InProc / Qdrant / LanceDB / Chroma / pgvector adapter tradeoffs.
- **[Cloud](docs/cloud.md)** — S3/GCS bundle URLs + Lambda / Cloud Run / Fly deploy recipes.
- **[GDPR forgetting](docs/gdpr.md)** — `POST /forget`, audit trail, summary cascade, compliance posture.
- **[LLM backends](docs/llm-backends.md)** — Ollama / OpenAI / Anthropic / vLLM / HF.
- **[Recall improvements](docs/recall-improvements.md)** — hybrid BM25, rerank, query expansion research agenda.
- **[Clients](docs/clients.md)** — TypeScript client, auth modes, retry middleware.
- **[Demos](docs/demos.md)** — every shipped demo, when to run it.
- [Positioning](docs/positioning.md) · [Pivot + roadmap](docs/plans/2026-04-15-memory-layer-pivot.md) · [Whitepaper](docs/whitepaper.md) · [Paper draft](benchmarks/reports/paper-draft.md)

## License

MIT
