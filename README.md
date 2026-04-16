# SOMA

**Local-first agent-memory layer.**

A drop-in replacement for `vector-store + RAG` where the store is a plastic graph that grows and prunes with use. Store text, retrieve by meaning, reconcile conversational facts, and let the structure reshape itself over time. Everything local, everything on your disk, LLM-agnostic.

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

# Framework adapters:
pip install -e ".[langchain]"
pip install -e ".[llamaindex]"

# Everything:
pip install -e ".[sbert,ann,serve,metrics,otel,qdrant,lancedb,langchain,llamaindex]"
```

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
| Pluggable vector backends (adapter protocol)   | no     | no         | no       | **yes** (InProc + Qdrant + LanceDB) |

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

**Auth** (`pip install "soma[serve]"`): per-bundle JWTs with `read`/`write`/`admin` scopes, HS256 or RS256, rotation via `soma auth rotate-secret`, single-token revocation via a file-backed blocklist (`SOMA_JWT_BLOCKLIST_PATH`). Full reference: [`docs/auth.md`](docs/auth.md).

**Observability** (`pip install "soma[metrics]"`): `GET /metrics` exposes 18+ Prometheus counters/gauges/histograms covering every MemoryLayer hot path plus per-route HTTP timings. Three importable Grafana dashboards ship under [`deploy/grafana/`](deploy/grafana/) (RED overview, auth, USE bundle-health). Set `SOMA_LOG_JSON=1` for Loki/Datadog-ready structured logs. OpenTelemetry spans via `[otel]` + `SOMA_OTEL_ENABLED=1`. Metric reference: [`docs/observability.md`](docs/observability.md).

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

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/new/template?template=https://github.com/soma-ai/SOMA&envs=SOMA_API_KEY)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/soma-ai/SOMA)

Fly.io: `fly launch --from https://github.com/soma-ai/SOMA --copy-config`. Kubernetes (Helm 3.14+): `helm install soma oci://ghcr.io/soma-ai/charts/soma --version 0.1.0` — runbook in [`docs/deployment-k8s.md`](docs/deployment-k8s.md). Per-platform notes: [`docs/deployment-cloud.md`](docs/deployment-cloud.md). Minimum tier: 2 GB RAM.

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
- **[Cookbook](docs/cookbook.md)** — 18 recipes (hybrid, rerank, multi-tenant REST, ConversationalMemory, multi-user, migrations).
- **[Auth](docs/auth.md)** — per-bundle JWTs, RS256 split, revocation, rotation.
- **[Observability](docs/observability.md)** — Prometheus metrics, JSON logs, OTel, Grafana dashboards.
- **[Backends](docs/backends.md)** — InProc / Qdrant / LanceDB adapter tradeoffs.
- **[LLM backends](docs/llm-backends.md)** — Ollama / OpenAI / Anthropic / vLLM / HF.
- **[Recall improvements](docs/recall-improvements.md)** — hybrid BM25, rerank, query expansion research agenda.
- **[Clients](docs/clients.md)** — TypeScript client, auth modes, retry middleware.
- **[Demos](docs/demos.md)** — every shipped demo, when to run it.
- [Positioning](docs/positioning.md) · [Pivot + roadmap](docs/plans/2026-04-15-memory-layer-pivot.md) · [Whitepaper](docs/whitepaper.md) · [Paper draft](benchmarks/reports/paper-draft.md)

## License

MIT
