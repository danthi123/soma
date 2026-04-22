# SOMA documentation

> This directory mixes user-facing docs, operator-facing docs, and a
> lot of internal research and plan notes. Use this index to tell
> them apart. If you're evaluating or integrating SOMA, the "Start
> here" and "Install / run" sections are what you want; the rest is
> navigable but secondary.

## 1. Start here

- [**Positioning**](positioning.md) — what SOMA is, what it competes
  against, and what we don't claim.
- [**Comparison**](comparison.md) — SOMA vs Chroma / Mem0 / Letta /
  Zep / Pinecone, decision matrix and migration notes.

## 2. Install / run

- [**Quickstart**](quickstart.md) — end-to-end agent flow (install →
  `soma serve` → JWT → `ConversationalMemory` → Grafana dashboards).
- [**Cookbook**](cookbook.md) — 26 copy-paste recipes (hybrid
  retrieval, rerank, multi-tenant REST, conversational memory,
  cloud bundles on S3/GCS, typed schemas, context packing, GDPR
  forgetting, etc.).
- [**LLM backends**](llm-backends.md) — Ollama / OpenAI / Anthropic /
  vLLM / LM Studio / local HuggingFace.
- [**Recall improvements**](recall-improvements.md) — hybrid BM25,
  cross-encoder rerank, query expansion, measured lift on LoCoMo
  and LongMemEval.

## 3. Reference

- [**REST API**](rest-api.md) — every route, auth requirement, and
  request/response shape.
- [**Typed schemas**](schemas.md) — define, store, retrieve, extend;
  31 built-in schemas across 8 domains plus the context packer.
- [**Backends**](backends.md) — InProc / Qdrant / LanceDB / Chroma /
  pgvector adapter tradeoffs, filter pushdown, bundle layout.
- [**Auth**](auth.md) — per-bundle JWTs, HS256 / RS256, revocation
  blocklist (file or Redis), rate limiting, `SOMA_API_KEY`
  deprecation.
- [**Observability**](observability.md) — Prometheus metrics,
  structured JSON logs, OpenTelemetry spans, importable Grafana
  dashboards.
- [**Clients**](clients.md) — TypeScript client (`npm install
  soma-memory`); auth modes; retry middleware.
- [**GDPR forgetting**](gdpr.md) — `POST /forget`, audit trail,
  summary cascade, compliance posture.

## 4. Deploy / operate

- [**Cloud (S3 / GCS bundles)**](cloud.md) — `s3://` and `gs://`
  bundle URLs plus AWS Lambda / Cloud Run / Fly Machines +
  Cloudflare R2 recipes.
- [**Deployment — cloud PaaS**](deployment-cloud.md) — Railway /
  Render / Fly / Kubernetes / DigitalOcean recipes.
- [**Deployment — Kubernetes (Helm)**](deployment-k8s.md) — Helm
  chart runbook, secret management, Prometheus Operator
  integration.
- [**Demos**](demos.md) — every shipped demo and when to run it.

## 5. Deep technical / background

- [**Whitepaper**](whitepaper.md) — the original
  brain-inspired-developmental-AI design. Now the research
  substrate beneath the memory-layer product; reader-orientation
  note at the top points to `positioning.md` for the current
  framing.
- [**Paper draft**](../benchmarks/reports/paper-draft.md) — every
  published number wired back to its script + committed report.

## 6. Milestones, plans, research

- [**Milestones**](milestones/) — pinned states. Current:
  [M1 — Hybrid Retrieval Validated (2026-04-20)](milestones/2026-04-20-hybrid-retrieval-validated.md).
- [**Plans**](plans/) — phase plans (phase-1 … phase-45), research
  direction plans (Direction 4a / 4b, Path A / Path B), topic
  designs, and periodic audits. Historical; treat as an archive of
  how the project reached its current state.
- [**Progress**](progress/) — pivot-history notes
  (`HYBRID_PIVOT.md`).
- [**Paper**](paper/) — in-progress research-paper sources.

Sections 5 and 6 are optional for users of the shipping product —
the product is covered by sections 1–4. The research substrate lives
in `src/soma/core/`, `growth/`, `metacognition/`, `consolidation/`,
`io/`, `deploy/` (see `CONTRIBUTING.md` "Scope" for the
product-vs-substrate split).
