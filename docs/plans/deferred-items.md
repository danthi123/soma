# Deferred items — post-gap-closing backlog

Things that came up during the 2026-04-16 gap-closing push (Phases 1-7b) but were explicitly scoped out or flagged for later. Organized by category with source pointers, rough effort, and reasoning.

> **Triage:** the remaining next-sprint set is LoCoMo-QA eval, threshold calibration, and LanceDB. Everything below those is tier-2+.

---

## Completed (post-2026-04-16)

### JWT revocation blocklist — done 2026-04-16
File-backed JSONL blocklist shipped. New module `src/soma/auth_revocation.py`
with `BlocklistBackend` Protocol + `FileBlocklist` impl. Wired through
`verify_token` (optional `blocklist=` kwarg) and `serve.py` module-level
`_blocklist = blocklist_from_env()`. New CLI subcommands: `soma auth revoke`
(accepts `--token` or `--jti/--exp`), `soma auth list-revoked`, `soma auth
gc`. New metric label `soma_auth_failures_total{reason="revoked_token"}`.
Design decision memo: `docs/plans/2026-04-16-jwt-revocation.md`. Redis-backed
variant stayed deferred (see tier-2 auth below).

---

## Recommended next-sprint set

### LoCoMo-QA eval with LLM-as-judge
- **Why:** Quantifies our Mem0/Zep positioning. Mem0 claims +26% QA accuracy on LoCoMo — we haven't measured SOMA there.
- **Shape:** extend `benchmarks/run_locomo.py` with `--run-qa-eval` flag. Retrieves top-k, asks LLM to answer, another LLM judges. Compare raw vs ConversationalMemory arms.
- **Source:** `docs/plans/2026-04-16-phase-2-conversational-memory.md` §8.
- **Effort:** ~4 h (adapter wiring, judge prompt, report column).

### Threshold calibration (ConversationalMemory)
- **Why:** `near_dup_threshold=0.92` / `ambiguous_threshold=0.75` are sbert rules-of-thumb — never tuned on our actual corpus. Poor defaults hurt Phase 2's real-world recall/precision.
- **Shape:** sweep both thresholds over `{0.65, 0.70, 0.75, 0.80}` × `{0.88, 0.90, 0.92, 0.94}` on a LoCoMo subset; plot fact-count vs LoCoMo QA accuracy.
- **Source:** `docs/plans/2026-04-16-phase-2-conversational-memory.md` Risks §3.
- **Effort:** ~2 h as a bench script.

### LanceDB adapter (completes local-first scale story)
- **Why:** Qdrant-local caps at ~20K. Qdrant-HTTP requires a server. LanceDB is truly embedded, arrow-based, scales to 10M+ in-proc. Fills the "no server, but need scale" niche.
- **Shape:** new `src/soma/memory/backends/lancedb.py` conforming to `VectorBackend` Protocol. New optional extra `soma[lancedb]`.
- **Source:** `docs/plans/2026-04-16-phase-6-vector-backend.md` §4.
- **Effort:** ~1.5 d (adapter + tests + matrix benchmark row).

---

## Tier 2 — conversational memory

- **Async extraction mode** (`extraction_mode="async"` with `ThreadPoolExecutor`, `flush()` drains). Phase 2 Stage 2.5. ~1 day. Source: `docs/plans/2026-04-16-phase-2-conversational-memory.md`.
- **Batch extraction mode** (accumulate K turns, one LLM call). Phase 2 Stage 3. ~1 day.
- **`extractor_llm=` kwarg** — let users pin a stronger model for extraction while chat uses a smaller one. Critical for 3B-local users. ~2 h.
- **Summary re-summarization from raw turns** every M × `summary_every` to prevent compounding drift. Phase 2 Risks §4. ~4 h.
- **GDPR-grade forgetting** — scrub derived facts + summaries that reference a piece of info. Goes well beyond `clear_session`. ~1 week.
- **Multi-user scoping** — `user_id` claim alongside `session_id`. Non-breaking metadata add. ~4 h.

## Tier 2 — observability

- **`soma_compaction_total` / `soma_compaction_seconds` metrics.** Phase 3 explicitly left these as follow-up; Phase 1 Task 5 compaction runs uninstrumented. Source: `docs/plans/2026-04-16-phase-3-observability.md`. ~1 h.
- **`SOMA_METRICS_PUBLIC=0` gate** — put `/metrics` behind the bearer scheme for sensitive deploys. Phase 3 Risks §2. ~2 h.
- **`SOMA_METRICS_BUNDLE_LABEL_DISABLE=1`** — drop `bundle` label for deploys with 10k+ bundles to control cardinality. Phase 3 Risks §1. ~1 h.
- **Histogram bucket tuning** after first real Grafana dashboard. Phase 3 open question.
- **`soma_reload_total`** already shipped but unused by dashboards — add a sample Grafana panel.

## Tier 2 — auth

- **Refresh-token endpoint.** Punted as OAuth-flow complexity. Source: Phase 4 plan §6. ~1 d if we build it.
- **Per-token rate limiting.** Punted to reverse proxy. A lightweight in-proc limiter would be ~1 d.
- **Hashed-token store.** Defence-in-depth for the secret at rest. Stores the jti as `sha256(jti)` so the blocklist file is safe to exfiltrate. Only matters once blocklist size grows enough to leak signal. ~3 h. Source: `docs/plans/2026-04-16-jwt-revocation.md`.
- **Redis-backed revocation blocklist.** Optional extra `soma[redis-revocation]`. Instant propagation across workers + automatic TTL from `exp`. For multi-host / k8s deploys where the file-backed store's 30 s poll lag is too slow. ~1 d. Source: `docs/plans/2026-04-16-jwt-revocation.md`.
- **Audience claim (`aud`)** — useful for multi-server fleets. Not needed for single-tenant.

## Tier 2 — backends (Phase 6 follow-ups)

- **Milvus adapter.** Heavier dep surface (pymilvus + server). Overlaps Qdrant's niche. ~2 d.
- **Weaviate adapter.** Light client, heavy server. ~2 d.
- **pgvector adapter.** High demand; filter pushdown over JSONB is its own design pass. ~3 d.
- **Chroma-as-backend.** Migration story: let Chroma users get SOMA features on top of their existing Chroma dbs. ~2 d, mostly adapter + tests.
- **`backend.search_near_id(node_id, k)`** — optional faster path for `related()` over HTTP backends. Avoids the get-vectors round-trip. ~4 h.
- **Async Qdrant client (`AsyncQdrantClient`)** — waits for FastAPI routes to go async. No ETA.
- **Per-bundle vs shared Qdrant collection.** Per-bundle is v1; shared collection with bundle_name tag scales to 1000+ bundles. Decision deferred to demand.
- **Snapshot version-compat tests** across Qdrant versions — today the adapter writes `qdrant_version` in `backend.json` but we don't test cross-version restore. ~1 d.

## Tier 2 — TypeScript client

- **npm scope claim** (`@soma-ai`) — operator-only manual step before first publish. Fallback scopes documented: `@soma-memory`, `@soma-ml`.
- **`soma-memory-client` unscoped alias** pointing at the scoped package via `deprecate` — captures npm-search hits. ~15 min.
- **Retry middleware as optional re-exports.** Not forks of generated code — simple fetch-level wrapper. ~2 h.
- **React Query integration** via `openapi-react-query` plugin. ~4 h, only if React users ask.

## Tier 2 — k8s / cloud

- **HPA templates** in the Helm chart. Locked off until Phase 6's external-backend story matures (single-writer WAL blocks multi-replica). Revisit when Qdrant HTTP backend is the documented scale path.
- **`SomaCluster` CRD operator.** Attractive once multi-tenant sharding matures. Phase 8+ territory. Big effort (~3 weeks for a minimal operator).
- **S3/GCS bundle backend** — unlocks Cloud Run / App Runner (scale-to-zero platforms) by moving bundle state off local disk. ~1 week.
- **`soma cloud deploy` CLI** — one-command managed deploy via Fly/Render API. ~3 d. Only meaningful if we host a real managed service.
- **Helm 4 migration** — stay on 3.x for v1; migrate when the ecosystem settles. No ETA.
- **Topology spread vs anti-affinity production hardening** — stubs in `values.yaml`; no tested defaults.

## Tier 2 — docs / ops polish

- **`soma chat` async streaming.** REPL today blocks on the full LLM response. Streaming tokens would feel better.
- **Prometheus sample dashboards** checked in under `deploy/grafana/`. Users currently have to build from the metrics list in `docs/observability.md`.
- **`soma bundle` subcommand group** — `bundle list`, `bundle delete`, `bundle info`, `bundle migrate` for bundle lifecycle. Today users manage via `mem.save/load`. ~1 d.
- **Benchmark scheduler / cron** — re-run the matrix on PRs affecting `src/soma/memory/` to catch perf regressions. ~1 d.

## Pre-existing (carried forward)

- **Task #173 — Lazy stable-capture.** Defer stable-capture re-run to first retrieve instead of consolidate end. Saves startup time on large stores. Spec at `docs/plans/2026-04-16-lazy-stable-capture.md`.
- **Task #174 — Enterprise scale benchmark (100K + 1M).** 1M still running at time of writing.
- **Pre-existing mypy errors** in `src/soma/memory/api.py` (~5 errors, all from the `has_encoder` pattern). Harmless but eventually worth fixing for `mypy --strict` cleanliness.

## Research agenda (out-of-sprint)

- **Graph re-rank re-activation.** The plastic-graph substrate ships but the growth thresholds (training-tuned) don't fire under memory-layer workloads. The research agenda in `benchmarks/reports/paper-draft.md` §5 details what would activate it.
- **Multimodal memory.** Current store is text-only. Arrow-native adapters (LanceDB) could host image/audio vectors behind the same API. No concrete plan.
- **Federated / multi-device sync.** Bundles are portable; sync across devices is a different problem (CRDT on the WAL? sync service?). No plan.

---

## How to triage this doc going forward

1. When a deferred item becomes a real user ask → promote to a dated plan under `docs/plans/`.
2. When a deferred item becomes impossible or irrelevant → delete with a one-line git commit message.
3. Review at least every quarter; items staler than two quarters with no ask are probably not happening — delete them.
