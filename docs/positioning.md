# SOMA — Local-First Agent Memory That Learns

> **Status:** Pre-1.0, API stabilising. See
> `docs/plans/2026-04-15-memory-layer-pivot.md` for the pivot that led
> here.

## What SOMA is

A drop-in agent memory layer that replaces `vector_store + RAG` with a
learning graph. Store text, retrieve by meaning, and let the structure
of the graph — which associations form, which prune, what the working
memory holds — reshape itself with use. Everything local, everything
on the user's disk, LLM-agnostic.

## What SOMA is not

- Not an LLM. SOMA holds memory; your LLM of choice does the talking.
- Not a vector DB clone. It has a vector-DB-shaped API (`store`,
  `retrieve`) but the implementation is a plastic graph with working
  memory, episodic store, consolidation, and structural growth.
- Not a cloud service. The entire "brain" is a directory. Back it up,
  move it to a new machine, share it across devices.

## Who it's for

- **Solo-dev / hobbyist agents** that want persistent user memory
  without spinning up Postgres + pgvector + a RAG pipeline.
- **Privacy-sensitive tools** where "user data goes to an external
  memory service" is a non-starter.
- **Multi-tenant chat apps** that need one shared bundle with isolated
  per-user scope — no extra infra, no per-user database.
- **Researchers** interested in plastic graph memory, structural
  sparsification, or complementary memory systems as an alternative
  to transformer-KV-plus-RAG.

## How it compares

| Capability                                        | Chroma + RAG | Mem0 / Zep | **SOMA** |
|---------------------------------------------------|:------------:|:----------:|:--------:|
| Local-first, zero deps                            | ✅           | ⚠️         | ✅       |
| Vector retrieval                                  | ✅           | ✅         | ✅       |
| Working-memory window                             | ❌           | ⚠️         | ✅       |
| Episodic-memory store                             | ❌           | ✅         | ✅       |
| Plastic graph substrate (in-place)                | ❌           | ❌         | ✅\*     |
| Consolidation hook (learning-ready)               | ❌           | ❌         | ✅       |
| Single-directory "brain" portability              | ❌           | ❌         | ✅       |
| Swap LLM without losing memory                    | ✅           | ⚠️         | ✅       |
| Survives crashes (WAL + atomic snapshot)          | ⚠️           | ⚠️         | ✅       |
| Conversational extract + reconcile                | ❌           | ✅         | ✅       |
| Multi-user scoping on a shared bundle             | ❌           | ⚠️         | ✅       |
| Per-bundle JWT auth + revocation blocklist        | ❌           | ⚠️         | ✅       |
| Prometheus `/metrics` + importable Grafana dashboards | ❌       | ❌         | ✅       |
| Pluggable vector backends (InProc / Qdrant / LanceDB) | ❌       | ❌         | ✅       |
| Learns from use                                   | ❌           | ⚠️         | ✅       |

⚠️ = partial / conditional on provider.
\* = plasticity substrate ships; current memory workload doesn't
trigger growth/pruning thresholds (see `paper-draft.md` §5 for the
research agenda to activate it).

## Where the wins are today

The differentiator today is **efficiency, operational posture, and the
substrate for learning to come**. On the benchmarks committed under
`benchmarks/reports/`:

- **Quality parity** with Chroma — identical Recall@3 / MRR@3 / NDCG@3
  on a labeled 50-fact / 26-query synthetic set with the same sbert
  embedder. By mathematical construction: same cosine over same vectors.
- **Disk** — 22.6× smaller at 50 facts, narrowing to 1.4× at 20K and
  1.42× at 100K. A bundle is `memory_embeddings.pt` plus a JSON index,
  not an HNSW + SQLite + metadata triple.
- **Store throughput** — 3.2–3.6× faster per op across the 1K–20K
  range. At 100K index-only, **0.4 s vs Chroma's 23.6 min** because
  SOMA's store is a tensor append while Chroma pays ~14 ms/op for
  SQLite + HNSW metadata (`scale_enterprise_100k.md`).
- **Retrieve** — HNSW backend runs 1.18–1.25× faster at 1K–20K,
  growing to **5.12× at 100K** while holding recall (`backend_matrix.md`).
- **Durability** — crash-safe WAL + atomic snapshot with three
  durability modes (`sync` / `batch` / `async`). A `Ctrl-C` mid-ingest
  loses nothing on `sync` mode.
- **Old memories don't rot** — a 30-day streaming-facts simulation
  holds old-fact Recall@3 at 0.883, essentially level with recent
  recall (0.938).
- **Beats Chroma + same reranker on two benchmarks** — the hybrid
  (BM25 + cosine) leg is SOMA's structural advantage; Chroma ships
  cosine alone, so users wanting lexical + semantic wire it themselves.

  * **LoCoMo** (5,882 turns, 1,982 queries, mxbai-embed-large,
    turn-level retrieval): SOMA hybrid+rerank reaches R@1=0.291,
    R@5=0.459, R@10=0.512 vs Chroma+same-rerank at R@1=0.259,
    R@5=0.405, R@10=0.471 — **+12% / +13% / +9% relative**. Latency
    24.6ms vs 8.8ms (2.8× slower; SOMA trades latency for recall).

  * **LongMemEval** (500 items, ~50 sessions each, sbert, long-horizon
    session retrieval): SOMA hybrid alone reaches R@1=0.892,
    R@5=**0.980**, R@10=0.992 vs Chroma+same-rerank at R@1=0.854,
    R@5=0.936, R@10=0.962 — **+4.4% / +4.7% / +3.1% relative** AT
    **8.3× LOWER latency** (34.4ms vs 286.7ms). Notably, the
    cross-encoder rerank HURTS on LongMemEval's long multi-topic
    sessions (adds noise that the reranker wasn't trained for);
    `mem.retrieve(query, k=5, hybrid_alpha=0.3)` with no reranker is
    the optimal config here.

  The lift is structural and corpus-general. The optional rerank is
  corpus-dependent: helps on atomic short docs (LoCoMo), hurts on
  long multi-topic docs (LongMemEval). See
  `research/developmental/results/recall_boost_mxbai_findings.md` and
  `research/developmental/results/longmemeval_retrieval_findings.md`.
- **+42% F1 end-to-end on LongMemEval QA** — same LLM
  (qwen3.5:4b-q8_0), same 3.8K-token context budget, only retrieval
  strategy varies. SOMA hybrid (α=0.3, no rerank) reaches F1=0.238 and
  R@5=0.990 vs Chroma cosine 0.168 / 0.850 and Chroma + same-reranker
  0.170 / 0.830 (N=100, first 100 items of LongMemEval small). SOMA
  strictly wins 27 items, ties 64, loses 9 vs Chroma cosine — i.e.
  the retrieval R@K advantage translates directly to **more correct
  answers from the same LLM**, not just higher retrieval numbers.
  For the same budget, `full_context` truncation (dump everything,
  drop oldest to fit) collapses to F1=0.029 / R@5=0.040 — retrieval
  crushes "just stuff it in" when the context budget can't hold the
  whole haystack. See
  `research/developmental/results/longmemeval_qa_compare_findings.md`.
- **Matches Chroma on recall, ~2.4× faster on retrieve** — on full
  LoCoMo (5,882 turns, 1,982 queries) at target_dim=128 with
  mxbai-embed-large, pure-cosine SOMA ties Chroma on R@5 (0.350 vs
  0.349) and edges it on R@1 (0.148 vs 0.147) while averaging
  **0.7ms per retrieve vs Chroma's 1.7ms** (same in-process HNSW
  setup; SOMA on CUDA, Chroma on CPU — which is how each would
  typically deploy). SOMA's graph-rerank layer is now **off by
  default** (`rerank_weight=0.0`) after a sweep (`w ∈ {0.0, 0.05,
  0.1, 0.2, 0.3}`) showed it was net-negative in the current
  formulation: the fingerprint signal adds noise to the ranked
  output. Turning rerank off + short-circuiting the graph forward
  pass cut SOMA retrieve from 17–19ms/query to 0.7–1.2ms. See
  `research/developmental/results/locomo_rerank_isolation_findings.md`.
  The rerank code is opt-in via `retrieve_hybrid(..., rerank_weight=w)`
  for non-retrieval use cases (temporal/prediction) or future
  reformulations.

The graph is plastic-by-construction (synaptogenesis, pruning,
myelination) but under the memory-only workload the growth knobs don't
fire — the substrate ships, the activation is an **open research
question** `benchmarks/reports/paper-draft.md` §5 scopes explicitly.
Today's efficiency and ops wins don't depend on it.

### What we've ruled out (honest)

- **LLM-distilled projections on retrieval**: Direction 4a (commit
  trail `e212798`..`767822b`) implemented teacher-distillation of
  SOMA's input projections using `mxbai-embed-large` as the teacher.
  On full LoCoMo, distilled variants underperformed both plain SOMA
  and Chroma (−0.008 to −0.015 on R@5). Root cause: distillation
  trained the input projections, but retrieval reads from the node
  fingerprint (dominated by Hebbian-trained node weights) — the
  signal dilutes across too many layers. Teacher infrastructure
  (`soma.llm.embedders`) ships as a clean dependency for future work,
  but the distillation-for-retrieval path is closed.

- **Spatial distillation on retrieval**: Direction 4b (commit trail
  `5be666a`..`1b8bf33`) extended Direction 4a with learnable node
  positions coupled to projection weights via a fixed random map, plus
  competitive top-K distillation. The stack is clean (Phase 1: 12 TDD
  tasks, 2551 tests green; Phase 2 multi-seed sanity PASSES stability
  and confirms positions reshape meaningfully, KS ≈ 0.20). On full
  LoCoMo it performs WORSE than both Direction 4a and plain SOMA
  (soma-spatial R@5 = 0.335 vs soma-distilled 0.344 vs soma-random
  0.337, chroma 0.349), and turning locality ON makes it worse still
  (−0.007). Root cause: coupling `p_i = normalize(P.T · W_i.flatten())`
  ties positions to projection weight matrices through a random
  compressor — positions end up encoding a rotation of existing
  projection information with no new semantic content. The teacher
  signal never reaches them through that channel. Infrastructure
  survives (position machinery, config flow-through, competitive top-K
  distill) for Option C (pairwise-distance distill) or Option D
  (spatial rerank) follow-ups, but this coupling formulation is
  closed. See `research/developmental/results/locomo_direction4b_findings.md`.

## Quick start

```python
from soma.memory import MemoryLayer

# Create with sentence-transformers (pip install soma[sbert]):
mem = MemoryLayer.with_sbert()

# Or load an existing brain:
# mem = MemoryLayer.load("my-brain/")

# Store
mem.store("user lives in Portland, OR", metadata={"source": "chat-2026-04-15"})
mem.store("user is vegetarian")

# Retrieve
hits = mem.retrieve("where does the user live?", k=3)
for hit in hits:
    print(hit.text, hit.score, hit.metadata)

# Graph queries (beyond what a vector DB can do)
neighbours = mem.related(hits[0].node_id, k=5)

# Let the graph adapt
mem.consolidate()   # triggers Hebbian learning + structural plasticity
mem.save("my-brain/")
```

For the end-to-end server + JWT + ConversationalMemory + Grafana flow,
see `docs/quickstart.md`.

## LangChain / LlamaIndex

```python
# LangChain:
from soma.integrations.langchain import SomaRetriever
retriever = SomaRetriever(memory=mem, k=5)
docs = retriever.invoke("what does the user do for work?")

# LlamaIndex:
from soma.integrations.llamaindex import SomaRetriever
nodes = SomaRetriever(memory=mem, k=5).retrieve("dietary restrictions")
```

## REST API

```bash
uvicorn soma.serve:app --port 8420
# or: docker compose up

curl -X POST http://localhost:8420/store \
  -H 'Content-Type: application/json' \
  -d '{"text": "user lives in Portland"}'
```

Per-bundle JWT auth, revocation blocklist, Prometheus metrics, and
importable Grafana dashboards (`deploy/grafana/`) are covered in the
quickstart and `docs/auth.md` / `docs/observability.md`.

## Licensing & commercial story

Pre-1.0, everything is MIT. Intent is to keep the core MIT post-1.0
and sell hosting / multi-device sync / optional enterprise features
rather than relicensing the core.

## Research appendix

For the architectural and empirical backstory — the developmental-AI
hypothesis, the hybrid-brain experiment, the three-regime ablation
that surfaced the projector-prior — see
`docs/whitepaper.md` and `docs/progress/HYBRID_PIVOT.md`.
