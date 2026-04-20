# SOMA — Local-First Agent Memory with Hybrid Retrieval

> **Status:** Pre-1.0, API stabilising. See
> `docs/plans/2026-04-15-memory-layer-pivot.md` for the pivot to an
> agent-memory-layer product.

## What SOMA is

A local-first agent memory layer with **hybrid retrieval (BM25 +
cosine) that beats vector-DB + RAG** on QA benchmarks under matched
budgets. Store text, retrieve by meaning *and* keywords, and ship an
entire "brain" as a single portable directory.

Measured wins over a Chroma-cosine baseline (same LLM, same context
budget, same items):

- **+22.8% F1 / +15.6% rank-1** on LongMemEval N=500 with qwen3.5:4b
- **+22.0% F1** replicates under qwen3.5:9b (model-size invariant)
- **+14.7% F1** replicates under Claude Sonnet as answerer
- **+23.7%** judge-accuracy under a strong independent Claude judge
- **+88.7% rank-1 / +45.4% hit@5** on LoCoMo N=1974 (full dataset)
- **+45% rank-1** lift widens under a weaker embedder (mxbai) — the
  lift is mechanical, not embedder-specific

SOMA also contains a **plastic graph substrate** (synaptogenesis,
consolidation, working/episodic memory) as a research vehicle. The
substrate ships, tests are green, and it's genuinely plastic — but
three serious attempts at routing a learning-graph signal into
retrieval have all been negative (plastic-graph activation,
LLM-distilled projections, spatial distillation). The graph is not
currently contributing to the product's retrieval wins and we don't
claim it is. See "What we've ruled out" below.

## What SOMA is not

- **Not an LLM.** SOMA holds memory; your LLM of choice does the
  talking.
- **Not a vector DB clone.** Same `store` / `retrieve` API shape, but
  the retrieval is hybrid (BM25 + cosine) out of the box, with
  working memory, episodic store, consolidation, single-directory
  portability, crash-safe WAL, and per-bundle JWT auth.
- **Not a cloud service.** The entire brain is a directory. Back it
  up, move it to a new machine, share it across devices.
- **Not a "learning graph memory" product today.** The plastic graph
  substrate is a research vehicle that has not yet produced a
  positive retrieval result across three architectural attempts.
  Current wins come from the hybrid retrieval + operational posture.

## Who it's for

- **Teams running Chroma / Pinecone / Weaviate + RAG today** who want
  a drop-in retrieval upgrade: the hybrid layer nets +22% F1 on
  LongMemEval and +89% rank-1 on LoCoMo against the same embedder.
- **Solo-dev / hobbyist agents** that want persistent user memory
  without spinning up Postgres + pgvector + a RAG pipeline.
- **Privacy-sensitive tools** where "user data goes to an external
  memory service" is a non-starter.
- **Multi-tenant chat apps** that need one shared bundle with
  isolated per-user scope — no extra infra, no per-user database.
- **Researchers** using the plastic graph substrate as a reference
  implementation of brain-inspired structural plasticity, including
  its current retrieval-channel limitations.

## How it compares

| Capability                                        | Chroma + RAG | Mem0 / Zep | **SOMA** |
|---------------------------------------------------|:------------:|:----------:|:--------:|
| Local-first, zero deps                            | ✅           | ⚠️         | ✅       |
| Vector retrieval                                  | ✅           | ✅         | ✅       |
| **Hybrid retrieval (BM25 + cosine) built in**     | ❌           | ⚠️         | ✅       |
| Working-memory window                             | ❌           | ⚠️         | ✅       |
| Episodic-memory store                             | ❌           | ✅         | ✅       |
| Single-directory "brain" portability              | ❌           | ❌         | ✅       |
| Swap LLM without losing memory                    | ✅           | ⚠️         | ✅       |
| Survives crashes (WAL + atomic snapshot)          | ⚠️           | ⚠️         | ✅       |
| Conversational extract + reconcile                | ❌           | ✅         | ✅       |
| Multi-user scoping on a shared bundle             | ❌           | ⚠️         | ✅       |
| Per-bundle JWT auth + revocation blocklist        | ❌           | ⚠️         | ✅       |
| Prometheus `/metrics` + importable Grafana dashboards | ❌       | ❌         | ✅       |
| Pluggable vector backends (InProc / Qdrant / LanceDB) | ❌       | ❌         | ✅       |
| Plastic graph substrate (research-only)           | ❌           | ❌         | ✅\*     |

⚠️ = partial / conditional on provider.
\* = plasticity substrate ships and is genuinely plastic, but has
not yet been shown to improve retrieval; kept for research use, not
positioned as a product benefit. See "What we've ruled out" below.

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
- **End-to-end QA on LongMemEval: +23% F1 overall, +59% F1 on
  single-session-user, +36% F1 on multi-session** — same LLM
  (qwen3.5:4b-q8_0), same 3.8K-token context budget, strict-answer
  prompt (1-5 words), only retrieval strategy varies. Full 500-item
  run. SOMA hybrid (α=0.3, no rerank) reaches overall **F1=0.368 /
  EM=0.242 / R@5=0.980** vs Chroma cosine F1=0.299 / EM=0.196 /
  R@5=0.932. Per-type lifts: single-session-user **+59%** (retrieval
  is binding), multi-session **+36%**, temporal-reasoning **+23%**,
  knowledge-update **+11%**. Roughly a quarter of SOMA's answers
  are exact matches to the gold string.

  (The earlier verbose-prompt numbers we reported — +10% overall F1,
  +49% on single-session-user — turned out to be biased by token-F1's
  penalty on preamble ("Based on the conversation history..."). With
  strict "answer in 1-5 words" prompting, F1 recovers and SOMA's
  advantage widens on the harder question types because the LLM can
  actually use the better retrieved evidence instead of spending
  output budget on preamble. See
  `research/developmental/results/longmemeval_qa_compare_qwen9b_findings.md`
  for the verbosity-penalty investigation.)

  Chroma+cross-encoder rerank is tied with chroma cosine on this
  corpus (F1=0.170 vs 0.168 on N=100 verbose) — the rerank-hurts-on-
  long-sessions retrieval finding shows up end-to-end too. At matched
  3.8K budget, `full_context` truncation collapses to F1=0.029 —
  retrieval crushes "just stuff it in" when budget is constrained.
  See `research/developmental/results/longmemeval_qa_compare_findings.md`
  and `longmemeval_qa_compare_strict_findings.md`.

  **What drives the +23% lift**: partition-based decomposition of the
  paired N=500 strict run shows two independent mechanisms:
  **34% from recall** (30 items where only SOMA retrieves gold —
  BM25 catches keyword queries cosine misses) and **66% from ranking**
  (460 items where BOTH retrieve gold in top-5, yet SOMA wins F1 on
  54 to chroma's 24 — SOMA's hybrid scoring pushes the gold session to
  rank 1-2 where the LLM can extract from it, while chroma leaves it
  at rank 4-5 buried behind less-useful semantic matches and the LLM
  says "I don't know"). The ranking mechanism is stable across verbose
  and strict prompting. Single-session-user is a **clean sweep**:
  SOMA wins 22 items, chroma wins 0, 48 tied on 70 items of that type.
  Direct truncation evidence: chroma's IDK rate climbs monotonically
  with gold rank on the full 500-item rank probe (rank 1 → 17%, rank
  3 → 100%). See
  `research/developmental/results/longmemeval_causation_findings.md`.

  **Direct rank measurement** (`longmemeval_rank_delta_findings.md`):
  on the full paired 500-item probe, SOMA's hybrid places the gold
  session at rank 1 on **96.3% of its hits** vs chroma's **83.3%**;
  mean rank drops from 1.32 → 1.16. Pair-level: SOMA rescues **58 of
  78** items (74%) where chroma had gold at rank 2-5, promoting them
  to rank 1-2. Net items where SOMA has gold strictly earlier in
  context = +71 (14.2% of corpus). This is the mechanical lift
  that produces the +22% end-to-end F1.

  **Cross-embedder: the lift is not SBERT-specific**. Swapping
  SBERT's all-MiniLM-L6-v2 (384-d) for mxbai-embed-large (1024-d,
  via ollama) at N=50 gives **SOMA rank1_frac=0.90 vs chroma=0.62**
  (+45% relative). mxbai underperforms SBERT as a base embedder (its
  512-token limit truncates long sessions), so BM25's contribution
  dominates — SOMA's lift widens when the cosine side is weaker.
  See `longmemeval_mxbai_cross_embedder.md`.

  **Replicated at scale on qwen3.5:9b**: same 500 items with strict
  prompting on the larger model give +22.0% F1 / +19.4% EM
  (vs +22.8%/+23.5% on 4b) — identical within noise. Decomposition
  stable at 37/63 recall/ranking (vs 34/66 on 4b). The SOMA lift is
  a retrieval-mechanical property, not LLM-specific. See
  `longmemeval_qa_compare_qwen9b_strict_findings.md`.

  **Validated with Claude Sonnet as the answerer** (via Unraid
  `claude-code-runner` Docker on the operator's Claude Max
  subscription, zero per-call API charges): N=500 strict,
  chroma_rerank (cosine + cross-encoder reranker) as the stronger
  baseline, SOMA hybrid as the challenger. SOMA F1 0.4161 vs chroma
  F1 0.3628 = **+14.7% F1 lift, +16.0% EM lift**. The lift is lower
  than on qwen4b because (a) chroma_rerank is a harder baseline than
  chroma_cosine (rerank closes most of the ranking gap), and (b) the
  stronger LLM is less sensitive to rank-position within the top-k.
  On `single-session-user` specifically, the lift stays dramatic:
  0.503 → 0.775 (+54%). See `longmemeval_claude_runner_findings.md`.

  **LLM-judge confirms the story on qwen9b too**: independent 4b-judge
  accuracy on the paired N=500 qwen9b strict predictions gives **+20.6%
  lift** (0.434 vs 0.360), nearly identical to the 4b model's +22.2%
  judge lift. single-session-user +65.7% lift on judge accuracy under
  qwen9b is the headline mechanistic result. Net disagreement count:
  53 items SOMA right / chroma wrong, 16 items reverse — SOMA
  correct where chroma isn't on +37 items. See
  `longmemeval_judge_findings_qwen9b.md`.

  **Cross-judge at N=500**: two independent LLM judges rate the same
  qwen4b-strict predictions: qwen4b as judge gives +22.2%, **Claude as
  judge (via the Unraid claude-code-runner) gives +23.7%** (0.418 vs
  0.338). Triple agreement on the headline (F1 +22.8% / qwen-judge
  +22.2% / Claude-judge +23.7%) is about as robust as a retrieval-driven
  QA lift claim can be without human annotation. Claude judge
  specifically widens the temporal-reasoning lift from +48% → **+58%**
  and the single-session-user lift from +59% → **+63%**. See
  `longmemeval_judge_findings_claude.md`.

  **LLM-judge on qwen4b** gave +22.2% judge-accuracy lift on the
  qwen4b predictions. Per-type reshuffling is the sharpest addition:
  - temporal-reasoning: F1 said +23%, judge says **+48%** (F1 was
    under-rewarding formatting variance like "two months ago" vs "2
    months prior").
  - single-session-preference: F1 said +2% (tied, benchmark-mismatch),
    judge says **+20%** (SOMA's specific-fact answers are correct
    against paraphrased-preference gold).
  - multi-session: F1 said +36%, judge says +4% (F1 was over-rewarding
    overlap on partial-wrong syntheses; this type is LLM-limited, not
    retrieval-limited).
  See `longmemeval_judge_findings.md` and
  `longmemeval_temporal_deep_dive.md`.
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
myelination) but across three serious attempts to route a
graph-shaped signal into retrieval (plastic-graph activation
benchmark, LLM-distilled projections, spatial distillation), none
produced a positive retrieval result. The substrate ships as a
research vehicle; **today's product wins are from hybrid retrieval,
not from the plastic graph**. See `paper-draft.md` §5 for the
research agenda.

### What we've ruled out (honest)

- **Plastic graph activation on retrieval**: Five consecutive
  measurement attempts on a synthetic 10-topic benchmark (v1 through
  v3c) failed to isolate a plastic-vs-frozen retrieval difference —
  the benchmark is architecturally intractable at current design
  (hybrid retrieve path bypasses the graph entirely; alternate paths
  time out or regress). After five nulls, this research thread is
  closed. See `plastic_graph_activation_findings.md`.

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
  `5be666a`..`d98d92a`) extended Direction 4a with learnable node
  positions coupled to projection weights via a fixed random map, plus
  competitive top-K distillation. The stack is clean (Phase 1: 12 TDD
  tasks, 2551 tests green; Phase 2 multi-seed sanity PASSES stability
  and confirms positions reshape meaningfully, KS ≈ 0.20). On full
  LoCoMo it performs WORSE than both Direction 4a and plain SOMA
  (soma-spatial R@5 = 0.335 vs soma-distilled 0.344 vs soma-random
  0.337, chroma 0.349), and turning locality ON makes it *worse still*
  (−0.007) — positions actively hurt retrieval, the definitive
  diagnostic. Root cause: coupling `p_i = normalize(P.T · W_i.flatten())`
  ties positions to projection weight matrices through a random
  compressor, so positions encode a rotation of existing projection
  information with no new semantic content. Infrastructure survives
  (position machinery, config flow-through, competitive top-K distill)
  for Option C (pairwise-distance distill) or Option D (spatial
  rerank) follow-ups, but this coupling formulation is closed. See
  `research/developmental/results/locomo_direction4b_findings.md`.

### Positive graph-side results (non-retrieval)

Two findings worth flagging since they constrain where the graph
substrate *does* work:

- **Positional locality on synthetic prediction (v0.5).** On a
  next-step MSE benchmark with a positional-locality filter
  (`synap_max_distance=0.5`), multi-seed SOMA beats the no-growth
  baseline by 0.0005–0.0009 MSE on hard regimes. Substrate-specific;
  does not transfer to retrieval (confirmed on LoCoMo and via
  Direction 4b above).
- **Retrieve latency.** Pure-cosine SOMA on CUDA runs retrieves at
  0.7ms/query vs Chroma's 1.7ms on CPU at 5,882 turns. The ops win is
  real even though the learning-graph win isn't.

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
