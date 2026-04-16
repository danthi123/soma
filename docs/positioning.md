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
- **Researchers** interested in plastic graph memory, structural
  sparsification, or complementary memory systems as an alternative
  to transformer-KV-plus-RAG.

## How it compares

| Capability                        | Chroma + RAG | Mem0 / Zep | **SOMA** |
|-----------------------------------|:------------:|:----------:|:--------:|
| Local-first, zero deps            | ✅           | ⚠️         | ✅       |
| Vector retrieval                  | ✅           | ✅         | ✅       |
| Working-memory window             | ❌           | ⚠️         | ✅       |
| Episodic-memory store             | ❌           | ✅         | ✅       |
| Plastic graph substrate (in-place) | ❌           | ❌         | ✅\*     |
| Consolidation hook (learning-ready) | ❌           | ❌         | ✅       |
| Single-file "brain" portability   | ❌           | ❌         | ✅       |
| Swap LLM without losing memory    | ✅           | ⚠️         | ✅       |
| Learns from use                   | ❌           | ⚠️         | ✅       |

⚠️ = partial / conditional on provider.
\* = plasticity substrate ships; current memory workload doesn't
trigger growth/pruning thresholds (see `paper-draft.md` §5 for the
research agenda to activate it).

The differentiator today is **efficiency + the substrate for
learning to come**. On benchmarks (see `benchmarks/reports/`):

- **SOMA matches Chroma on retrieval quality** (identical Recall@3 /
  MRR@3 / NDCG@3 on a labeled 50-fact / 26-query synthetic set with
  the same sbert embedder).
- **SOMA is 22.6× smaller on disk** (85 KB vs 1920 KB) and **1.3×
  faster to retrieve** than Chroma at the same N. Store is **2.8×
  faster** per op.
- **Old memories don't rot.** A 30-day simulation with 150 facts
  streamed in holds old-fact Recall@3 at 0.883 — essentially level
  with recent-fact recall (0.938).

The graph is plastic-by-construction (synaptogenesis, pruning,
myelination) but under the memory-only workload the growth knobs
don't fire — graph stays at seed size, retrieval scores stay at
cosine-over-sbert. The graph-as-retrieval-signal is an **open
research question** the `benchmarks/reports/paper-draft.md` document
scopes explicitly; today's efficiency win does not depend on it.

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

## Licensing & commercial story

Pre-1.0, everything is MIT. Intent is to keep the core MIT post-1.0
and sell hosting / multi-device sync / optional enterprise features
rather than relicensing the core.

## Roadmap

- **Stage 2** — MemoryLayer API + persistent-chat demo
- **Stage 3** — `pip install soma-memory`, LangChain + LlamaIndex
  connectors, benchmark harness vs Chroma+RAG and Mem0
- **Stage 4** — research side-bets (graph plasticity vs static
  retrieval, WM vs recency-window), verbalizer-on-retrieved-context
  retrain, Docker / deploy story

Detailed roadmap: `docs/plans/2026-04-15-memory-layer-pivot.md`.

## Research appendix

For the architectural and empirical backstory — the developmental-AI
hypothesis, the hybrid-brain experiment, the three-regime ablation
that surfaced the projector-prior — see
`docs/whitepaper.md` and `docs/progress/HYBRID_PIVOT.md`.
