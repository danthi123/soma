# SOMA

**Local-first agent memory that learns.**

A drop-in replacement for vector-DB + RAG where the store is a plastic graph that grows and prunes with use. Store text, retrieve by meaning, and let the structure reshape itself over time. Everything local, everything on your disk, LLM-agnostic.

> **60-second tour**: install, index a folder, chat — see [`docs/quickstart.md`](docs/quickstart.md). Picking SOMA over Mem0/Letta/Zep/Chroma? [`docs/comparison.md`](docs/comparison.md). Patterns + recipes: [`docs/cookbook.md`](docs/cookbook.md).

## Install

```bash
# Minimal (torch + tokenizers only, ~2GB):
pip install -e .

# With sentence-transformers for quality retrieval:
pip install -e ".[sbert]"

# With REST API server:
pip install -e ".[serve]"

# With FAISS for >10K entries:
pip install -e ".[ann]"

# With framework adapters:
pip install -e ".[langchain]"   # LangChain
pip install -e ".[llamaindex]"  # LlamaIndex

# Everything:
pip install -e ".[sbert,ann,serve,langchain,llamaindex]"
```

## Quick start

```python
from soma.memory import MemoryLayer

mem = MemoryLayer.with_sbert()  # uses all-MiniLM-L6-v2

# Store with metadata (same shape as Chroma)
mem.store("user lives in Portland, OR", metadata={"user": "alex"})
mem.store("user is vegetarian", metadata={"user": "alex"})
mem.store("user's dog is named Luna", metadata={"user": "alex"})

# Retrieve — vector DB parity
hits = mem.retrieve("dietary restrictions", k=3)

# Or with recall boosters (go beyond any peer DB's ceiling):
hits = mem.retrieve(
    "dietary restrictions", k=3,
    where={"user": "alex"},   # metadata pre-filter (Chroma-style)
    hybrid_alpha=0.3,          # blend BM25 + cosine
)

# Optional cross-encoder rerank for +21 pp R@5:
from soma.memory.rerank import CrossEncoderReranker
mem.attach_reranker(CrossEncoderReranker())
hits = mem.retrieve("...", k=5, rerank_top_n=20)

# Persist (portable single-directory bundle)
mem.save("my-brain/")
mem = MemoryLayer.load("my-brain/")
```

**End-to-end chat with your LLM of choice:**

```python
from soma.llm import RAGSession, backend_from_env
# Auto-picks Ollama if running, else OpenAI/Anthropic if API key set,
# else local HuggingFace. Override with SOMA_LLM_BACKEND.
chat = RAGSession(memory=mem, llm=backend_from_env())
print(chat.ask("where does the user live?").text)
```

## How it compares

| Capability | Chroma | Mem0 / Zep | Pinecone | **SOMA** |
|---|:---:|:---:|:---:|:---:|
| Vector retrieval | yes | yes | yes | yes |
| Local-first, zero cloud deps | yes | partial | no | yes |
| Metadata `where` filter at retrieve | yes | yes | yes | **yes** |
| Hybrid BM25 + vector (built-in) | no | partial | partial | **yes** |
| Cross-encoder rerank (built-in) | no | no | partial | **yes** |
| LLM query expansion (built-in) | no | partial | no | **yes** |
| Plug-and-play LLM backends | no | partial | no | **yes** (5 shipped) |
| Plastic graph substrate (in-place) | no | no | no | **yes**\* |
| Single-directory brain portability | partial | no | no | **yes** |
| Multi-tenant REST (bundles/{name}) | no | yes | yes | **yes** |
| Swap LLM without losing memory | yes | partial | yes | **yes** |

\* substrate ships; current memory workload doesn't trigger growth/pruning thresholds — see `benchmarks/reports/paper-draft.md` §5 for the research agenda to activate it.

Full comparison + migration notes: [`docs/comparison.md`](docs/comparison.md).

**Benchmark (same sbert embedder, measured vs Chroma):**

*Mechanics — SOMA wins everywhere:*
- Quality parity: identical Recall@3 / MRR@3 / NDCG@3 at same embedder (by construction).
- Disk: **22.6× smaller at 50 facts**, narrowing to **1.4× at 20K** and **1.42× at 100K**.
- Store (full pipeline 1K–20K): **3.2–3.6× faster** per op (durable).
- Store (index-only 100K): **~3500× faster** (SOMA 0.4 s vs Chroma 23.6 min) — Chroma pays ~14 ms/op for SQLite+HNSW metadata regardless of embed cost.
- Retrieve HNSW backend: **1.18–1.25× faster** at 1K–20K, growing to **5.12× at 100K** while preserving identical recall.
- Drift: 30-day simulation, old-fact Recall@3 = 0.883 ≈ recent 0.938 (memory doesn't rot).

*Recall boosters — SOMA goes beyond the same-embedder ceiling:*

Peer vector DBs (Chroma, LanceDB, Pinecone) all tie SOMA on recall when using the same embedder — by mathematical construction (identical cosine over identical vectors). To beat them, SOMA ships three opt-in boosters that they don't have built-in:

| Retrieval strategy | R@1 | R@5 | Lift R@5 vs cosine |
| --- | ---: | ---: | ---: |
| Pure cosine (= any peer DB's ceiling) | 0.098 | 0.238 | — |
| Hybrid BM25+cosine | 0.207 | 0.415 | **+17.7 pp (+74%)** |
| Cross-encoder rerank | 0.203 | 0.309 | +7.1 pp |
| **Hybrid + rerank** | **0.287** | **0.450** | **+21.2 pp (+89%)** |

Measured on LoCoMo (5,882 turns, 1,982 questions). Turning both knobs on triples R@1 and adds ~34 ms of latency on top of baseline 13 ms.

See `benchmarks/reports/` for the full suite + paper-draft aggregator.

## Graph consolidation (optional, research)

Attach a SOMA graph to surface the plasticity substrate. Under the
current memory-only workload the growth thresholds (training-tuned)
don't fire and the graph stays at seed size — the substrate ships,
the activation is the open research question (see paper-draft §5):

```python
from soma.memory import MemoryLayer
from soma.system import SOMA
from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

mem = MemoryLayer.with_sbert()
soma_tokenizer = train_bpe_tokenizer(["..."], vocab_size=1024)
soma_encoder = TextEncoder(soma_tokenizer, embed_dim=32, max_seq_len=128)
soma = SOMA(SOMAConfig(
    vocab_size=1024, text_embed_dim=32, sensor_output_dim=32,
))
mem.attach_soma(soma, soma_tokenizer, soma_encoder)
mem.consolidate()  # incremental: only processes entries past the cursor
```

Graph re-rank is off by default (`graph_rerank_alpha=0.0`); set
non-zero to opt into the experimental blend.

## Framework integrations

```python
# LangChain
from soma.integrations.langchain import SomaRetriever
retriever = SomaRetriever(memory=mem, k=5)
docs = retriever.invoke("what's the user's name?")

# LlamaIndex
from soma.integrations.llamaindex import SomaRetriever
nodes = SomaRetriever(memory=mem, k=5).retrieve("what's the user's name?")
```

## REST API / Docker

```bash
# Local:
uvicorn soma.serve:app --port 8420

# Docker:
docker compose up

# Then:
curl -X POST http://localhost:8420/store \
  -H 'Content-Type: application/json' \
  -d '{"text": "user lives in Portland"}'

curl -X POST http://localhost:8420/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "where does the user live?", "k": 3}'
```

## Scaling

MemoryLayer auto-switches to a FAISS `IndexFlatIP` (exact) when the store exceeds 10K entries (configurable via `faiss_threshold`). Below that, the O(N) linear scan is faster with zero overhead.

For multi-K stores where retrieve speed matters, opt into the approximate HNSW backend:

```python
mem = MemoryLayer.with_sbert(...)
mem._faiss_index_type = "hnsw"  # or pass via constructor
```

HNSW preserves identical Recall@3 on the labeled benchmark and runs **1.18–1.21× faster than Chroma's HNSW** across the 1K/5K/20K range tested, growing to **5.12× faster at 100K** under an index-only methodology that amortizes the sbert embed cost. Defaults stay exact-flat so callers get vector-DB-equivalent recall guarantees out of the box.

At enterprise scale (100K entries, pre-computed embeddings), SOMA ingests the entire corpus in **0.4 seconds vs Chroma's 23.6 minutes** — SOMA's store is essentially a tensor-append while Chroma pays ~14 ms per insert for SQLite + HNSW metadata. See `benchmarks/reports/scale_enterprise_100k.md`.

## CLI

```bash
pip install -e .
soma index --wiki path/to/docs --bundle my-brain/
soma chat  --bundle my-brain/                  # auto-picks LLM backend
soma stats --bundle my-brain/
soma serve --port 8420                          # REST API
```

`soma chat` auto-detects a backend: Ollama if running, OpenAI/Anthropic if `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` is set, otherwise local HuggingFace. Override with `--backend ollama|openai|anthropic|openai-compat|hf|dry-run`. See [`docs/llm-backends.md`](docs/llm-backends.md).

## Demos

```bash
python scripts/demo_memory_layer.py             # pure API tour, no GPU
python scripts/demo_chat_persistent.py --dry-run  # persistent conversation
python scripts/demo_wiki_chat.py --wiki-dir docs/ --bundle artifacts/wiki-brain --index --chat --dry-run
python scripts/demo_memory_inspect.py stats --bundle artifacts/wiki-brain
python scripts/demo_related_browser.py --bundle artifacts/wiki-brain --seed "consolidation"
```

Walkthrough: [`docs/demos.md`](docs/demos.md).

## Benchmarks

```bash
python -m benchmarks.run_retrieval              # vs Chroma at 50 facts
python -m benchmarks.run_scale_vs_chroma        # vs Chroma at 1K/5K/20K
python -m benchmarks.run_scale_enterprise --n 100000  # index-only, 100K/1M
python -m benchmarks.run_locomo                 # real conversations (10 dialogues, 1986 Qs)
python -m benchmarks.run_graph_ablation         # alpha sweep + stable capture
python -m benchmarks.run_plasticity_scale       # graph growth at 100-2000
python -m benchmarks.run_longitudinal_drift     # 30-day drift simulation
```

Reports land in `benchmarks/reports/`. The paper-draft aggregator (`benchmarks/reports/paper-draft.md`) wires every claim back to its committed script + report.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
mypy src/soma/
```

## Docs

- **[Quickstart](docs/quickstart.md)** — 60 seconds to a working chat-with-your-wiki demo
- **[Comparison](docs/comparison.md)** — SOMA vs Chroma / Mem0 / Letta / Zep / Pinecone
- **[Cookbook](docs/cookbook.md)** — 16 recipes (hybrid search, rerank, query expansion, metadata filtering, multi-tenant REST, migrations)
- **[Demos](docs/demos.md)** — every shipped demo, what it shows, when to run it
- **[LLM backends](docs/llm-backends.md)** — Ollama / OpenAI / Anthropic / vLLM / HF
- **[Recall improvements](docs/recall-improvements.md)** — hybrid BM25, cross-encoder rerank, research agenda
- [Product positioning](docs/positioning.md)
- [Pivot decision + roadmap](docs/plans/2026-04-15-memory-layer-pivot.md)
- [Architecture whitepaper](docs/whitepaper.md)
- [Paper draft (benchmark aggregator)](benchmarks/reports/paper-draft.md)

## License

MIT
