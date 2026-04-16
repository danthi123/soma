# SOMA

**Local-first agent memory that learns.**

A drop-in replacement for vector-DB + RAG where the store is a plastic graph that grows and prunes with use. Store text, retrieve by meaning, and let the structure reshape itself over time. Everything local, everything on your disk, LLM-agnostic.

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

# Store
mem.store("user lives in Portland, OR")
mem.store("user is vegetarian")
mem.store("user's dog is named Luna")

# Retrieve
hits = mem.retrieve("dietary restrictions", k=3)
for hit in hits:
    print(hit.text, hit.score)

# Persist
mem.save("my-brain/")

# Later...
mem = MemoryLayer.load("my-brain/")
hits = mem.retrieve("where does the user live?", k=1)
```

## How it compares

| Capability | Chroma + RAG | Mem0 / Zep | **SOMA** |
|---|:---:|:---:|:---:|
| Vector retrieval | yes | yes | yes |
| Local-first, zero cloud deps | yes | partial | yes |
| Plastic graph substrate (in-place) | no | no | **yes**\* |
| Consolidation hook (learning-ready) | no | no | **yes** |
| Single-directory brain portability | no | no | **yes** |
| Swap LLM without losing memory | yes | partial | **yes** |

\* substrate ships; current memory workload doesn't trigger growth/pruning thresholds — see `benchmarks/reports/paper-draft.md` §5 for the research agenda to activate it.

**Benchmark (same sbert embedder):**
- Quality: identical Recall@3 / MRR@3 / NDCG@3 to Chroma at 50 facts.
- Disk: **22.6× smaller at 50** narrowing to **1.4× at 20K** as Chroma's overhead amortizes.
- Store: **3.2–3.6× faster across all N tested** (durable claim).
- Retrieve: SOMA-flat trails Chroma's HNSW by 7–22%; opt-in HNSW backend wins by a durable **1.18–1.21× across every N tested**, identical recall preserved.
- Drift: 30-day simulation, old-fact Recall@3 = 0.883 ≈ recent 0.938 (memory doesn't rot).

See `benchmarks/reports/` for the full benchmark suite + paper-draft aggregator.

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

## Demos

```bash
python scripts/demo_memory_layer.py             # pure API, no GPU
python scripts/demo_chat_persistent.py --dry-run  # persistent chat
```

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

- [Product positioning](docs/positioning.md)
- [Pivot decision + roadmap](docs/plans/2026-04-15-memory-layer-pivot.md)
- [Architecture whitepaper](docs/whitepaper.md)
- [Paper draft (benchmark aggregator)](benchmarks/reports/paper-draft.md)

## License

MIT
