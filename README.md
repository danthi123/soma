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
| Graph structure that grows/prunes | no | no | **yes** |
| Consolidation (sleep replay) | no | no | **yes** |
| Single-directory brain portability | no | no | **yes** |
| Swap LLM without losing memory | yes | partial | **yes** |

Benchmark (50 facts, same embedder): identical Recall@3, **4x faster retrieves**, **22x smaller on disk** vs Chroma. See `reports/memory-layer-vs-rag-benchmark.md`.

## Graph consolidation (optional)

Attach a SOMA graph to get structural plasticity — the memory doesn't just store, it *restructures* with use:

```python
from soma.memory import MemoryLayer
from soma.system import SOMA
from soma.core.config import SOMAConfig

mem = MemoryLayer.with_sbert()
soma = SOMA(SOMAConfig())

mem.attach_soma(soma, tokenizer, encoder)
mem.consolidate()  # triggers synaptogenesis, pruning, myelination
```

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

MemoryLayer auto-switches to a FAISS index when the store exceeds 10K entries (configurable via `faiss_threshold`). Below that, the O(N) linear scan is faster with zero overhead.

## Demos

```bash
python scripts/demo_memory_layer.py           # pure API, no GPU
python scripts/demo_chat_persistent.py --dry-run  # persistent chat
python scripts/benchmark_memory.py            # vs Chroma benchmark
python scripts/experiment_plasticity.py       # graph plasticity proof
```

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
- [Benchmark report](reports/memory-layer-vs-rag-benchmark.md)
- [Plasticity experiment](reports/plasticity-experiment.md)

## License

MIT
