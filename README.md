# SOMA

**Local-first agent memory that learns.**

A drop-in replacement for vector-DB + RAG where the store is a plastic graph that grows and prunes with use. Store text, retrieve by meaning, and let the structure reshape itself over time. Everything local, everything on your disk, LLM-agnostic.

## Quick start

```bash
pip install -e .
```

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

## Demos

```bash
# Pure API demo (no GPU needed):
python scripts/demo_memory_layer.py

# Persistent chat with LLM (--dry-run skips LLM):
python scripts/demo_chat_persistent.py --dry-run

# Benchmark vs Chroma:
python scripts/benchmark_memory.py
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

## License

MIT
