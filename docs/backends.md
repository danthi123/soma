# Vector backends

SOMA's memory layer is pluggable at the vector-store layer. A
`VectorBackend` is a small Protocol that MemoryLayer routes every
`add` / `remove` / `search` / `get_vectors` call through; everything
else (ids, texts, metadata, timestamps, WAL, BM25, cross-encoder
rerank, graph-aware retrieval) stays inside MemoryLayer regardless of
which backend is attached.

Two adapters ship in-tree:

- **`InProcBackend`** (default, zero new deps)
- **`QdrantBackend`** (optional extra: `pip install soma[qdrant]`)

Swapping adapters does not change MemoryLayer's Python-facing API:
`store`, `retrieve(where=...)`, `related`, `forget`, `consolidate`,
`save`, and `load` behave the same. What changes is where the vectors
live and how the query runs.

## When to choose which

| Use case | Backend | Why |
| --- | --- | --- |
| Local agent, <=100K entries | `InProcBackend` (flat) | Exact recall, no setup, no network |
| Local agent, 100K-1M entries | `InProcBackend` (hnsw) | Fast ANN, still no deps beyond FAISS |
| Local agent, <=20K entries, want durable on-disk | `QdrantBackend(mode="local")` | Single-file persistence, survives process restarts |
| Production agent, >20K or multi-host | **`QdrantBackend(mode="http")`** | Horizontal scale, real durability, multi-tenant |
| Quick tests / CI | `QdrantBackend(mode="memory")` | Embedded in-process, volatile |

**Rule of thumb:** start with `InProcBackend`. Switch to
`QdrantBackend(mode="http")` when you cross 100K entries per bundle
or need multiple processes sharing the same memory.

## Protocol

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class VectorBackend(Protocol):
    @property
    def ntotal(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def supports_filter_pushdown(self) -> bool: ...
    @property
    def name(self) -> str: ...

    def open(self) -> None: ...
    def close(self) -> None: ...
    def clear(self) -> None: ...

    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def remove(self, ids: list[str]) -> None: ...
    def get_vectors(self, ids: list[str]) -> np.ndarray: ...

    def search(self, query, k, *, exclude_ids=None, where=None) -> list[tuple[str, float]]: ...
    def search_subset(self, query, candidate_ids, k) -> list[tuple[str, float]]: ...

    def snapshot(self, bundle_dir) -> None: ...
    def restore(self, bundle_dir) -> None: ...
```

Vectors are numpy `float32` at the protocol boundary — MemoryLayer
handles the torch→numpy conversion inside `_vec_to_np` so adapters
never see a torch tensor. This keeps backends that speak
arrow/C/HTTP out of torch's import graph.

Scores are cosine similarity in `[-1, 1]`. Higher is more similar.

## InProcBackend

```python
from soma.memory.api import MemoryLayer  # default is InProc

mem = MemoryLayer.with_sbert()  # or tokenizer/encoder
mem.store("hello")
mem.retrieve("hi", k=5)
```

Under the hood: a `(ntotal, dim)` float32 matrix kept in numpy, with a
lazy FAISS index built when `ntotal >= faiss_threshold` (default
10 000). Two FAISS modes:

- `faiss_index_type="flat"` — exact, SIMD. Recall@10 = 1.0. Default.
- `faiss_index_type="hnsw"` — approximate, much faster at >100K.
  Recall@10 ~ 0.98 on typical sbert corpora.

Snapshot writes `memory_embeddings.pt` (a torch tensor of shape
`(ntotal, dim)`) — identical to pre-Phase-6 bundles so old bundles
keep loading without migration.

**Tradeoffs:**
- Pro: zero network, zero setup, zero new deps beyond FAISS (which
  was already optional).
- Pro: exact recall with `flat`, strong approximate recall with HNSW.
- Con: single-process. Multi-worker uvicorn shares WAL but each
  worker rebuilds its own FAISS index.
- Con: memory-resident — `ntotal * dim * 4 bytes` in RAM at all
  times. 1M entries at 384-dim = ~1.5 GB.

## QdrantBackend

Install: `pip install soma[qdrant]`.

```python
from soma.memory.api import MemoryLayer
from soma.memory.backends.qdrant import QdrantBackend

# HTTP mode — the scale story
backend = QdrantBackend(
    mode="http",
    dim=384,
    url="http://qdrant.internal:6333",
    collection="prod_agent",
)
mem = MemoryLayer.with_sbert(backend=backend)
```

### HTTP mode (recommended for production)

Qdrant stays authoritative on disk; MemoryLayer's WAL is the replay
source if the collection ever has to be rebuilt. Snapshot writes a
`backend.json` sidecar with the URL + collection name so `restore`
re-points a fresh client at the same data without re-ingesting.

Scale: tens of millions of vectors on a single Qdrant node. Billions
across a cluster.

### Local-file mode (≤20K cap)

```python
backend = QdrantBackend(mode="local", dim=384, path="./qdrant_data")
```

Embedded Qdrant core stored in `./qdrant_data`. Fine for personal
agents, demos, single-user setups. A `UserWarning` fires when
`ntotal > 20 000` — past that, migrate to HTTP. The cap isn't a hard
limit, it's a "we've measured the cliff here" flag. The 20K threshold
matches benchmarks on consumer hardware (RTX 3090, 32 GB RAM).

### In-memory mode (tests, demos)

```python
backend = QdrantBackend(mode="memory", dim=384)
```

Embedded Qdrant core, volatile. Useful for tests (no disk cleanup) or
demos (no network). Not for production — restart = empty store.

### Filter pushdown

`QdrantBackend.supports_filter_pushdown = True`. MemoryLayer's
`retrieve(where=...)` delegates to the backend, which translates the
Chroma-style dict into Qdrant's native `Filter` model. Supported
operators mirror `_COMPARE_OPS`:

```python
mem.retrieve("quantum", k=5, where={"tag": "physics"})
mem.retrieve("2023", k=5, where={"year": {"$gte": 2020}})
mem.retrieve("any", k=5, where={"tag": {"$in": ["a", "b", "c"]}})
```

Unsupported operators (e.g. `$regex`) raise
`FilterPushdownUnsupported`; MemoryLayer catches and falls back to
its Python pre-filter path. This keeps semantics identical across
adapters — callers never have to check which backend is attached.

### Node-ID mapping

Qdrant requires integer (or uuid-string) point ids; SOMA uses
`uuid4().hex` node ids. QdrantBackend maintains an internal
`node_id ↔ int64 point_id` dict and stores the `node_id` as a
payload field so rehydration via `scroll` works after reopening a
collection.

## Writing a new backend

Implement the `VectorBackend` protocol. The runtime-checkable
decorator means `isinstance(obj, VectorBackend)` works on any
duck-typed object — you don't have to subclass anything.

```python
from pathlib import Path
import numpy as np

class MyBackend:
    supports_filter_pushdown = False
    name = "my-backend"

    def __init__(self, *, dim: int) -> None:
        self._dim = dim
        # ... your store ...

    @property
    def ntotal(self) -> int: ...
    # etc.
```

Pass it to `MemoryLayer(backend=MyBackend(dim=384))`. MemoryLayer
will route every vector op through it. If you set
`supports_filter_pushdown=True`, also implement the `where=` branch
of `search` and raise `FilterPushdownUnsupported` for any op you
can't translate — the fallback path still works.

## Benchmarks

See `benchmarks/run_backend_matrix.py` for the adapter matrix
harness: scale 100K and 1M, shared sbert cache, recall@10 vs exact,
retrieve p50/p95. The generated report lives at
`benchmarks/reports/backend_matrix.md`.

Ship-blocker thresholds pinned by Phase 6:

- QdrantHTTP retrieve p50 ≤ 2× InProc at 1M
- Recall@10 within 0.02 of InProc at same k
- Store throughput ≥ 500 ops/s on HTTP mode
