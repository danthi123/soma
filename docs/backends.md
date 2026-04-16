# Vector backends

SOMA's memory layer is pluggable at the vector-store layer. A
`VectorBackend` is a small Protocol that MemoryLayer routes every
`add` / `remove` / `search` / `get_vectors` call through; everything
else (ids, texts, metadata, timestamps, WAL, BM25, cross-encoder
rerank, graph-aware retrieval) stays inside MemoryLayer regardless of
which backend is attached.

Four adapters ship in-tree:

- **`InProcBackend`** (default, zero new deps)
- **`LanceDBBackend`** (optional extra: `pip install soma[lancedb]`)
- **`QdrantBackend`** (optional extra: `pip install soma[qdrant]`)
- **`ChromaBackend`** (optional extra: `pip install soma[chroma]`)

Swapping adapters does not change MemoryLayer's Python-facing API:
`store`, `retrieve(where=...)`, `related`, `forget`, `consolidate`,
`save`, and `load` behave the same. What changes is where the vectors
live and how the query runs.

## When to choose which

| Use case | Backend | Why |
| --- | --- | --- |
| Local agent, <=100K entries | `InProcBackend` (flat) | Exact recall, no setup, no network |
| Local agent, 100K-1M entries | `InProcBackend` (hnsw) | Fast ANN, still no deps beyond FAISS |
| Local agent, 1M+ entries, durable on-disk, no server | **`LanceDBBackend`** | Embedded, arrow-native, 10M+ scale |
| Local agent, <=20K entries, want durable on-disk | `QdrantBackend(mode="local")` | Single-file persistence, survives process restarts |
| Production agent, multi-host | **`QdrantBackend(mode="http")`** | Horizontal scale, multi-tenant, over-the-wire |
| Quick tests / CI | `QdrantBackend(mode="memory")` | Embedded in-process, volatile |
| Migrating from existing Chroma RAG | **`ChromaBackend`** | Point SOMA at your existing Chroma collection — zero export/reimport |

**Rule of thumb:** start with `InProcBackend`. Switch to
`LanceDBBackend` when you want persistence past 20K without running
a server; switch to `QdrantBackend(mode="http")` when you need
multi-host or multi-tenant scale. Pick `ChromaBackend` when you
already have a populated Chroma collection and don't want to
migrate vectors out of it.

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

## LanceDBBackend

Install: `pip install soma[lancedb]`.

```python
from soma.memory.api import MemoryLayer
from soma.memory.backends.lancedb import LanceDBBackend

backend = LanceDBBackend(
    path="./lancedb_data",
    dim=384,
    table_name="prod_agent",
    index_type="ivf_pq",  # flat | ivf_pq | hnsw
)
mem = MemoryLayer.with_sbert(backend=backend)
```

**The pitch: local-first scale past Qdrant-local's 20K cap, no
server.** LanceDB is arrow-native, columnar, and fully embedded — a
table IS a directory on disk, which composes cleanly with SOMA's
bundle layout. Scales to 10M+ vectors in-process with IVF-PQ or
HNSW indexes. No daemon, no network, no Docker image to ship — same
"one directory = one brain" story as InProc, with persistence that
survives process restarts and doesn't balloon RAM linearly with N.

### Index-type choice

| Index | Recall | Query | Build | Disk | Sweet spot |
| --- | :---: | :---: | :---: | :---: | --- |
| `flat` | 1.0 (exact) | O(N) linear scan | instant | smallest | <=100K |
| `ivf_pq` | 0.95-0.98 | sub-linear ANN | seconds | smallest | 100K-10M |
| `hnsw` (`IVF_HNSW_SQ`) | 0.98-0.99 | sub-linear ANN | minutes | larger | 1M-10M |

**Recommended starting defaults:** `index_type="flat"` up to 100K,
`index_type="ivf_pq"` with `num_partitions=256`,
`num_sub_vectors=96` past that. Index build is lazy — it fires the
first time `ntotal` crosses `auto_index_threshold` (default 50K).
Below the threshold, LanceDB just scans the table in sorted row
order, which is fine at small N.

### Bundle layout

```
bundle_dir/
├── memory_index.json        # MemoryLayer entry metadata
├── memory_embeddings.pt     # (still present for cross-backend parity;
│                            #  LanceDB has its own copy in lancedb/)
├── tokenizer.json           # if TextEncoder path
├── encoder.pt               # if TextEncoder path
├── backend.json             # {"backend": "lancedb", "dim": ..., ...}
└── lancedb/                 # the actual LanceDB table directory —
    ├── vectors.lance/       #   versioned append-only fragments
    └── ...                  #   manifest + data files
```

`snapshot(bundle_dir)` compacts the LanceDB table (to shrink the
fragment count) then copies the directory. `restore(bundle_dir)` is
the inverse — wipe the live path, copy the bundle's `lancedb/` back,
reopen the table. This makes the bundle fully self-contained: an
operator can ship the directory to another machine and reopen it
there without any re-ingest work.

### Filter pushdown

`LanceDBBackend.supports_filter_pushdown = True`. MemoryLayer's
`retrieve(where=...)` delegates to the backend, which translates the
Chroma-style dict into LanceDB's SQL-like predicate string via
`lancedb_filter.to_lancedb_where`. Supported operators mirror
`_COMPARE_OPS`:

```python
mem.retrieve("quantum", k=5, where={"tag": "physics"})
mem.retrieve("2023", k=5, where={"year": {"$gte": 2020}})
mem.retrieve("any", k=5, where={"tag": {"$in": ["a", "b", "c"]}})
```

The v1 LanceDB schema is `{id: utf8, vector: list<float32>[dim]}`.
Metadata columns are not first-class yet (they live on MemoryLayer's
Python side); when MemoryLayer's `_retrieve_with_filter` tries to
push down a filter on a non-existent column the backend converts
LanceDB's rust-side schema error into `FilterPushdownUnsupported`,
and MemoryLayer transparently falls back to its Python pre-filter +
`search_subset` path (which LanceDB serves via `WHERE id IN (...)`).
The observable contract is identical across backends: only entries
matching the filter come back.

**Tradeoffs:**
- Pro: embedded, no server — unlike Qdrant HTTP there's no daemon
  to run or monitor.
- Pro: on-disk from day one. RAM footprint independent of N.
- Pro: scales well past Qdrant-local's 20K cap; tested to 10M+ in
  LanceDB's own benchmarks.
- Pro: arrow-native columnar storage means snapshots are small and
  compose with pandas / polars for offline analytics.
- Con: ANN indexes (`ivf_pq`, `hnsw`) take seconds-to-minutes to
  build at scale. Build is lazy to avoid paying that cost until the
  store is big enough to need it.
- Con: metadata filters are not server-side yet — they fall back to
  Python pre-filter + subset rank. Future work can widen the
  table schema to mirror metadata fields for true pushdown.

## ChromaBackend

Install: `pip install soma[chroma]`.

```python
from soma.memory.api import MemoryLayer
from soma.memory.backends.chroma import ChromaBackend

backend = ChromaBackend(
    path="./chroma_data",     # chromadb.PersistentClient directory
    collection_name="my_agent",
    dim=384,
)
mem = MemoryLayer.with_sbert(backend=backend)
```

**The pitch: drop-in replacement for an existing Chroma store.** A
large fraction of agent deployments already keep their vectors in a
Chroma collection; the switching cost from "export + reimport" to
"point `MemoryLayer` at the same Chroma store" is this adapter.
SOMA keeps ids/texts/metadata/WAL/BM25/cross-encoder/graph-rerank on
its own side; vectors stay in Chroma.

For non-default setups (custom auth, tenancy/database, remote HTTP
mode), pass a pre-built client instead of a path:

```python
import chromadb

client = chromadb.HttpClient(host="chroma.internal", port=8000)
backend = ChromaBackend(client=client, collection_name="my_agent", dim=384)
```

Filter pushdown mirrors the other adapters (`$eq`/`$ne`/`$gt`/
`$gte`/`$lt`/`$lte`/`$in`/`$nin`). Metadata written via
MemoryLayer's own `store(metadata=...)` path stays on SOMA's side
rather than flowing into Chroma's `metadatas`, so `retrieve(where=...)`
falls back to SOMA's Python pre-filter + `search_subset` — the
observable contract (only matching rows come back) is identical.

Pinned minimum: `chromadb>=0.5`. Earlier versions had a different
`delete_collection` + recreate story and still exposed a now-removed
`persist()` method; requiring 0.5 lets the adapter skip that branch.

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

### Cross-version snapshot testing

`QdrantBackend` stamps the `qdrant-client` library version into the
bundle's `backend.json` when snapshotting. For HTTP deploys, the
durable format on the wire is Qdrant's own server-side snapshot —
that's what an operator would use to migrate a collection between
Qdrant versions (e.g., bumping the server from 1.11 to 1.13 during a
rolling upgrade). A silent snapshot-format regression here would
corrupt a production collection on restore.

Phase 24 added an optional test matrix that pins that contract:

```
tests/test_memory/test_qdrant_version_compat.py
```

The matrix spins real Qdrant containers at versions **1.11.3**,
**1.12.4**, and **1.13.5** via `testcontainers-python`, then:

- **3 smoke tests** — round-trip `add` + `search` through
  `QdrantBackend(mode="http")` pointed at each container version.
- **9 cross-version tests** — every (src, tgt) pair: create a native
  Qdrant snapshot on the src container, upload+recover it on a fresh
  tgt container, and assert that `QdrantBackend.search` still
  returns the expected top-1 id with score > 0.99.

The matrix is pinned to Qdrant **1.11+** — the REST endpoint layout
for `recover-from-snapshot` was reworked between 1.10 and 1.11, so
restoring a 1.10-era snapshot into 1.11+ is not a supported path and
would need adapter-side version dispatch we haven't written.

#### Running locally

The matrix is gated behind an environment flag AND the
`slow_qdrant` pytest marker so default runs stay fast:

```bash
pip install -e ".[qdrant,qdrant-test]"
SOMA_QDRANT_VERSION_MATRIX=1 \
    pytest tests/test_memory/test_qdrant_version_compat.py -q
```

Requires a working Docker daemon (Docker Desktop on Windows/macOS,
`dockerd` on Linux). Total wall-clock: ~6–10 min for all 12 tests.
Default runs (without the env flag or without `testcontainers`
installed) skip the file cleanly at collection time.

If you add a new Qdrant version to the matrix, bump
`_QDRANT_VERSIONS` in `tests/test_memory/test_qdrant_version_compat.py`
and re-run locally. A follow-up will wire a weekly CI job once the
host target (GitHub Actions vs a self-hosted Gitea runner) is
decided.

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
