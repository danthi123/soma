# Phase 27: Chroma-as-Backend Adapter

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Ship a Chroma-backed `VectorBackend` so Chroma users can
drop SOMA on top of their existing collections. Strategic migration
hook: lowers the switching cost from "export + reimport" to
"point `MemoryLayer` at the same Chroma store." Fourth pluggable
adapter alongside InProcFlat, Qdrant (local + HTTP), and LanceDB.

**Architecture:**
- `src/soma/memory/backends/chroma.py` — `ChromaBackend` class
  implementing the `VectorBackend` Protocol. Uses the `chromadb`
  Python client directly (not HTTP — the client embeds an in-process
  store when no remote URL is given, matching our "local-first" story).
- Client modes:
  * `ChromaBackend(path="./chroma.db")` — persistent local store.
  * `ChromaBackend(client=...)` — caller passes a pre-built
    `chromadb.PersistentClient` / `HttpClient`. Escape hatch for
    non-default configs.
- `supports_filter_pushdown=True` via a `chroma_filter.to_chroma_where`
  translator (similar to `lancedb_filter` — Chroma's `where` clause
  uses MongoDB-like `$eq`/`$gt`/`$in` ops, which map cleanly from our
  internal filter spec).
- Snapshot = dump Chroma's persist dir into the bundle; restore =
  inverse. Chroma's `PersistentClient` already handles on-disk
  consistency, so snapshot is a directory copy (locked / drained
  first via `client.persist()` if available on the installed version).
- `ntotal`, `dim`, `add`, `search`, `get_vectors`, `remove`, `clear`
  implemented directly against the collection API. `search_near_id`
  delegates to the default (`get_vectors` → `search`) — Chroma has no
  server-side "find similar to id" call.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/backends.md` major update, `deferred-items.md` strikethrough.

---

### Task 1: `ChromaBackend` + filter translator

**Files:**
- Create: `src/soma/memory/backends/chroma.py`
- Create: `src/soma/memory/backends/chroma_filter.py`
- Create: `tests/test_memory/test_chroma_backend.py`
- Create: `tests/test_memory/test_chroma_filter.py`
- Modify: `pyproject.toml` — add `[project.optional-dependencies]`
  entry `chroma = ["chromadb>=0.5"]`

**`ChromaBackend` skeleton:**
```python
from __future__ import annotations

try:
    import chromadb
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False


class ChromaBackend:
    supports_filter_pushdown = True

    def __init__(
        self,
        *,
        path: str | None = None,
        client: Any = None,  # chromadb.ClientAPI when available
        collection_name: str = "soma",
        dim: int | None = None,
    ) -> None:
        if not _HAS_CHROMA:
            raise ImportError(
                'chromadb not installed; pip install "soma-memory[chroma]"'
            )
        if client is None:
            if path is None:
                raise ValueError("ChromaBackend requires path= or client=")
            client = chromadb.PersistentClient(path=path)
        self._client = client
        self._coll = client.get_or_create_collection(collection_name)
        self._dim = dim
        self._path = path
        self._name = collection_name

    @property
    def ntotal(self) -> int: ...
    @property
    def dim(self) -> int: ...  # lazily read first vector's length
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int, *, filter=None): ...
    def search_subset(self, query, k, subset_ids, *, filter=None): ...
    def get_vectors(self, ids: list[str]) -> np.ndarray: ...
    def remove(self, ids: list[str]) -> None: ...
    def clear(self) -> None: ...
    def snapshot(self, bundle_dir: Path) -> None: ...
    def restore(self, bundle_dir: Path) -> None: ...
```

**Filter translator:**
```python
def to_chroma_where(spec: dict | None) -> dict | None:
    """SOMA filter → Chroma where.

    Chroma's where uses MongoDB-style: {"field": {"$eq": "value"}}.
    SOMA's spec is flat: {"field": "value"} or {"field": {"$in": [...]}}.
    """
```

Supported ops: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.
Unsupported ops raise `FilterPushdownUnsupported` so MemoryLayer
falls back to Python pre-filter + `search_subset`.

**Step 1: Failing tests.**

`test_chroma_filter.py`:
```python
def test_eq_maps_to_chroma():
    assert to_chroma_where({"user_id": "alice"}) == {"user_id": {"$eq": "alice"}}

def test_in_passes_through():
    assert to_chroma_where({"tag": {"$in": ["a", "b"]}}) == {"tag": {"$in": ["a", "b"]}}

def test_unsupported_op_raises():
    with pytest.raises(FilterPushdownUnsupported):
        to_chroma_where({"x": {"$regex": "^a"}})

def test_empty_and_none():
    assert to_chroma_where(None) is None
    assert to_chroma_where({}) is None
```

`test_chroma_backend.py` (gated: `pytest.importorskip("chromadb")`):
```python
def test_add_and_search_round_trip(tmp_path):
    b = ChromaBackend(path=str(tmp_path/"chroma"), dim=8)
    b.add(["a","b","c"], np.random.randn(3,8).astype(np.float32))
    ids, scores = b.search(np.random.randn(8).astype(np.float32), k=2)
    assert len(ids) == 2
    assert set(ids) <= {"a","b","c"}

def test_ntotal_and_dim(tmp_path):
    b = ChromaBackend(path=str(tmp_path/"c"), dim=4)
    assert b.ntotal == 0
    b.add(["a"], np.ones((1,4), dtype=np.float32))
    assert b.ntotal == 1
    assert b.dim == 4

def test_remove_drops_ids(tmp_path): ...
def test_clear_resets(tmp_path): ...
def test_get_vectors_returns_stored(tmp_path): ...
def test_filter_pushdown_equality(tmp_path): ...
def test_filter_pushdown_in(tmp_path): ...
def test_filter_pushdown_unsupported_op_falls_back(tmp_path): ...
def test_snapshot_round_trip(tmp_path): ...
def test_import_error_when_chromadb_missing(monkeypatch, tmp_path): ...
def test_memory_layer_end_to_end(tmp_path): ...
    # Build a MemoryLayer with ChromaBackend, add some notes,
    # retrieve them, verify identity + recall.
```

**Step 2-4: TDD.**

**Step 5:** `git commit -m "feat(backends): ChromaBackend adapter + filter translator"`

---

### Task 2: Register in protocol contract suite

**Files:**
- Modify: `tests/test_memory/test_backend_protocol.py`

Add a factory entry for `ChromaBackend` so the parametrized contract
suite (all-adapters invariant tests) covers it. Follow the existing
LanceDB row as the template. Skip cleanly when `chromadb` is missing.

**Step 5:** `git commit -m "test(backends): ChromaBackend in protocol contract suite"`

---

### Task 3: Docs stub

**Files:**
- Modify: `docs/backends.md` — short Chroma section (plan Phase 27
  will do the deep section centrally; for now just add the
  minimal "when to use Chroma" paragraph + code example).

**Step 5:** `git commit -m "docs(backends): ChromaBackend stub"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
pytest tests/test_memory/test_chroma_backend.py tests/test_memory/test_chroma_filter.py tests/test_memory/test_backend_protocol.py -q
pytest tests/test_memory -q
```

Baseline post-Phase-25: 414 tests in `tests/test_memory`. Target +~12
new Phase 27 tests (4 filter + 8 backend + 1 protocol-suite row), 0
regressions.

**Gotchas to handle:**
- Chroma's collection creation is idempotent via `get_or_create_collection`
  but recreating after a `clear()` needs care — the Chroma Python
  API's `delete_collection` + recreate semantics changed between
  0.4.x and 0.5.x. Pin the minimum to `>=0.5` to avoid the old API.
- Chroma accepts `metadatas` on add(); make sure filter pushdown
  actually reaches the metadata dict — test round-trips a filtered
  search against a stored metadata field.
- Snapshot: if `client.persist()` isn't callable on the installed
  version (0.5.x removed it in favour of auto-persist), just copy
  the dir. Document the version-floor in the docstring.
