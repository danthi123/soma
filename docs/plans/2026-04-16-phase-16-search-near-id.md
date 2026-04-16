# Phase 16: `search_near_id` for HTTP Backends

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Add an optional `search_near_id(node_id, k, exclude_self=True)` method to `VectorBackend` so `MemoryLayer.related()` can skip a round-trip when the backend can do "given this stored id, find its k nearest neighbours" server-side. Important for Qdrant-HTTP and LanceDB where the current code path is `get_vectors([node_id]) → search(vector, k+1)` — two round-trips for what Qdrant natively exposes as `recommend` and LanceDB as a self-join.

**Architecture:**
1. Extend the `VectorBackend` Protocol with `search_near_id` as a *default* method (not required). The default implementation calls `get_vectors` + `search` so any adapter that doesn't override keeps working.
2. Override in `QdrantBackend` to call Qdrant's `recommend` API.
3. Override in `LanceDBBackend` to run a single SQL-style `WHERE id = ?` + nearest-neighbour query.
4. `InProcBackend` uses the default (in-RAM; the round-trip is meaningless).
5. `MemoryLayer.related()` just calls `backend.search_near_id(...)`; no branching needed because the default preserves current behaviour.

**Tech Stack:** no new deps; Qdrant's recommend API is already in the pinned client.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/backends.md`.

---

### Task 1: Protocol extension + default implementation

**Files:**
- Modify: `src/soma/memory/backend.py`
- Modify: `tests/test_memory/test_backend_protocol.py`

**API:**
```python
class VectorBackend(Protocol):
    # ... existing methods ...

    def search_near_id(
        self, node_id: str, k: int, *, exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Return (id, score) pairs for the k nearest neighbours of
        ``node_id``'s stored vector. Default implementation uses
        :meth:`get_vectors` + :meth:`search` — adapters override to
        avoid the round-trip over HTTP backends.
        """
```

**Default implementation** (in a mixin or a helper used by concrete backends that don't override):
```python
def _default_search_near_id(
    backend, node_id, k, *, exclude_self=True,
):
    vectors = backend.get_vectors([node_id])
    if not vectors:
        return []
    query_k = k + 1 if exclude_self else k
    hits = backend.search(vectors[0], query_k)
    if exclude_self:
        hits = [(nid, s) for nid, s in hits if nid != node_id][:k]
    return hits
```

**Step 1: Write failing tests.**
```python
def test_search_near_id_default_excludes_self(shipped_backend):
    # Store 5 vectors including node "A"; call search_near_id("A", k=3).
    # Assert "A" is not in result; len == 3; ordered by score desc.

def test_search_near_id_default_include_self(shipped_backend):
    # exclude_self=False returns k hits including self at top.

def test_search_near_id_unknown_id_returns_empty(shipped_backend):
    # Unknown node_id returns [].
```

Use the parametrized `shipped_backend` fixture so all 3 adapters get covered.

**Step 5:** `git commit -m "feat(memory): VectorBackend.search_near_id default impl"`

---

### Task 2: QdrantBackend override using `recommend`

**Files:**
- Modify: `src/soma/memory/backends/qdrant.py`
- Modify: `src/soma/memory/backends/qdrant_filter.py` if needed

**Implementation:** Qdrant has `client.recommend(collection, positive=[point_id], limit=k, query_filter=...)`. Use it — it's exactly this operation server-side.

Handle the edge case where the point doesn't exist (Qdrant raises; catch and return []).

**Tests:** the parametrized suite auto-covers. Add one Qdrant-specific test that asserts we DON'T call `get_vectors` (patch-mock-based) to prove we took the fast path.

**Step 5:** `git commit -m "feat(qdrant): server-side search_near_id via recommend API"`

---

### Task 3: LanceDBBackend override

**Files:**
- Modify: `src/soma/memory/backends/lancedb.py`

**Implementation:** LanceDB supports a two-step but in-process pattern — `tbl.to_arrow(filter=f"id = '{id}'")` to get the vector, then `tbl.search(vector).limit(k+1)` on the same table. Technically still two operations but both are in-process and cheap. The win is more ergonomic (no Python round-trip with the vector serialized) than semantic.

Actually the biggest LanceDB win is if we use `tbl.search(vector_id=id)` — but LanceDB doesn't natively support this. Just implement the in-process fast path and move on.

**Step 5:** `git commit -m "feat(lancedb): search_near_id fast path"`

---

### Task 4: Route MemoryLayer.related() through it

**Files:**
- Modify: `src/soma/memory/api.py` — in `related()`, replace the current `_get_backend_vectors + backend.search` with `backend.search_near_id(node_id, k, exclude_self=True)`.

**Tests:** existing `related()` tests should pass unchanged.

**Step 5:** `git commit -m "feat(memory): related() uses backend.search_near_id"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
SOMA_EMBED_MODEL=stub pytest tests/test_memory -q
```

Baseline 388 tests pass in memory suite; target: 388 + new Phase 16 tests, 0 regressions.
