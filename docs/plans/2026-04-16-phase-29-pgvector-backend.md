# Phase 29: pgvector Backend Adapter

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Ship a pgvector-backed `VectorBackend` so Postgres shops
(many of our target users) can point SOMA at a cluster they already
operate. Fifth pluggable adapter alongside InProc, Qdrant (local +
HTTP), LanceDB, and Chroma. Covers the "we already have Postgres,
don't give us another database to run" pitch.

**Architecture:**
- `src/soma/memory/backends/pgvector.py` — `PgvectorBackend` using
  `psycopg` (v3, preferred — async-ready and faster than psycopg2).
  Falls back to `psycopg2` if v3 isn't installed (v2 is still the
  more widely deployed driver).
- `supports_filter_pushdown=True` via `pgvector_filter.to_pgvector_where`:
  translates our internal filter spec to `WHERE metadata @> '{...}'` for
  equality and `(metadata->>'field')::type op value` for comparisons.
  Uses JSONB ops so one `metadata JSONB` column handles all filtering.
- Schema:
  ```sql
  CREATE TABLE soma_vectors (
      id       TEXT PRIMARY KEY,
      vector   vector(%(dim)s),
      metadata JSONB DEFAULT '{}'::jsonb
  );
  CREATE INDEX ON soma_vectors USING ivfflat (vector vector_cosine_ops);
  CREATE INDEX ON soma_vectors USING GIN (metadata);
  ```
  Table name configurable via `table_name` kwarg (multi-tenant
  deployments may want per-tenant tables). Schema creation is
  idempotent (`CREATE TABLE IF NOT EXISTS ...`); dim mismatch
  raises explicit error (ALTER TABLE vector(n) is not painless).
- Snapshot: `COPY soma_vectors TO STDOUT` → gzip → bundle. Restore
  is the inverse. Cheaper than `pg_dump` for the single-table case.
- `search_near_id` delegates to the default (get_vectors + search)
  for now; the pgvector `<=>` operator works fine, no need for a
  dedicated recommend-style endpoint.

**Testing strategy:**
- **Unit tests** (always run): cursor mocked via `unittest.mock`.
  Asserts on SQL strings and parameter tuples. Covers filter
  translation, schema DDL, insert/delete/search SQL.
- **Integration tests** (gated): testcontainers-python pulls
  `pgvector/pgvector:pg16` image and runs a real PG. Gated behind
  `SOMA_PGVECTOR_INTEGRATION=1` + `@pytest.mark.slow_pgvector`
  marker + Docker availability. Mirrors the Phase 24 Qdrant gate
  pattern exactly.
- Integration tests skipped cleanly when Docker isn't available
  (testcontainers raises `DockerException`; test module catches
  and skips).

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/backends.md` major update, `deferred-items.md` strikethrough.

---

### Task 1: Filter translator + unit tests

**Files:**
- Create: `src/soma/memory/backends/pgvector_filter.py`
- Create: `tests/test_memory/test_pgvector_filter.py`

**API:**
```python
def to_pgvector_where(spec: dict | None) -> tuple[str, list[Any]] | None:
    """SOMA filter spec → (WHERE clause fragment, params).

    Returns None when the spec is empty — caller omits the WHERE
    clause entirely.

    WHERE fragments use %s placeholders for psycopg parameter binding.
    Example:
      spec = {"user_id": "alice", "score": {"$gte": 0.5}}
      → ("metadata @> %s AND (metadata->>'score')::float >= %s",
         ['{"user_id": "alice"}', 0.5])
    """
```

Supported ops: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.
Unsupported ops raise `FilterPushdownUnsupported`.

**Step 1: Failing tests.**
```python
def test_eq_uses_jsonb_containment():
    clause, params = to_pgvector_where({"user_id": "alice"})
    assert clause == "metadata @> %s::jsonb"
    assert params == ['{"user_id": "alice"}']

def test_gte_casts_and_compares():
    clause, params = to_pgvector_where({"score": {"$gte": 0.5}})
    assert "(metadata->>'score')::float" in clause
    assert ">=" in clause
    assert params == [0.5]

def test_in_expands_to_array_any():
    clause, params = to_pgvector_where({"tag": {"$in": ["a", "b"]}})
    assert "ANY" in clause or "IN " in clause
    assert set(params) == {"a", "b"} or params == [["a", "b"]]
    # Accept either ANY(%s) or IN (%s, %s) — implementer choice.

def test_multiple_fields_joined_by_and():
    clause, params = to_pgvector_where({"user_id": "alice", "year": 2026})
    assert " AND " in clause
    assert len(params) == 2

def test_unsupported_op_raises():
    with pytest.raises(FilterPushdownUnsupported):
        to_pgvector_where({"x": {"$regex": "^a"}})

def test_empty_and_none():
    assert to_pgvector_where(None) is None
    assert to_pgvector_where({}) is None

def test_string_escaping_safe():
    # SQL injection safety: params, never interpolation.
    clause, params = to_pgvector_where({"name": "'; DROP TABLE x; --"})
    assert "DROP" not in clause
    assert params == ['{"name": "\'; DROP TABLE x; --"}']
```

**Step 5:** `git commit -m "feat(pgvector): filter translator for JSONB metadata"`

---

### Task 2: `PgvectorBackend` adapter + unit tests

**Files:**
- Create: `src/soma/memory/backends/pgvector.py`
- Create: `tests/test_memory/test_pgvector_backend_unit.py`
- Modify: `pyproject.toml` — add `pgvector = ["psycopg[binary]>=3", "pgvector>=0.3"]`
  under `[project.optional-dependencies]`.

**Adapter skeleton:**
```python
from __future__ import annotations

try:
    import psycopg
    _HAS_PSYCOPG = True
except ImportError:
    _HAS_PSYCOPG = False

try:
    from pgvector.psycopg import register_vector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False


class PgvectorBackend:
    supports_filter_pushdown = True

    def __init__(
        self,
        *,
        dsn: str,
        table_name: str = "soma_vectors",
        dim: int | None = None,
        connect_kwargs: dict | None = None,
    ) -> None:
        if not (_HAS_PSYCOPG and _HAS_PGVECTOR):
            raise ImportError(
                'pgvector deps not installed; '
                'pip install "soma-memory[pgvector]"'
            )
        self._dsn = dsn
        self._table = table_name
        self._dim = dim
        self._conn = psycopg.connect(dsn, **(connect_kwargs or {}))
        register_vector(self._conn)
        self._ensure_schema()

    # ... protocol methods
```

Unit tests mock the connection + cursor via `unittest.mock`. Assert:
- `_ensure_schema` emits `CREATE EXTENSION IF NOT EXISTS vector`,
  `CREATE TABLE IF NOT EXISTS`, two `CREATE INDEX IF NOT EXISTS`.
- `add(ids, vectors, metadatas=None)` emits one `INSERT ... ON
  CONFLICT (id) DO UPDATE` with the right parameter tuple.
- `search(query, k)` emits `SELECT id, 1 - (vector <=> %s) AS score
  FROM {table} ORDER BY vector <=> %s LIMIT %s`, returns (ids, scores).
- `search(query, k, filter=...)` includes the WHERE fragment from
  the translator.
- `remove(ids)` emits `DELETE FROM {table} WHERE id = ANY(%s)`.
- `clear()` emits `TRUNCATE {table}` (or `DELETE FROM {table}` if
  operators might deny TRUNCATE — implementer's call, document it).
- `ntotal` emits `SELECT COUNT(*) FROM {table}`.
- Snapshot: emits `COPY {table} TO STDOUT` — mock the `copy()`
  context manager.

**Step 5:** `git commit -m "feat(pgvector): PgvectorBackend adapter"`

---

### Task 3: Integration tests (gated)

**Files:**
- Create: `tests/test_memory/test_pgvector_integration.py`

**Pattern** (mirror Phase 24):
```python
import os
import pytest

try:
    from testcontainers.postgres import PostgresContainer
    _HAS_TC = True
except ImportError:
    _HAS_TC = False

pytestmark = [
    pytest.mark.skipif(not _HAS_TC, reason="testcontainers not installed"),
    pytest.mark.skipif(
        os.environ.get("SOMA_PGVECTOR_INTEGRATION") != "1",
        reason="set SOMA_PGVECTOR_INTEGRATION=1 to run live pgvector",
    ),
    pytest.mark.slow_pgvector,
]

@pytest.fixture
def pgvector_container():
    pg = PostgresContainer(image="pgvector/pgvector:pg16")
    pg.start()
    yield pg.get_connection_url()
    pg.stop()

def test_add_search_round_trip(pgvector_container):
    backend = PgvectorBackend(dsn=pgvector_container, dim=8)
    ids = ["a", "b", "c"]
    vectors = np.random.randn(3, 8).astype(np.float32)
    backend.add(ids, vectors)
    got_ids, _ = backend.search(vectors[0], k=2)
    assert got_ids[0] == "a"  # query == vector[0], top-1 should be 'a'

def test_filter_pushdown_equality(pgvector_container): ...
def test_filter_pushdown_in(pgvector_container): ...
def test_snapshot_restore_round_trip(pgvector_container, tmp_path): ...
def test_schema_is_idempotent(pgvector_container): ...
def test_memory_layer_end_to_end(pgvector_container): ...
```

Register `slow_pgvector` marker in `pyproject.toml`
`[tool.pytest.ini_options].markers` (next to `slow_qdrant`).

Also add a new `[project.optional-dependencies]` entry:
`pgvector-test = ["testcontainers[postgresql]>=4"]` mirroring the
`qdrant-test` pattern.

**Step 5:** `git commit -m "test(pgvector): gated integration tests via testcontainers"`

---

### Task 4: Register in protocol contract suite

**Files:**
- Modify: `tests/test_memory/test_backend_protocol.py`

Add a PgvectorBackend factory entry. Gate with `testcontainers` +
env guard — when the integration gate is off, the row is skipped;
when it's on, the backend gets the full parametrized contract
invariants suite applied.

**Step 5:** `git commit -m "test(pgvector): register in protocol contract suite"`

---

### Task 5: Docs stub

**Files:**
- Modify: `docs/backends.md` — short when-to-pick section + minimal
  example + integration-test enablement guide.

**Step 5:** `git commit -m "docs(backends): PgvectorBackend stub"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
pytest tests/test_memory/test_pgvector_filter.py tests/test_memory/test_pgvector_backend_unit.py -q
# Integration (only if Docker is up):
SOMA_PGVECTOR_INTEGRATION=1 pytest tests/test_memory/test_pgvector_integration.py -q
pytest tests/test_memory -q    # default run; integration tests skipped cleanly
```

Baseline post-Phase-27: 466 passed, 16 skipped in `tests/test_memory`.
Target: +~20 unit tests always-on (7 filter + ~13 adapter) + 6 gated
integration tests. Default `pytest` run stays +20, not +26.

**Gotchas:**
- pgvector's distance operator changed naming between 0.4 and 0.5
  — `<=>` is cosine distance; `1 - (vector <=> %s)` converts to
  similarity (so top-k ORDER BY score DESC matches our internal
  contract where higher = more similar).
- `register_vector(conn)` must be called before ANY query that
  touches a `vector` column, or psycopg won't know how to serialize
  numpy arrays. Do it right after `psycopg.connect`.
- `connect_timeout` should default to something sane (5–10s) so a
  dead Postgres doesn't hang test startup.
- pgvector image tag: `pgvector/pgvector:pg16` is the current
  stable; the `pg15` variant also works if CI pins it.
