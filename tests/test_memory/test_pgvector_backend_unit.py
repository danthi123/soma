"""Unit tests for :class:`PgvectorBackend` with a mocked psycopg cursor.

These tests don't spin a real Postgres — they stub ``psycopg.connect``
+ ``register_vector`` and assert on the SQL strings the adapter emits.
They run as part of the default ``pytest`` invocation so any change
to the adapter's schema DDL or query shape is caught before the
(slow, Docker-gated) integration tests run.

The integration tests in ``test_pgvector_integration.py`` pick up the
same adapter against a real pgvector-enabled Postgres.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from soma.memory.backend import FilterPushdownUnsupported, VectorBackend

# ---------------------------------------------------------------------------
# Fixture: patches psycopg.connect + pgvector.psycopg.register_vector so a
# PgvectorBackend can be constructed without any Postgres running.
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal cursor stub — records every query ever executed."""

    def __init__(self, connection: _FakeConnection) -> None:
        self._conn = connection
        # What ``fetchone`` / ``fetchall`` should return for the next call.
        # The backend stages these before issuing the SELECT.
        self._next_row: tuple | None = None
        self._next_rows: list[tuple] = []
        self._next_copy_rows: list[tuple] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:  # noqa: D401 - context mgr protocol
        return None

    def execute(self, query: str, params: object = None) -> None:
        self._conn.executed.append((query, params))

    def executemany(self, query: str, seq_of_params: object) -> None:
        # Record each (query, params) pair separately so assertions can
        # inspect the full batch without special-casing executemany.
        seq_list = list(seq_of_params)
        for p in seq_list:
            self._conn.executed.append((query, p))
        self._conn.executemany_calls.append((query, seq_list))

    def fetchone(self) -> tuple | None:
        row = self._next_row
        self._next_row = None
        return row

    def fetchall(self) -> list[tuple]:
        rows = self._next_rows
        self._next_rows = []
        return rows

    def copy(self, query: str):  # noqa: ANN201 - context manager
        self._conn.executed.append((query, None))
        return _FakeCopy(self._next_copy_rows)


class _FakeCopy:
    """Stand-in for psycopg's ``Copy`` context manager."""

    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self._written: list[tuple] = []

    def __enter__(self) -> _FakeCopy:
        return self

    def __exit__(self, *exc: object) -> None:  # noqa: D401 - context mgr
        return None

    def rows(self):  # noqa: ANN201 - iterator
        """COPY TO STDOUT — yield the pre-staged rows."""
        yield from self._rows

    def write_row(self, row: tuple) -> None:
        """COPY FROM STDIN via structured row write."""
        self._written.append(row)

    def write(self, payload: bytes) -> None:
        """COPY FROM STDIN via raw line write — the adapter uses this
        path when restoring from the gzipped dump."""
        self._written.append(payload)


class _FakeConnection:
    """Minimal connection stub — keeps every executed query for
    assertion and hands out :class:`_FakeCursor` instances."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []
        self.executemany_calls: list[tuple[str, list]] = []
        self._cursor: _FakeCursor | None = None
        self.closed_flag = False
        self.committed = 0

    def cursor(self) -> _FakeCursor:
        if self._cursor is None:
            self._cursor = _FakeCursor(self)
        return self._cursor

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed_flag = True


@pytest.fixture
def fake_conn() -> _FakeConnection:
    return _FakeConnection()


@pytest.fixture
def make_backend(fake_conn: _FakeConnection):
    """Return a factory that constructs a PgvectorBackend wired to the
    ``fake_conn``."""
    from soma.memory.backends import pgvector as pg_mod

    def _factory(dim: int = 8, **kw: object):
        with (
            patch.object(pg_mod, "psycopg", create=True) as mock_psycopg,
            patch.object(pg_mod, "register_vector", create=True) as mock_rv,
            patch.object(pg_mod, "_HAS_PSYCOPG", True),
            patch.object(pg_mod, "_HAS_PGVECTOR", True),
        ):
            mock_psycopg.connect.return_value = fake_conn
            mock_rv.return_value = None
            b = pg_mod.PgvectorBackend(
                dsn="postgresql://test/db",
                dim=dim,
                **kw,  # type: ignore[arg-type]
            )
            return b

    return _factory


# ---------------------------------------------------------------------------
# Construction + schema DDL
# ---------------------------------------------------------------------------


def test_missing_deps_raises_import_error() -> None:
    from soma.memory.backends import pgvector as pg_mod

    with (
        patch.object(pg_mod, "_HAS_PSYCOPG", False),
        patch.object(pg_mod, "_HAS_PGVECTOR", True),
        pytest.raises(ImportError, match=r"soma\[pgvector\]"),
    ):
        pg_mod.PgvectorBackend(dsn="postgresql://x", dim=8)


def test_ensure_schema_emits_extension_table_indexes(
    make_backend, fake_conn: _FakeConnection
) -> None:
    make_backend()
    statements = [q.strip() for q, _ in fake_conn.executed]
    joined = " ; ".join(statements)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in joined
    assert "CREATE TABLE IF NOT EXISTS" in joined
    # Two indexes: ivfflat on vector column + GIN on metadata JSONB.
    assert "ivfflat" in joined
    assert "USING GIN" in joined or "USING gin" in joined


def test_schema_respects_table_name_kwarg(
    make_backend, fake_conn: _FakeConnection
) -> None:
    make_backend(table_name="tenant_alpha_vectors")
    joined = " ; ".join(q for q, _ in fake_conn.executed)
    assert "tenant_alpha_vectors" in joined


def test_register_vector_called_on_connection() -> None:
    """``register_vector(conn)`` MUST be called before any query that
    touches a ``vector`` column, or psycopg won't know how to
    serialise numpy arrays. Verify it fires right after connect."""
    from soma.memory.backends import pgvector as pg_mod

    fake = _FakeConnection()
    with (
        patch.object(pg_mod, "psycopg", create=True) as mock_psycopg,
        patch.object(pg_mod, "register_vector", create=True) as mock_rv,
        patch.object(pg_mod, "_HAS_PSYCOPG", True),
        patch.object(pg_mod, "_HAS_PGVECTOR", True),
    ):
        mock_psycopg.connect.return_value = fake
        mock_rv.return_value = None
        pg_mod.PgvectorBackend(dsn="postgresql://x", dim=4)
        mock_rv.assert_called_once_with(fake)


# ---------------------------------------------------------------------------
# Add / remove / clear
# ---------------------------------------------------------------------------


def test_add_issues_upsert_with_params(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    before_count = len(fake_conn.executed)
    ids = ["a", "b"]
    vectors = np.ones((2, 8), dtype=np.float32)
    b.add(ids, vectors)
    new_queries = fake_conn.executed[before_count:]
    # At least one INSERT ... ON CONFLICT ... DO UPDATE statement.
    inserts = [q for q, _ in new_queries if "INSERT" in q.upper()]
    assert inserts, f"no INSERT in {new_queries}"
    q = inserts[0].upper()
    assert "ON CONFLICT" in q
    assert "DO UPDATE" in q


def test_add_empty_is_noop(make_backend, fake_conn: _FakeConnection) -> None:
    b = make_backend()
    before = len(fake_conn.executed)
    b.add([], np.empty((0, 8), dtype=np.float32))
    assert len(fake_conn.executed) == before


def test_add_shape_mismatch_raises(make_backend) -> None:
    b = make_backend(dim=8)
    with pytest.raises(ValueError):
        b.add(["a"], np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        b.add(["a", "b"], np.ones((1, 8), dtype=np.float32))


def test_remove_emits_delete_any(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    b.remove(["a", "b", "c"])
    deletes = [q for q, _ in fake_conn.executed if q.upper().startswith("DELETE")]
    assert deletes
    assert "= ANY(%s)" in deletes[0] or "ANY(%s)" in deletes[0]


def test_remove_empty_is_noop(make_backend, fake_conn: _FakeConnection) -> None:
    b = make_backend()
    before = len(fake_conn.executed)
    b.remove([])
    assert len(fake_conn.executed) == before


def test_clear_truncates_or_deletes(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    before = len(fake_conn.executed)
    b.clear()
    after = fake_conn.executed[before:]
    qs = " ".join(q.upper() for q, _ in after)
    # Either TRUNCATE or DELETE FROM — both are acceptable.
    assert "TRUNCATE" in qs or "DELETE FROM" in qs


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_ntotal_emits_count(make_backend, fake_conn: _FakeConnection) -> None:
    b = make_backend()
    # Stage a row for ``cursor.fetchone``.
    fake_conn.cursor()._next_row = (7,)
    n = b.ntotal
    assert n == 7
    counts = [q for q, _ in fake_conn.executed if "COUNT(" in q.upper()]
    assert counts
    assert "COUNT(*)" in counts[-1].upper()


def test_search_emits_distance_operator_and_limit(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    # Stage ntotal + search result rows.
    cur = fake_conn.cursor()
    cur._next_row = (3,)  # for ntotal guard
    cur._next_rows = [("a", 0.99), ("b", 0.80)]
    hits = b.search(np.ones(8, dtype=np.float32), k=2)
    selects = [
        q for q, _ in fake_conn.executed if q.upper().strip().startswith("SELECT")
    ]
    # Find the SELECT that uses the <=> cosine-distance operator.
    score_selects = [q for q in selects if "<=>" in q]
    assert score_selects, f"no vector-distance SELECT in {selects}"
    q = score_selects[0]
    assert "ORDER BY" in q.upper()
    assert "LIMIT" in q.upper()
    # ``1 - (vector <=> %s)`` converts distance to cosine similarity.
    assert "1 -" in q or "1-" in q
    # Results come back as (id, score) with similarity in [-1, 1].
    assert hits == [("a", 0.99), ("b", 0.80)]


def test_search_empty_k_zero_returns_empty(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    cur = fake_conn.cursor()
    cur._next_row = (5,)
    assert b.search(np.ones(8, dtype=np.float32), k=0) == []


def test_search_applies_filter_pushdown(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    cur = fake_conn.cursor()
    cur._next_row = (3,)
    cur._next_rows = []
    b.search(
        np.ones(8, dtype=np.float32),
        k=5,
        where={"tag": "fiction"},
    )
    selects = [
        (q, p) for q, p in fake_conn.executed if q.upper().strip().startswith("SELECT")
    ]
    score_selects = [(q, p) for q, p in selects if "<=>" in q]
    assert score_selects
    q, params = score_selects[0]
    assert "WHERE" in q.upper()
    # JSONB containment clause from the translator.
    assert "metadata @>" in q
    # The containment payload is a JSON string — first bind param.
    assert isinstance(params, (tuple, list))


def test_search_unsupported_filter_raises(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    cur = fake_conn.cursor()
    cur._next_row = (1,)
    with pytest.raises(FilterPushdownUnsupported):
        b.search(
            np.ones(8, dtype=np.float32),
            k=3,
            where={"tag": {"$regex": "^a"}},
        )


def test_search_subset_includes_id_in_where(
    make_backend, fake_conn: _FakeConnection
) -> None:
    b = make_backend()
    cur = fake_conn.cursor()
    cur._next_rows = [("x", 0.7)]
    b.search_subset(
        np.ones(8, dtype=np.float32),
        ["x", "y", "z"],
        k=2,
    )
    selects = [q for q, _ in fake_conn.executed if q.upper().strip().startswith("SELECT")]
    score_selects = [q for q in selects if "<=>" in q]
    assert score_selects
    q = score_selects[0]
    assert "WHERE" in q.upper()
    assert "id = ANY" in q or "id IN" in q


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------


def test_snapshot_writes_sidecar_and_copy(
    make_backend, fake_conn: _FakeConnection, tmp_path: Path
) -> None:
    b = make_backend()
    cur = fake_conn.cursor()
    cur._next_copy_rows = [("a", "[1,2,3,4,5,6,7,8]", "{}")]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    b.snapshot(bundle)
    copy_qs = [q for q, _ in fake_conn.executed if "COPY" in q.upper()]
    assert copy_qs, "snapshot should issue a COPY ... TO STDOUT"
    assert "TO STDOUT" in copy_qs[0].upper()
    # Sidecar metadata
    import json

    meta = json.loads((bundle / "backend.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "pgvector"
    assert meta["dim"] == 8


def test_restore_replays_copy_from_stdin(
    make_backend, fake_conn: _FakeConnection, tmp_path: Path
) -> None:
    b = make_backend()
    # Produce the bundle via snapshot so the input shape matches.
    cur = fake_conn.cursor()
    cur._next_copy_rows = [("a", "[1,0,0,0,0,0,0,0]", "{}")]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    b.snapshot(bundle)
    fake_conn.executed.clear()
    # Now restore — must issue a COPY ... FROM STDIN.
    b.restore(bundle)
    copy_qs = [q for q, _ in fake_conn.executed if "COPY" in q.upper()]
    assert copy_qs
    assert "FROM STDIN" in copy_qs[0].upper()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_protocol_isinstance(make_backend) -> None:
    b = make_backend()
    assert isinstance(b, VectorBackend)


def test_supports_filter_pushdown_true(make_backend) -> None:
    b = make_backend()
    assert b.supports_filter_pushdown is True


def test_name_is_pgvector(make_backend) -> None:
    b = make_backend()
    assert b.name == "pgvector"
