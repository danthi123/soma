"""Integration tests for :class:`PgvectorBackend` against a real
pgvector-enabled Postgres container.

Phase 29 — gated slow matrix. The default ``pytest`` run skips these
entirely. To exercise them locally::

    pip install -e ".[pgvector,pgvector-test]"
    SOMA_PGVECTOR_INTEGRATION=1 \\
      pytest tests/test_memory/test_pgvector_integration.py -q

Requires a working Docker daemon (Docker Desktop on Windows/macOS,
``dockerd`` on Linux). The fixture pulls ``pgvector/pgvector:pg16``
and runs a real Postgres with the ``vector`` extension available —
no mocked cursors here.

Mirrors the Phase 24 Qdrant gate pattern exactly:

- Module-level ``pytestmark`` skip when ``testcontainers`` isn't
  importable, psycopg isn't importable, or the env gate isn't set.
- ``slow_pgvector`` marker applied when all preconditions are met.
- Tests themselves assert against live SQL rather than mock output.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guards — module must be importable even when testcontainers and/or
# psycopg are not installed. The module-level skip then hides every test at
# collection time so the default ``pytest`` run stays green.
# ---------------------------------------------------------------------------
try:
    from testcontainers.postgres import PostgresContainer  # type: ignore[import-not-found]

    _HAS_TC = True
except ImportError:  # pragma: no cover - env-dependent
    PostgresContainer = None  # type: ignore[assignment,misc]
    _HAS_TC = False

try:
    import psycopg  # noqa: F401  # type: ignore[import-not-found]

    _HAS_PSYCOPG = True
except ImportError:  # pragma: no cover - env-dependent
    _HAS_PSYCOPG = False

try:
    from pgvector.psycopg import register_vector  # noqa: F401

    _HAS_PGVECTOR = True
except ImportError:  # pragma: no cover - env-dependent
    _HAS_PGVECTOR = False


_GATE_ENV = "SOMA_PGVECTOR_INTEGRATION"


def _gate_active() -> bool:
    return os.environ.get(_GATE_ENV) == "1"


# Module-level guard. If ANY dep is missing or the gate env isn't set,
# skip collection entirely so this file is a no-op in default runs.
if not _HAS_TC:
    pytestmark = pytest.mark.skip(reason="testcontainers not installed")
elif not _HAS_PSYCOPG:
    pytestmark = pytest.mark.skip(reason="psycopg not installed")
elif not _HAS_PGVECTOR:
    pytestmark = pytest.mark.skip(reason="pgvector not installed")
elif not _gate_active():
    pytestmark = pytest.mark.skip(
        reason=f"set {_GATE_ENV}=1 to run the slow pgvector integration matrix"
    )
else:
    pytestmark = pytest.mark.slow_pgvector


_DIM = 8


def _rand_vectors(n: int, dim: int = _DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def pgvector_container() -> Iterator[str]:
    """Spin a pgvector-enabled Postgres container, yield its DSN, stop it.

    Uses the official ``pgvector/pgvector:pg16`` image, which has the
    ``vector`` extension pre-installed — the adapter's
    ``CREATE EXTENSION IF NOT EXISTS vector`` will succeed without
    any superuser dance.
    """
    pg = PostgresContainer(image="pgvector/pgvector:pg16")
    pg.start()
    try:
        # testcontainers returns a SQLAlchemy-style URL
        # (``postgresql+psycopg2://...``); strip the driver suffix so
        # psycopg v3 is happy.
        url = pg.get_connection_url()
        url = url.replace("postgresql+psycopg2://", "postgresql://")
        url = url.replace("postgresql+psycopg://", "postgresql://")
        yield url
    finally:
        pg.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_add_search_round_trip(pgvector_container: str) -> None:
    """Basic smoke: add 3 vectors, query by the first, expect self-match
    at position 0 with score near 1.0."""
    from soma.memory.backends.pgvector import PgvectorBackend

    backend = PgvectorBackend(
        dsn=pgvector_container,
        dim=_DIM,
        table_name=f"smoke_{uuid.uuid4().hex[:8]}",
    )
    try:
        ids = ["a", "b", "c"]
        vectors = _rand_vectors(3, seed=1)
        backend.add(ids, vectors)
        assert backend.ntotal == 3
        hits = backend.search(vectors[0], k=2)
        assert len(hits) == 2
        assert hits[0][0] == "a"
        assert hits[0][1] > 0.99
    finally:
        backend.close()


def test_filter_pushdown_equality(pgvector_container: str) -> None:
    """JSONB containment (``metadata @> ...``) selects only rows whose
    metadata dict matches the equality filter."""
    from soma.memory.backends.pgvector import PgvectorBackend

    backend = PgvectorBackend(
        dsn=pgvector_container,
        dim=_DIM,
        table_name=f"eq_{uuid.uuid4().hex[:8]}",
    )
    try:
        ids = ["a", "b", "c"]
        vectors = _rand_vectors(3, seed=2)
        backend.add(
            ids,
            vectors,
            metadatas=[{"tag": "x"}, {"tag": "y"}, {"tag": "x"}],
        )
        hits = backend.search(vectors[0], k=5, where={"tag": "x"})
        returned = {nid for nid, _ in hits}
        assert returned == {"a", "c"}
    finally:
        backend.close()


def test_filter_pushdown_in(pgvector_container: str) -> None:
    """``$in`` → ``metadata->>'field' = ANY(%s)``; selects the union."""
    from soma.memory.backends.pgvector import PgvectorBackend

    backend = PgvectorBackend(
        dsn=pgvector_container,
        dim=_DIM,
        table_name=f"in_{uuid.uuid4().hex[:8]}",
    )
    try:
        ids = ["a", "b", "c", "d"]
        vectors = _rand_vectors(4, seed=3)
        backend.add(
            ids,
            vectors,
            metadatas=[
                {"tag": "x"},
                {"tag": "y"},
                {"tag": "z"},
                {"tag": "x"},
            ],
        )
        hits = backend.search(
            vectors[0], k=10, where={"tag": {"$in": ["x", "y"]}}
        )
        returned = {nid for nid, _ in hits}
        assert returned == {"a", "b", "d"}
    finally:
        backend.close()


def test_snapshot_restore_round_trip(
    pgvector_container: str, tmp_path: Path
) -> None:
    """``snapshot`` writes a gzipped COPY dump; ``restore`` re-hydrates
    a fresh backend so ``search`` returns the pre-snapshot top-1."""
    from soma.memory.backends.pgvector import PgvectorBackend

    table = f"snap_{uuid.uuid4().hex[:8]}"
    backend = PgvectorBackend(
        dsn=pgvector_container,
        dim=_DIM,
        table_name=table,
    )
    ids = [f"id-{i}" for i in range(5)]
    vectors = _rand_vectors(5, seed=42)
    try:
        backend.add(ids, vectors)
        assert backend.ntotal == 5
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        backend.snapshot(bundle)
        # Wipe the table out-of-band so restore has real work to do.
        backend.clear()
        assert backend.ntotal == 0
        backend.restore(bundle)
        assert backend.ntotal == 5
        hits = backend.search(vectors[0], k=1)
        assert hits and hits[0][0] == "id-0"
    finally:
        backend.close()


def test_schema_is_idempotent(pgvector_container: str) -> None:
    """Repeated ``open()`` calls must not error and must preserve rows.

    The adapter's ``_ensure_schema`` uses ``IF NOT EXISTS`` so a
    second open against the same table is a no-op on DDL and must
    leave existing rows alone.
    """
    from soma.memory.backends.pgvector import PgvectorBackend

    table = f"idem_{uuid.uuid4().hex[:8]}"
    backend = PgvectorBackend(
        dsn=pgvector_container,
        dim=_DIM,
        table_name=table,
    )
    try:
        backend.add(["a"], _rand_vectors(1, seed=99))
        assert backend.ntotal == 1
        # Re-running _ensure_schema (via open) must not drop rows.
        backend.open()
        assert backend.ntotal == 1
    finally:
        backend.close()


def test_memory_layer_end_to_end(
    pgvector_container: str, tmp_path: Path
) -> None:
    """Store, retrieve, forget, save, load — the operator contract for
    a pgvector-backed MemoryLayer."""
    import torch

    from soma.memory.api import MemoryLayer
    from soma.memory.backends.pgvector import PgvectorBackend

    dim = 16

    def _embed(text: str) -> torch.Tensor:
        h = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        vec: Any = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return torch.from_numpy(vec)

    table = f"ml_{uuid.uuid4().hex[:8]}"
    backend = PgvectorBackend(
        dsn=pgvector_container,
        dim=dim,
        table_name=table,
    )
    mem = MemoryLayer(
        embed_fn=_embed,
        embed_dim=dim,
        bundle_path=tmp_path / "bundle",
        backend=backend,
    )
    try:
        ida = mem.store("apples are red")
        idb = mem.store("bananas are yellow")
        idc = mem.store("cherries are red")
        assert len(mem) == 3

        hits = mem.retrieve("apples are red", k=3)
        assert hits[0].node_id == ida

        assert mem.forget(idb) is True
        remaining = {h.node_id for h in mem.retrieve("bananas", k=5)}
        assert idb not in remaining
        assert idc in {ida, idc}
        assert len(mem) == 2
    finally:
        mem.close()
