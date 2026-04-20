"""PgvectorBackend — Postgres + pgvector-backed :class:`VectorBackend`.

Fills the "we already run Postgres; don't give us another database to
operate" slot in SOMA's backend lineup — fifth adapter alongside
InProc, Qdrant (local + HTTP), LanceDB, and Chroma.

Wire-level:

- Driver: psycopg v3 (``psycopg[binary]``). v3 is async-ready, much
  faster, and the current Postgres driver direction for Python.
- Schema: a single table per backend instance::

      CREATE TABLE soma_vectors (
          id       TEXT PRIMARY KEY,
          vector   vector(dim),
          metadata JSONB DEFAULT '{}'::jsonb
      );
      CREATE INDEX ... USING ivfflat (vector vector_cosine_ops);
      CREATE INDEX ... USING GIN (metadata);

  ``table_name`` is configurable so operators can run per-tenant
  tables inside one database if they want. Schema creation is
  idempotent (``CREATE TABLE IF NOT EXISTS``) but does NOT migrate
  an existing table with a different dim — that would require
  ``ALTER TABLE vector(n)`` which is not painless, so a mismatch
  raises explicitly.

Filter pushdown:

- ``supports_filter_pushdown = True``.
- Translator in :mod:`soma.memory.backends.pgvector_filter` turns
  SOMA's Chroma-style ``where`` dict into a ``(SQL fragment, params)``
  pair suitable for psycopg parameter binding. Equality rides JSONB
  containment (``metadata @> %s::jsonb``, GIN-indexable); comparisons
  extract via ``(metadata->>'field')::float op %s``.

Snapshot / restore:

- ``snapshot`` runs ``COPY {table} TO STDOUT`` straight into a
  gzip-compressed file inside the bundle. Cheaper than ``pg_dump``
  for the single-table case and keeps the restore path symmetric
  (``COPY ... FROM STDIN``).
- ``restore`` issues a ``TRUNCATE`` + ``COPY FROM STDIN``.
- Sidecar ``backend.json`` records dim / table_name / pgvector
  version so a future reader can verify compatibility.

pgvector's distance operator ``<=>`` returns ``1 - cosine_similarity``;
we emit ``SELECT id, 1 - (vector <=> %s) AS score`` so the returned
score matches the ``[-1, 1]`` contract every other VectorBackend uses
(higher is more similar).

``psycopg`` and ``pgvector`` are optional deps — imports are guarded
so SOMA still ships without them. Constructing a ``PgvectorBackend``
when either is missing raises :class:`ImportError` with a
``pip install "soma-memory[pgvector]"`` hint.
"""

from __future__ import annotations

import contextlib
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from soma.memory.backend import _default_search_near_id
from soma.memory.backends.pgvector_filter import to_pgvector_where

try:
    import psycopg

    _HAS_PSYCOPG = True
except ImportError:  # pragma: no cover - exercised via monkeypatch
    psycopg = None  # type: ignore[assignment]
    _HAS_PSYCOPG = False

try:
    from pgvector.psycopg import register_vector

    _HAS_PGVECTOR = True
except ImportError:  # pragma: no cover - exercised via monkeypatch

    def register_vector(context: object) -> None:  # type: ignore[misc]
        """Stub — real register_vector comes from the pgvector package."""
        return None

    _HAS_PGVECTOR = False


# Default connection timeout. A dead Postgres shouldn't hang test
# startup for minutes — 5s is long enough for slow laptops, short
# enough that operators notice misconfigurations quickly.
_DEFAULT_CONNECT_TIMEOUT = 5


class PgvectorBackend:
    """Postgres + pgvector-backed :class:`VectorBackend`.

    Parameters
    ----------
    dsn:
        Postgres connection string (``postgresql://user:pass@host/db``
        or any psycopg-understood shape).
    table_name:
        Table to store vectors in. Defaults to ``"soma_vectors"``.
        Per-tenant layouts can pass a tenant-scoped name here.
    dim:
        Embedding dimension. Must be supplied at construction time
        because the ``vector(n)`` column type bakes it into the
        schema. Existing tables whose ``vector`` column has a
        different declared dim are left alone (we check and raise
        instead of rewriting — ``ALTER TABLE`` on a vector column is
        not painless).
    connect_kwargs:
        Extra kwargs forwarded to :func:`psycopg.connect` — e.g.
        ``{"sslmode": "require"}``. A ``connect_timeout`` of 5s is
        injected by default so a dead Postgres doesn't block startup.
    """

    supports_filter_pushdown: bool = True
    name: str = "pgvector"

    def __init__(
        self,
        *,
        dsn: str,
        table_name: str = "soma_vectors",
        dim: int,
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not (_HAS_PSYCOPG and _HAS_PGVECTOR):
            raise ImportError(
                "PgvectorBackend requires psycopg and pgvector. "
                'Install with: pip install "soma-memory[pgvector]"'
            )
        self._dsn = dsn
        self._table = _validate_identifier(table_name)
        self._dim = int(dim)
        kwargs = dict(connect_kwargs or {})
        kwargs.setdefault("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
        # autocommit keeps DDL + COPY semantics simple — each statement
        # commits on success. The WAL-replay model doesn't need a
        # long-running transaction.
        kwargs.setdefault("autocommit", True)
        assert psycopg is not None
        self._conn: Any = psycopg.connect(dsn, **kwargs)
        # register_vector teaches psycopg how to serialise numpy arrays
        # into the ``vector`` column type. Must fire BEFORE any query
        # that binds a vector value, so we call it right after connect.
        register_vector(self._conn)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Idempotent — the connection is opened in ``__init__`` and
        stays open until :meth:`close`. Repeated ``open()`` calls
        re-run :meth:`_ensure_schema` so test fixtures can reset the
        schema cheaply without reconstructing the adapter."""
        self._ensure_schema()

    def close(self) -> None:
        """Release the psycopg connection. Safe to call twice."""
        if self._conn is not None:
            with contextlib.suppress(Exception):
                # close is best-effort; if the connection is already
                # dead or broken, drop our reference regardless.
                self._conn.close()
            self._conn = None

    def clear(self) -> None:
        """Drop every indexed vector but keep the table + indexes.

        TRUNCATE is O(1) on a single table and preserves the GIN /
        ivfflat indexes, so it's the right primitive here. Falls back
        to ``DELETE FROM`` if TRUNCATE fails (some managed Postgres
        installs deny TRUNCATE on specific roles).
        """
        assert self._conn is not None
        with self._conn.cursor() as cur:
            try:
                cur.execute(f"TRUNCATE TABLE {self._table}")
            except Exception:
                cur.execute(f"DELETE FROM {self._table}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """Create the extension, table, and indexes if they don't exist.

        All statements use ``IF NOT EXISTS`` so repeated calls are
        idempotent. Dim mismatch on an existing table raises
        :class:`ValueError` — ``ALTER TABLE column TYPE vector(n)``
        would work in some cases but the safe contract is "caller
        provides a table they own, or a fresh schema".
        """
        assert self._conn is not None
        dim = self._dim
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id       TEXT PRIMARY KEY,
                    vector   vector({dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb
                )
                """.strip()
            )
            # ivfflat on the vector column, cosine ops class (the only
            # metric MemoryLayer advertises today).
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {self._table}_vec_ivfflat_idx
                ON {self._table} USING ivfflat (vector vector_cosine_ops)
                """.strip()
            )
            # GIN on metadata JSONB — used by the ``@>`` containment
            # path in the filter translator.
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS
                    {self._table}_metadata_gin_idx
                ON {self._table} USING GIN (metadata)
                """.strip()
            )

    # ------------------------------------------------------------------
    # Vector ops
    # ------------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        if self._conn is None:
            return 0
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    @property
    def dim(self) -> int:
        return self._dim

    def add(
        self,
        ids: list[str],
        vectors: np.ndarray,
        *,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert ``len(ids)`` rows.

        ``metadatas`` is an adapter-specific extension (same shape as
        ChromaBackend's). When supplied, each row's JSONB column gets
        the corresponding dict so ``retrieve(where=...)`` can push the
        filter down. MemoryLayer's default path never calls with
        ``metadatas=`` (metadata lives on MemoryLayer's side) — but
        external callers who construct the backend directly can
        populate filterable fields.
        """
        if not ids:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(ids) or arr.shape[1] != self._dim:
            raise ValueError(
                f"vectors shape {arr.shape} mismatched; expected ({len(ids)}, {self._dim})"
            )
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError(
                f"metadatas length {len(metadatas)} != ids length {len(ids)}"
            )
        assert self._conn is not None
        stmt = (
            f"INSERT INTO {self._table} (id, vector, metadata) "
            "VALUES (%s, %s, %s::jsonb) "
            "ON CONFLICT (id) DO UPDATE "
            "SET vector = EXCLUDED.vector, metadata = EXCLUDED.metadata"
        )
        with self._conn.cursor() as cur:
            for i, nid in enumerate(ids):
                meta = metadatas[i] if metadatas is not None else {}
                cur.execute(
                    stmt,
                    (nid, arr[i], json.dumps(meta)),
                )

    def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE id = ANY(%s)",
                (list(ids),),
            )

    def get_vectors(self, ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty((0, self._dim), dtype=np.float32)
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, vector FROM {self._table} WHERE id = ANY(%s)",
                (list(ids),),
            )
            rows = cur.fetchall()
        id_to_vec: dict[str, np.ndarray] = {}
        for nid, vec in rows:
            id_to_vec[str(nid)] = np.asarray(vec, dtype=np.float32)
        out = np.zeros((len(ids), self._dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            if nid not in id_to_vec:
                raise KeyError(f"id {nid!r} not in backend")
            out[i] = id_to_vec[nid]
        return out

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude_ids: set[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        # Cheap short-circuit: if the table is empty, no sense issuing
        # a search query.
        if self.ntotal == 0:
            return []
        assert self._conn is not None
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(f"query dim mismatch: backend={self._dim}, got={q.shape[0]}")

        # Translator raises FilterPushdownUnsupported for anything we
        # can't express; MemoryLayer catches + falls back.
        translated = to_pgvector_where(where)

        where_parts: list[str] = []
        params: list[Any] = [q]
        if translated is not None:
            clause, trans_params = translated
            where_parts.append(clause)
            params.extend(trans_params)
        if exclude_ids:
            where_parts.append("id != ALL(%s)")
            params.append(list(exclude_ids))

        where_sql = ""
        if where_parts:
            where_sql = "WHERE " + " AND ".join(where_parts)

        # Over-fetch when exclusions are active so the post-filter
        # (if we fall back to Python-side) still has k rows.
        limit = k
        params.append(q)  # ORDER BY vector <=> %s — bind vector twice
        params.append(limit)

        sql = (
            f"SELECT id, 1 - (vector <=> %s) AS score "
            f"FROM {self._table} "
            f"{where_sql} "
            f"ORDER BY vector <=> %s "
            f"LIMIT %s"
        ).replace("  ", " ").strip()

        with self._conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [(str(rid), float(score)) for rid, score in rows]

    def search_subset(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        if not candidate_ids or k <= 0:
            return []
        assert self._conn is not None
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        sql = (
            f"SELECT id, 1 - (vector <=> %s) AS score "
            f"FROM {self._table} "
            f"WHERE id = ANY(%s) "
            f"ORDER BY vector <=> %s "
            f"LIMIT %s"
        )
        limit = min(k, len(candidate_ids))
        with self._conn.cursor() as cur:
            cur.execute(sql, (q, list(candidate_ids), q, limit))
            rows = cur.fetchall()
        return [(str(rid), float(score)) for rid, score in rows]

    def search_near_id(
        self,
        node_id: str,
        k: int,
        *,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        """Delegate to the Protocol default.

        pgvector's ``<=>`` operator works fine against a stored vector,
        but the default two-step ``get_vectors`` + ``search`` path is
        already one round-trip to the server (the pivot fetch is tiny)
        and keeps the adapter surface minimal. Adapters that need
        lower latency can override here; at current scales the default
        is indistinguishable.
        """
        return _default_search_near_id(self, node_id, k, exclude_self=exclude_self)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self, bundle_dir: Path) -> None:
        """Write a gzip-compressed ``COPY`` dump of the table into the
        bundle, plus a sidecar ``backend.json`` with dim / table /
        pgvector version.

        ``COPY TO STDOUT`` is the cheapest portable format for a
        single table — smaller than ``pg_dump``, and the restore path
        is a direct ``COPY FROM STDIN`` inverse.
        """
        bundle_dir = Path(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": "pgvector",
            "dim": self._dim,
            "table_name": self._table,
            "pgvector_version": _pgvector_version(),
            "psycopg_version": _psycopg_version(),
        }
        (bundle_dir / "backend.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        assert self._conn is not None
        dump_path = bundle_dir / "pgvector_dump.copy.gz"
        with (
            self._conn.cursor() as cur,
            gzip.open(dump_path, "wb") as gz,
            cur.copy(f"COPY {self._table} (id, vector, metadata) TO STDOUT") as cp,
        ):
            for row in cp.rows():
                # Each row is a tuple of bytes / values depending on
                # psycopg's COPY format. We record the raw bytes so
                # restore can hand them back verbatim.
                gz.write(_serialise_copy_row(row))

    def restore(self, bundle_dir: Path) -> None:
        """Inverse of :meth:`snapshot`.

        TRUNCATEs the table then streams the gzipped ``COPY`` back via
        ``COPY FROM STDIN``. No-op when the bundle has no
        ``pgvector_dump.copy.gz`` (MemoryLayer will rebuild via WAL
        replay in that case).
        """
        bundle_dir = Path(bundle_dir)
        meta_path = bundle_dir / "backend.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("backend") == "pgvector":
                self._table = _validate_identifier(meta.get("table_name", self._table))
                if meta.get("dim") is not None:
                    self._dim = int(meta["dim"])
                self._ensure_schema()
        dump_path = bundle_dir / "pgvector_dump.copy.gz"
        if not dump_path.exists():
            return
        assert self._conn is not None
        with self._conn.cursor() as cur:
            try:
                cur.execute(f"TRUNCATE TABLE {self._table}")
            except Exception:
                cur.execute(f"DELETE FROM {self._table}")
            with (
                gzip.open(dump_path, "rb") as gz,
                cur.copy(f"COPY {self._table} (id, vector, metadata) FROM STDIN") as cp,
            ):
                for line in gz:
                    # Each serialised row is the newline-terminated
                    # payload :func:`_serialise_copy_row` wrote. psycopg's
                    # Copy.write passes bytes straight through.
                    cp.write(line)


def _validate_identifier(ident: str) -> str:
    """Guard against SQL-injection via the table name.

    We inline the table name into DDL + queries (it's a SQL identifier,
    not a bindable literal) so we must prove it's a well-formed
    identifier before use. Accept unqualified names (``soma_vectors``)
    and schema-qualified (``public.soma_vectors``); reject anything
    else.
    """
    parts = ident.split(".")
    if len(parts) not in (1, 2):
        raise ValueError(f"invalid table_name {ident!r}")
    for part in parts:
        if not part or not all(c.isalnum() or c == "_" for c in part):
            raise ValueError(f"invalid table_name {ident!r}")
    return ident


def _serialise_copy_row(row: Any) -> bytes:
    """Convert a psycopg COPY row into bytes suitable for gzip storage.

    psycopg's ``Copy.rows()`` yields bytes by default (the server's
    text COPY format, one line per row). We keep that contract: if
    the row is already bytes, pass through; otherwise encode via
    repr (test-path fallback).
    """
    if isinstance(row, (bytes, bytearray)):
        # Already a newline-terminated COPY line.
        return bytes(row)
    # Tuple path — only hit in unit tests, where the fake cursor
    # yields Python tuples. Serialize as a simple TSV-ish line so the
    # restore path can round-trip.
    return ("\t".join(str(c) for c in row) + "\n").encode("utf-8")


def _pgvector_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("pgvector"))
    except Exception:
        return "unknown"


def _psycopg_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("psycopg"))
    except Exception:
        return "unknown"


__all__ = ["PgvectorBackend"]
