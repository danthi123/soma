"""LanceDBBackend — LanceDB-backed ``VectorBackend`` adapter.

LanceDB fills the "local-first, scales past 20K without a server"
niche that Qdrant-local can't serve and Qdrant-HTTP needs infra for.
Arrow-native, columnar, and fully embedded: a LanceDB table *is* a
directory on disk, which composes naturally with SOMA's bundle
layout.

Three index profiles, one class:

- ``"flat"`` — no index. Exact cosine on every query. Fine up to
  ~100K rows on consumer hardware.
- ``"ivf_pq"`` — IVF + Product Quantization. Default approximate
  path: much smaller memory footprint + fast at 100K to tens of
  millions.
- ``"hnsw"`` — IVF_HNSW_SQ. Higher recall than PQ at the same N,
  costs more disk.

Features:

- Cosine distance (the only metric MemoryLayer advertises today).
- Opaque SOMA ``node_id`` strings stored as the table's ``id``
  column; rows are upserted via ``merge_insert`` so repeated adds
  don't duplicate. Drops go through ``table.delete("id IN (...)")``.
- ``supports_filter_pushdown=True``. The translator
  (:mod:`soma.memory.backends.lancedb_filter`) renders Chroma-style
  ``where`` dicts as LanceDB SQL-like predicate strings.
- Snapshot writes nothing special — the LanceDB directory IS the
  snapshot. :meth:`snapshot` just copies the directory into the
  bundle so operators can archive / ship / reopen the bundle
  elsewhere.

lancedb is an optional dep — the import is guarded so SOMA still
ships without it. Constructing a ``LanceDBBackend`` when the dep is
missing raises :class:`ImportError` with a pip install hint.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np

from soma.memory.backend import FilterPushdownUnsupported
from soma.memory.backends.lancedb_filter import to_lancedb_where

# LanceDB indexes only become worthwhile past a few thousand rows.
# Below this, the query planner does a full scan anyway and the
# index-build overhead is wasted. Operators can force a build with
# ``auto_index_threshold=0``.
DEFAULT_AUTO_INDEX_THRESHOLD = 50_000


def _ensure_lancedb_available() -> None:
    try:
        import lancedb  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "LanceDBBackend requires lancedb. Install with: pip install 'soma[lancedb]'"
        ) from exc


class LanceDBBackend:
    """LanceDB-backed :class:`VectorBackend`.

    Parameters
    ----------
    path:
        Directory root for the LanceDB dataset. Created on first open.
    dim:
        Embedding dimension. Pinned at construction; mismatch on
        upsert raises.
    table_name:
        Which LanceDB table to open/create. Defaults to ``"vectors"``;
        MemoryLayer can pass the bundle name for multi-tenant layouts.
    distance:
        Distance metric — ``"cosine"`` (default), ``"l2"``, or
        ``"dot"``. MemoryLayer today always uses cosine; the other
        two are wired for future callers that bring their own
        embedder with non-unit-norm vectors.
    index_type:
        ``"flat"`` (default, no index), ``"ivf_pq"``, or ``"hnsw"``.
        The index is built lazily — first search past
        ``auto_index_threshold`` rows triggers it.
    num_partitions / num_sub_vectors:
        IVF-PQ tuning. Defaults follow LanceDB's documented sweet
        spot for 100K-1M rows.
    m / ef_construction:
        HNSW tuning. Ignored unless ``index_type="hnsw"``.
    auto_index_threshold:
        Minimum row count before index build fires. Default 50K.
    recreate:
        When True, drop any existing table before upserting. Useful
        for tests and benchmarks.
    """

    supports_filter_pushdown: bool = True
    name: str = "lancedb"

    def __init__(
        self,
        *,
        path: str | Path,
        dim: int,
        table_name: str = "vectors",
        distance: Literal["cosine", "l2", "dot"] = "cosine",
        index_type: Literal["flat", "ivf_pq", "hnsw"] = "flat",
        num_partitions: int = 256,
        num_sub_vectors: int = 96,
        m: int = 16,
        ef_construction: int = 200,
        auto_index_threshold: int = DEFAULT_AUTO_INDEX_THRESHOLD,
        recreate: bool = False,
    ) -> None:
        _ensure_lancedb_available()
        if distance not in ("cosine", "l2", "dot"):
            raise ValueError(f"distance must be cosine|l2|dot, got {distance!r}")
        if index_type not in ("flat", "ivf_pq", "hnsw"):
            raise ValueError(f"index_type must be flat|ivf_pq|hnsw, got {index_type!r}")
        self._path = Path(path)
        self._dim = int(dim)
        self._table_name = table_name
        self._distance = distance
        self._index_type = index_type
        self._num_partitions = int(num_partitions)
        self._num_sub_vectors = int(num_sub_vectors)
        self._m = int(m)
        self._ef_construction = int(ef_construction)
        self._auto_index_threshold = int(auto_index_threshold)
        self._recreate = recreate
        self._db: Any = None
        self._table: Any = None
        self._indexed = False
        self._opened = False
        self.open()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return
        import lancedb
        import pyarrow as pa

        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))
        schema = pa.schema(
            [
                pa.field("id", pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), self._dim)),
            ]
        )
        existing = _list_tables(self._db)
        if self._recreate and self._table_name in existing:
            self._db.drop_table(self._table_name)
            existing = _list_tables(self._db)
        if self._table_name in existing:
            self._table = self._db.open_table(self._table_name)
        else:
            self._table = self._db.create_table(self._table_name, schema=schema)
        # LanceDB persists indexes on disk; track whether one already
        # exists so we don't rebuild it on every open().
        try:
            self._indexed = bool(self._table.list_indices())
        except Exception:
            self._indexed = False
        self._opened = True

    def close(self) -> None:
        # LanceDB's Python client holds handles via the underlying rust
        # core; dropping our references is enough for the OS to release
        # them. There's no explicit close() to call.
        self._table = None
        self._db = None
        self._opened = False

    def clear(self) -> None:
        """Drop every indexed vector but keep the table open."""
        assert self._table is not None
        if self._table.count_rows() > 0:
            # `delete("true")` clears all rows without recreating the
            # table — cheaper than drop + recreate and preserves any
            # existing index schema.
            self._table.delete("true")
        self._indexed = False

    # ------------------------------------------------------------------
    # Vector ops
    # ------------------------------------------------------------------
    @property
    def ntotal(self) -> int:
        if self._table is None:
            return 0
        return int(self._table.count_rows())

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        assert self._table is not None
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(ids) or arr.shape[1] != self._dim:
            raise ValueError(
                f"vectors shape {arr.shape} mismatched; expected ({len(ids)}, {self._dim})"
            )
        rows = [{"id": nid, "vector": vec.tolist()} for nid, vec in zip(ids, arr, strict=True)]
        # merge_insert upserts: when the id already exists we replace
        # the vector, when it doesn't we insert. MemoryLayer never
        # re-adds an id (uuid4 per entry) but the WAL-replay path and
        # benchmarks occasionally re-seed an existing table, so
        # idempotent upsert is the safer contract.
        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        assert self._table is not None
        # Delete via `id IN (...)` — LanceDB accepts SQL-style `IN`
        # lists up to a few thousand items in one statement. Beyond
        # that we chunk to stay inside the parser's comfort zone.
        chunk_size = 1000
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            literals = ", ".join(f"'{nid.replace(chr(39), chr(39) + chr(39))}'" for nid in chunk)
            self._table.delete(f"id IN ({literals})")

    def get_vectors(self, ids: list[str]) -> np.ndarray:
        if not ids:
            return np.empty((0, self._dim), dtype=np.float32)
        assert self._table is not None
        import pyarrow as pa
        import pyarrow.compute as pc

        arr_tbl = self._table.to_arrow()
        mask = pc.is_in(arr_tbl["id"], pa.array(ids))
        filtered = arr_tbl.filter(mask)
        id_to_vec: dict[str, np.ndarray] = {
            str(nid): np.asarray(vec, dtype=np.float32)
            for nid, vec in zip(
                filtered["id"].to_pylist(),
                filtered["vector"].to_pylist(),
                strict=True,
            )
        }
        rows = np.zeros((len(ids), self._dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            vec = id_to_vec.get(nid)
            if vec is None:
                raise KeyError(f"id {nid!r} not in backend")
            rows[i] = vec
        return rows

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        exclude_ids: set[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if self.ntotal == 0 or k <= 0:
            return []
        assert self._table is not None
        self._maybe_build_index()
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(f"query dim mismatch: backend={self._dim}, got={q.shape[0]}")

        predicate_parts: list[str] = []
        if where is not None:
            # Translator raises FilterPushdownUnsupported for anything
            # we can't express; MemoryLayer catches + falls back.
            clause = to_lancedb_where(where)
            if clause:
                predicate_parts.append(clause)
        if exclude_ids:
            literals = ", ".join(_sql_string_literal(nid) for nid in exclude_ids)
            predicate_parts.append(f"id NOT IN ({literals})")

        builder = self._table.search(q)
        try:
            builder = builder.distance_type(self._distance)
        except AttributeError:
            # Older lancedb builds used ``.metric(...)``; keep a
            # fallback so the adapter stays working across the extra's
            # min-version range.
            builder = builder.metric(self._distance)  # type: ignore[attr-defined]
        if predicate_parts:
            builder = builder.where(" AND ".join(predicate_parts))
        # Over-fetch a handful when exclusions are active in case the
        # planner can't push them into the ANN search itself.
        limit = k + (len(exclude_ids) if exclude_ids else 0)
        try:
            rows = builder.limit(limit).to_list()
        except Exception as exc:
            # LanceDB raises RuntimeError (wrapping a schema error)
            # when the filter references a column that doesn't exist
            # in the current table schema. The current schema is
            # ``{id, vector}`` so metadata filters on other fields
            # fail here. Convert to FilterPushdownUnsupported so
            # MemoryLayer falls back to the Python pre-filter +
            # ``search_subset`` path, which every backend (including
            # this one via WHERE id IN (...)) can serve.
            msg = str(exc)
            if _is_schema_error(msg):
                raise FilterPushdownUnsupported(
                    op="schema",
                    field=_extract_missing_field(msg),
                    message=(
                        "LanceDB table schema does not include the "
                        "referenced field(s); MemoryLayer will use its "
                        "Python pre-filter path instead."
                    ),
                ) from exc
            raise
        out = [
            (
                str(row["id"]),
                _distance_to_score(float(row["_distance"]), metric=self._distance),
            )
            for row in rows
        ]
        if exclude_ids:
            out = [(nid, s) for nid, s in out if nid not in exclude_ids]
        return out[:k]

    def search_subset(
        self,
        query: np.ndarray,
        candidate_ids: list[str],
        k: int,
    ) -> list[tuple[str, float]]:
        if not candidate_ids or k <= 0:
            return []
        assert self._table is not None
        self._maybe_build_index()
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        literals = ", ".join(_sql_string_literal(nid) for nid in candidate_ids)
        where_clause = f"id IN ({literals})"
        builder = self._table.search(q)
        try:
            builder = builder.distance_type(self._distance)
        except AttributeError:
            builder = builder.metric(self._distance)  # type: ignore[attr-defined]
        rows = builder.where(where_clause).limit(min(k, len(candidate_ids))).to_list()
        return [
            (
                str(row["id"]),
                _distance_to_score(float(row["_distance"]), metric=self._distance),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self, bundle_dir: Path) -> None:
        """Flush + copy the LanceDB directory into the bundle.

        LanceDB's on-disk format IS its snapshot — tables are
        append-only layered files with a manifest pointing at the
        current version. We compact_files() first so the copy is as
        small as possible, then write a ``lancedb`` subdirectory
        inside the bundle. :meth:`restore` does the inverse.
        """
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if self._table is not None:
            # Best-effort compaction — reduces the number of file
            # fragments we have to ship. Newer LanceDB uses
            # ``optimize()``; older builds exposed ``compact_files``.
            # Either path is safe to skip on error (the snapshot still
            # works with fragmented files, just larger on disk).
            try:
                optimize = getattr(self._table, "optimize", None)
                if optimize is not None:
                    optimize()
                else:
                    self._table.compact_files()
            except Exception:
                pass
        meta = {
            "backend": "lancedb",
            "dim": self._dim,
            "table_name": self._table_name,
            "distance": self._distance,
            "index_type": self._index_type,
            "lancedb_version": _lancedb_version(),
        }
        (bundle_dir / "backend.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        dest = bundle_dir / "lancedb"
        if dest.exists():
            shutil.rmtree(dest)
        if self._path.exists():
            shutil.copytree(self._path, dest)

    def restore(self, bundle_dir: Path) -> None:
        """Inverse of :meth:`snapshot`.

        Copies the ``lancedb/`` subdirectory of the bundle back into
        ``self._path`` (wiping what was there) and reopens the table.
        If the bundle has no ``lancedb/`` subdir, we leave the current
        state alone — MemoryLayer will rebuild via WAL replay.
        """
        meta_path = bundle_dir / "backend.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("backend") == "lancedb":
                self._table_name = meta.get("table_name", self._table_name)
                self._distance = meta.get("distance", self._distance)
                self._index_type = meta.get("index_type", self._index_type)
        src = bundle_dir / "lancedb"
        if not src.exists():
            return
        self.close()
        if self._path.exists():
            shutil.rmtree(self._path)
        shutil.copytree(src, self._path)
        self.open()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _maybe_build_index(self) -> None:
        """Lazily build the configured index once the table is big enough.

        Keeps construction cheap by skipping index work until the
        store actually crosses the threshold. Rebuilds are idempotent
        (LanceDB's ``replace=True`` default), so a second call after
        more rows were added will refresh it.
        """
        if self._index_type == "flat":
            return
        if self._table is None:
            return
        if self.ntotal < self._auto_index_threshold:
            return
        if self._indexed:
            return
        try:
            metric = _LANCE_METRIC_MAP[self._distance]
            if self._index_type == "ivf_pq":
                self._table.create_index(
                    metric=metric,
                    vector_column_name="vector",
                    index_type="IVF_PQ",
                    num_partitions=self._num_partitions,
                    num_sub_vectors=self._num_sub_vectors,
                )
            elif self._index_type == "hnsw":
                self._table.create_index(
                    metric=metric,
                    vector_column_name="vector",
                    index_type="IVF_HNSW_SQ",
                    num_partitions=self._num_partitions,
                    m=self._m,
                    ef_construction=self._ef_construction,
                )
            self._indexed = True
        except Exception:
            # A failed index build should not break search — we fall
            # back to the exact-scan path LanceDB uses when no index
            # is present. The next add/open can retry.
            self._indexed = False


def _is_schema_error(message: str) -> bool:
    """Heuristic: does the LanceDB error message say "no such column"?

    LanceDB wraps a rust-side ``Schema`` error in a Python
    ``RuntimeError`` whose text contains ``Schema error: No field
    named ...``. We match on the substring rather than exception
    type because the exact rust-layer type is not part of the
    Python API.
    """
    low = message.lower()
    return "no field named" in low or "schema error" in low or "cannot find column" in low


def _extract_missing_field(message: str) -> str | None:
    """Pull the offending column name out of a LanceDB schema error."""
    import re

    match = re.search(r"No field named (\w+)", message)
    if match is not None:
        return match.group(1)
    return None


def _list_tables(db: Any) -> list[str]:
    """Return the list of table names from a LanceDBConnection.

    Older LanceDB releases exposed ``table_names()``; newer ones
    ship ``list_tables()`` returning a response object with a
    ``.tables`` attribute. Probe both so the adapter works across
    the extra's min-version range without deprecation spam.
    """
    lister = getattr(db, "list_tables", None)
    if lister is not None:
        try:
            resp = lister()
        except TypeError:
            # Some client builds require a positional namespace arg.
            resp = lister(None)
        if hasattr(resp, "tables"):
            return list(resp.tables)
        if isinstance(resp, list):
            return list(resp)
    # Legacy path; silent deprecation on newer clients but still
    # functionally correct.
    return list(db.table_names())


def _sql_string_literal(value: str) -> str:
    """Render a Python string as a LanceDB SQL string literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _distance_to_score(distance: float, *, metric: str) -> float:
    """Convert a LanceDB distance into a cosine-similarity-shaped score.

    LanceDB reports the distance each metric natively uses:

    - ``cosine``: ``1 - cosine_similarity`` in ``[0, 2]``. We return
      ``1 - distance`` so the score range + direction matches every
      other backend (higher = more similar, ``[-1, 1]``).
    - ``l2``: squared L2. Monotone with -similarity; we return
      ``-distance`` so order is preserved.
    - ``dot``: negated inner product (LanceDB stores it that way so
      smaller-is-closer is uniform across metrics). We return
      ``-distance`` for the same reason.
    """
    if metric == "cosine":
        return 1.0 - distance
    # For l2 / dot, "higher == more similar" is achieved by negating;
    # MemoryLayer only cares about ordering for these metrics today.
    return -distance


def _lancedb_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("lancedb"))
    except Exception:
        return "unknown"


_LANCE_METRIC_MAP: dict[str, str] = {
    "cosine": "cosine",
    "l2": "l2",
    "dot": "dot",
}


__all__ = ["LanceDBBackend", "DEFAULT_AUTO_INDEX_THRESHOLD"]
