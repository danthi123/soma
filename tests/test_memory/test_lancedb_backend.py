"""Tests for LanceDBBackend — embedded arrow-native vector store.

Skipped entirely when ``lancedb`` isn't installed, so the base test
suite stays green without the optional dep. When installed, the
tests exercise the full adapter surface (add/search/get_vectors/
remove/clear/snapshot/restore) plus filter pushdown for every
supported operator plus a MemoryLayer end-to-end round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

lancedb = pytest.importorskip("lancedb")

from soma.memory.api import MemoryLayer  # noqa: E402
from soma.memory.backend import VectorBackend  # noqa: E402
from soma.memory.backends.lancedb import LanceDBBackend  # noqa: E402


def _rand_vecs(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _make(tmp_path: Path, *, dim: int = 8, **kw) -> LanceDBBackend:
    return LanceDBBackend(
        path=tmp_path / "lancedb",
        table_name="test",
        dim=dim,
        **kw,
    )


# ----------------------------------------------------------------------
# Basic identity + contract
# ----------------------------------------------------------------------


def test_protocol_isinstance(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        assert isinstance(b, VectorBackend)
    finally:
        b.close()


def test_name_is_lancedb(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        assert b.name == "lancedb"
    finally:
        b.close()


def test_supports_filter_pushdown_true(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        assert b.supports_filter_pushdown is True
    finally:
        b.close()


def test_dim_matches_constructor(tmp_path: Path) -> None:
    b = _make(tmp_path, dim=13)
    try:
        assert b.dim == 13
    finally:
        b.close()


def test_invalid_distance_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distance"):
        LanceDBBackend(
            path=tmp_path / "bad",
            dim=8,
            distance="manhattan",  # type: ignore[arg-type]
        )


def test_invalid_index_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="index_type"):
        LanceDBBackend(
            path=tmp_path / "bad",
            dim=8,
            index_type="brute_force",  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------
# Round-trips
# ----------------------------------------------------------------------


def test_add_then_search_top_k(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(6)]
        vecs = _rand_vecs(6, 8)
        b.add(ids, vecs)
        assert b.ntotal == 6
        hits = b.search(vecs[0], k=3)
        assert len(hits) == 3
        # Cosine self-match always first.
        assert hits[0][0] == "id-0"
        # Cosine similarity of an L2-normalized match is ~1.0; LanceDB
        # reports cosine-distance so we convert — top score should be
        # very near 1.0.
        assert hits[0][1] > 0.999
    finally:
        b.close()


def test_search_respects_exclude_ids(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(6)]
        vecs = _rand_vecs(6, 8)
        b.add(ids, vecs)
        hits = b.search(vecs[0], k=5, exclude_ids={"id-0"})
        returned = {nid for nid, _ in hits}
        assert "id-0" not in returned
        assert len(hits) == 5
    finally:
        b.close()


def test_remove_omits_from_search(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(6)]
        vecs = _rand_vecs(6, 8)
        b.add(ids, vecs)
        b.remove(["id-3"])
        assert b.ntotal == 5
        hits = b.search(vecs[0], k=10)
        returned = {nid for nid, _ in hits}
        assert "id-3" not in returned
    finally:
        b.close()


def test_remove_missing_id_is_noop(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["a", "b"], _rand_vecs(2, 8))
        # Removing an unknown id should not raise, and the table
        # should be unchanged.
        b.remove(["does-not-exist"])
        assert b.ntotal == 2
    finally:
        b.close()


def test_get_vectors_roundtrip(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(4)]
        vecs = _rand_vecs(4, 8, seed=11)
        b.add(ids, vecs)
        got = b.get_vectors(["id-2", "id-0"])
        assert got.shape == (2, 8)
        # LanceDB returns vectors as stored (no re-normalization like
        # Qdrant does), so we can assert exact equality.
        np.testing.assert_allclose(got[0], vecs[2])
        np.testing.assert_allclose(got[1], vecs[0])
    finally:
        b.close()


def test_get_vectors_missing_id_raises(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["present"], _rand_vecs(1, 8))
        with pytest.raises(KeyError):
            b.get_vectors(["missing"])
    finally:
        b.close()


def test_search_subset_restricts_to_candidates(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(8)]
        vecs = _rand_vecs(8, 8, seed=3)
        b.add(ids, vecs)
        subset = ["id-2", "id-5", "id-7"]
        hits = b.search_subset(vecs[0], subset, k=10)
        returned = {nid for nid, _ in hits}
        assert returned <= set(subset)
        assert 0 < len(hits) <= 3
    finally:
        b.close()


def test_clear_empties_store(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["a", "b"], _rand_vecs(2, 8))
        assert b.ntotal == 2
        b.clear()
        assert b.ntotal == 0
        # After clear, subsequent add still works without reopening.
        b.add(["c"], _rand_vecs(1, 8))
        assert b.ntotal == 1
    finally:
        b.close()


def test_add_upserts_on_duplicate_id(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        first = _rand_vecs(1, 8, seed=1)
        second = _rand_vecs(1, 8, seed=2)
        b.add(["dup"], first)
        b.add(["dup"], second)
        # Count stays at 1 — merge_insert replaced the row.
        assert b.ntotal == 1
        got = b.get_vectors(["dup"])
        np.testing.assert_allclose(got[0], second[0])
    finally:
        b.close()


def test_add_shape_mismatch_raises(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        # Wrong dim.
        with pytest.raises(ValueError):
            b.add(["a"], _rand_vecs(1, 4))
        # Wrong row count.
        with pytest.raises(ValueError):
            b.add(["a", "b"], _rand_vecs(1, 8))
    finally:
        b.close()


def test_add_empty_is_noop(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add([], np.empty((0, 8), dtype=np.float32))
        assert b.ntotal == 0
    finally:
        b.close()


# ----------------------------------------------------------------------
# Filter pushdown coverage
# ----------------------------------------------------------------------


def test_search_where_eq_pushdown(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        # We have to use string ids only here — LanceDB can't push a
        # filter on columns that don't exist. For pushdown to actually
        # work, the backend needs to accept extra columns on add. The
        # current adapter schema is {id, vector}, so tests that exercise
        # pushdown prove the WHERE-on-id path works (that IS the
        # pushdown contract for ids). Tests on richer schemas are
        # covered at the MemoryLayer level where `_retrieve_with_filter`
        # falls back to the Python path when the backend schema is
        # minimal.
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8)
        b.add(ids, vecs)
        # Filter by id using the backend's SQL surface directly.
        hits = b.search(vecs[0], k=5, where={"id": {"$eq": "id-2"}})
        assert [nid for nid, _ in hits] == ["id-2"]
    finally:
        b.close()


def test_search_where_ne_pushdown(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8)
        b.add(ids, vecs)
        hits = b.search(vecs[0], k=5, where={"id": {"$ne": "id-0"}})
        returned = [nid for nid, _ in hits]
        assert "id-0" not in returned
    finally:
        b.close()


def test_search_where_in_pushdown(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8)
        b.add(ids, vecs)
        hits = b.search(
            vecs[0], k=5, where={"id": {"$in": ["id-1", "id-3"]}}
        )
        returned = {nid for nid, _ in hits}
        assert returned == {"id-1", "id-3"}
    finally:
        b.close()


def test_search_where_nin_pushdown(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8)
        b.add(ids, vecs)
        hits = b.search(
            vecs[0], k=5, where={"id": {"$nin": ["id-1", "id-3"]}}
        )
        returned = {nid for nid, _ in hits}
        assert returned == {"id-0", "id-2", "id-4"}
    finally:
        b.close()


def test_search_where_unsupported_op_raises(tmp_path: Path) -> None:
    from soma.memory.backend import FilterPushdownUnsupported

    b = _make(tmp_path)
    try:
        b.add(["a"], _rand_vecs(1, 8))
        with pytest.raises(FilterPushdownUnsupported):
            b.search(
                _rand_vecs(1, 8)[0],
                k=1,
                where={"id": {"$regex": "^a"}},
            )
    finally:
        b.close()


# ----------------------------------------------------------------------
# Snapshot / restore
# ----------------------------------------------------------------------


def test_snapshot_restore_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    bundle = tmp_path / "bundle"
    dst = tmp_path / "dst"
    b = LanceDBBackend(path=src, table_name="snap", dim=8)
    try:
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8, seed=42)
        b.add(ids, vecs)
        b.snapshot(bundle)
    finally:
        b.close()

    # Fresh backend at a different path; restore pulls the data back.
    b2 = LanceDBBackend(path=dst, table_name="snap", dim=8)
    try:
        b2.restore(bundle)
        assert b2.ntotal == 5
        got = b2.get_vectors(["id-3"])
        np.testing.assert_allclose(got[0], vecs[3])
    finally:
        b2.close()


def test_snapshot_writes_sidecar_meta(tmp_path: Path) -> None:
    b = LanceDBBackend(
        path=tmp_path / "l",
        table_name="snap",
        dim=8,
        index_type="ivf_pq",
    )
    try:
        b.add(["a"], _rand_vecs(1, 8))
        bundle = tmp_path / "bundle"
        b.snapshot(bundle)
        meta_path = bundle / "backend.json"
        assert meta_path.exists()
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["backend"] == "lancedb"
        assert meta["dim"] == 8
        assert meta["table_name"] == "snap"
        assert meta["index_type"] == "ivf_pq"
    finally:
        b.close()


def test_restore_without_sidecar_is_noop(tmp_path: Path) -> None:
    b = LanceDBBackend(path=tmp_path / "l", table_name="t", dim=8)
    try:
        b.add(["a"], _rand_vecs(1, 8))
        # Empty bundle dir — no backend.json, no lancedb/.
        empty = tmp_path / "empty"
        empty.mkdir()
        # Should not wipe the current table or raise.
        b.restore(empty)
        assert b.ntotal == 1
    finally:
        b.close()


def test_reopen_same_path_sees_existing_rows(tmp_path: Path) -> None:
    """Closing + reopening the same LanceDB path should recover the
    previously-written rows. Implicit snapshot is just the directory
    itself; no extra work needed."""
    path = tmp_path / "lance"
    b = LanceDBBackend(path=path, table_name="t", dim=8)
    try:
        b.add([f"id-{i}" for i in range(3)], _rand_vecs(3, 8))
        assert b.ntotal == 3
    finally:
        b.close()

    b2 = LanceDBBackend(path=path, table_name="t", dim=8)
    try:
        assert b2.ntotal == 3
    finally:
        b2.close()


def test_recreate_drops_existing_table(tmp_path: Path) -> None:
    path = tmp_path / "lance"
    b = LanceDBBackend(path=path, table_name="t", dim=8)
    try:
        b.add([f"id-{i}" for i in range(3)], _rand_vecs(3, 8))
        assert b.ntotal == 3
    finally:
        b.close()

    b2 = LanceDBBackend(
        path=path, table_name="t", dim=8, recreate=True
    )
    try:
        assert b2.ntotal == 0
    finally:
        b2.close()


# ----------------------------------------------------------------------
# Optional-dep error path
# ----------------------------------------------------------------------


def test_lancedb_backend_raises_clear_error_when_dep_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """Simulate lancedb being absent: instantiation must fail with a
    pip install hint so users know which extra to add."""
    import soma.memory.backends.lancedb as lancedb_module

    def _fake_ensure():
        raise ImportError(
            "LanceDBBackend requires lancedb. "
            "Install with: pip install 'soma[lancedb]'"
        )

    monkeypatch.setattr(
        lancedb_module, "_ensure_lancedb_available", _fake_ensure
    )
    with pytest.raises(ImportError, match=r"soma\[lancedb\]"):
        LanceDBBackend(path=tmp_path / "x", dim=8)


# ----------------------------------------------------------------------
# MemoryLayer end-to-end
# ----------------------------------------------------------------------


def test_memory_layer_end_to_end_store_retrieve(tmp_path: Path) -> None:
    """Store, retrieve, forget, save, load, retrieve — the operator
    contract for a LanceDB-backed MemoryLayer."""
    # Deterministic tiny embedder: map each unique text to a
    # normalized hash-seeded vector. Lets the test stay fast + avoids
    # the sbert dep.
    dim = 16

    def _embed(text: str) -> torch.Tensor:
        h = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return torch.from_numpy(vec)

    backend = LanceDBBackend(
        path=tmp_path / "mem_lance", table_name="mem", dim=dim
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

        # Retrieve on a seen text — the self-embedding is the top hit.
        hits = mem.retrieve("apples are red", k=3)
        assert hits[0].node_id == ida

        # Forget then confirm retrieve no longer returns it.
        assert mem.forget(idb) is True
        remaining = {h.node_id for h in mem.retrieve("bananas", k=5)}
        assert idb not in remaining
        assert len(mem) == 2

        # Save bundle.
        save_dir = tmp_path / "bundle_out"
        mem.save(save_dir)
    finally:
        mem.close()

    # Load into a fresh MemoryLayer with a new LanceDB path.
    backend2 = LanceDBBackend(
        path=tmp_path / "mem_lance2", table_name="mem", dim=dim
    )
    mem2 = MemoryLayer(
        embed_fn=_embed,
        embed_dim=dim,
        bundle_path=tmp_path / "bundle_out2",
        backend=backend2,
    )
    try:
        # Replay the saved snapshot through the new backend.
        # MemoryLayer.load is the canonical path; but for this
        # test we use an explicit in-place rehydrate via save() then
        # reopen — the key invariant is that retrieve works after
        # store+save+reopen.
        reopened = MemoryLayer.load(
            save_dir,
            embed_fn=_embed,
        )
        try:
            hits = reopened.retrieve("apples", k=5)
            returned_ids = {h.node_id for h in hits}
            assert ida in returned_ids
            assert idc in returned_ids
            assert idb not in returned_ids
        finally:
            reopened.close()
    finally:
        mem2.close()


def test_memory_layer_where_pushdown_on_lancedb(tmp_path: Path) -> None:
    """`retrieve(where=...)` over LanceDB should filter correctly.

    LanceDB's schema today only exposes id/vector columns, so
    metadata filters can't be pushed down at the adapter level. The
    MemoryLayer `_retrieve_with_filter` path catches
    FilterPushdownUnsupported and falls back to the Python pre-filter
    + `search_subset`, which LanceDB handles via WHERE id IN (...).
    The observable result is identical either way: only entries
    matching the filter come back.
    """
    dim = 16

    def _embed(text: str) -> torch.Tensor:
        h = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return torch.from_numpy(vec)

    backend = LanceDBBackend(
        path=tmp_path / "ml_filter", table_name="t", dim=dim
    )
    mem = MemoryLayer(
        embed_fn=_embed,
        embed_dim=dim,
        backend=backend,
    )
    try:
        mem.store("alpha", metadata={"tag": "a"})
        mem.store("beta", metadata={"tag": "b"})
        mem.store("gamma", metadata={"tag": "a"})
        hits = mem.retrieve("alpha", k=5, where={"tag": "a"})
        tags = {h.metadata["tag"] for h in hits}
        assert tags == {"a"}
        assert len(hits) == 2
    finally:
        mem.close()
