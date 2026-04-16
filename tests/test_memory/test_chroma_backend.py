"""Tests for ChromaBackend — Chroma-backed VectorBackend adapter.

Skipped entirely when ``chromadb`` isn't installed, so the base suite
stays green without the optional dep. When installed, the tests
exercise the full adapter surface (add/search/get_vectors/remove/
clear/snapshot/restore) plus filter pushdown for supported operators
plus a MemoryLayer end-to-end round-trip.

Chroma's ``where`` only filters metadata (it cannot filter on the
primary id column), so pushdown tests add entries WITH metadata via
the adapter's ``metadatas=`` kwarg on ``add``. MemoryLayer's default
code path does not pass metadata to ``backend.add`` — metadata lives
in MemoryLayer's Python side — so the MemoryLayer-level filter test
exercises the fallback path where ``search`` raises
``FilterPushdownUnsupported`` and MemoryLayer uses ``search_subset``
against a Python-pre-filtered candidate set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

chromadb = pytest.importorskip("chromadb")

from soma.memory.api import MemoryLayer  # noqa: E402
from soma.memory.backend import FilterPushdownUnsupported, VectorBackend  # noqa: E402
from soma.memory.backends.chroma import ChromaBackend  # noqa: E402


def _rand_vecs(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _make(tmp_path: Path, *, dim: int = 8, **kw) -> ChromaBackend:
    return ChromaBackend(
        path=str(tmp_path / "chroma"),
        collection_name="soma_test",
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


def test_name_is_chroma(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        assert b.name == "chroma"
    finally:
        b.close()


def test_supports_filter_pushdown_true(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        assert b.supports_filter_pushdown is True
    finally:
        b.close()


def test_requires_path_or_client(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path|client"):
        ChromaBackend(collection_name="soma_test", dim=8)


def test_accepts_prebuilt_client(tmp_path: Path) -> None:
    """Caller can pass a pre-built chromadb client (escape hatch for
    non-default configs — auth, tenancy, HTTP mode)."""
    client = chromadb.EphemeralClient()
    b = ChromaBackend(client=client, collection_name="soma_test", dim=4)
    try:
        b.add(["a"], _rand_vecs(1, 4))
        assert b.ntotal == 1
    finally:
        b.close()


# ----------------------------------------------------------------------
# Round-trips
# ----------------------------------------------------------------------


def test_add_and_search_round_trip(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(3)]
        vecs = _rand_vecs(3, 8)
        b.add(ids, vecs)
        assert b.ntotal == 3
        hits = b.search(vecs[0], k=2)
        assert len(hits) == 2
        assert set(nid for nid, _ in hits) <= set(ids)
        # Cosine self-match first — score near 1.0.
        assert hits[0][0] == "id-0"
        assert hits[0][1] > 0.999
    finally:
        b.close()


def test_ntotal_and_dim(tmp_path: Path) -> None:
    b = ChromaBackend(
        path=str(tmp_path / "c"), collection_name="soma_test", dim=4
    )
    try:
        assert b.ntotal == 0
        assert b.dim == 4
        b.add(["a"], np.ones((1, 4), dtype=np.float32))
        assert b.ntotal == 1
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


def test_remove_drops_ids(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(4)]
        vecs = _rand_vecs(4, 8)
        b.add(ids, vecs)
        b.remove(["id-1"])
        assert b.ntotal == 3
        hits = b.search(vecs[0], k=10)
        returned = {nid for nid, _ in hits}
        assert "id-1" not in returned
    finally:
        b.close()


def test_remove_missing_id_is_noop(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["a", "b"], _rand_vecs(2, 8))
        # Chroma only warns on unknown id; adapter must not raise.
        b.remove(["does-not-exist"])
        assert b.ntotal == 2
    finally:
        b.close()


def test_clear_resets(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["a", "b"], _rand_vecs(2, 8))
        assert b.ntotal == 2
        b.clear()
        assert b.ntotal == 0
        # Post-clear add still works.
        b.add(["c"], _rand_vecs(1, 8))
        assert b.ntotal == 1
    finally:
        b.close()


def test_get_vectors_returns_stored(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        ids = [f"id-{i}" for i in range(4)]
        vecs = _rand_vecs(4, 8, seed=11)
        b.add(ids, vecs)
        got = b.get_vectors(["id-2", "id-0"])
        assert got.shape == (2, 8)
        # Chroma stores vectors as-supplied; compare exactly (float32
        # round-trip through the collection preserves the value).
        np.testing.assert_allclose(got[0], vecs[2], atol=1e-5)
        np.testing.assert_allclose(got[1], vecs[0], atol=1e-5)
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


def test_add_shape_mismatch_raises(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        with pytest.raises(ValueError):
            b.add(["a"], _rand_vecs(1, 4))
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


def test_filter_pushdown_equality(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(
            ["a", "b", "c"],
            _rand_vecs(3, 8),
            metadatas=[{"tag": "x"}, {"tag": "y"}, {"tag": "x"}],
        )
        hits = b.search(_rand_vecs(1, 8)[0], k=5, where={"tag": "x"})
        returned = {nid for nid, _ in hits}
        assert returned == {"a", "c"}
    finally:
        b.close()


def test_filter_pushdown_in(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(
            ["a", "b", "c", "d"],
            _rand_vecs(4, 8),
            metadatas=[
                {"tag": "x"},
                {"tag": "y"},
                {"tag": "z"},
                {"tag": "x"},
            ],
        )
        hits = b.search(
            _rand_vecs(1, 8)[0], k=5, where={"tag": {"$in": ["x", "y"]}}
        )
        returned = {nid for nid, _ in hits}
        assert returned == {"a", "b", "d"}
    finally:
        b.close()


def test_filter_pushdown_unsupported_op_raises(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["a"], _rand_vecs(1, 8), metadatas=[{"tag": "x"}])
        with pytest.raises(FilterPushdownUnsupported):
            b.search(
                _rand_vecs(1, 8)[0],
                k=1,
                where={"tag": {"$regex": "^x"}},
            )
    finally:
        b.close()


def test_filter_pushdown_on_missing_metadata_falls_back(tmp_path: Path) -> None:
    """When ``add`` was called without ``metadatas=``, Chroma has
    nothing to filter on. The adapter raises FilterPushdownUnsupported
    so MemoryLayer falls back to its Python pre-filter + search_subset
    path. This is the same pattern LanceDB uses when the requested
    column isn't in the table schema."""
    b = _make(tmp_path)
    try:
        b.add(["a", "b"], _rand_vecs(2, 8))  # no metadatas
        with pytest.raises(FilterPushdownUnsupported):
            b.search(
                _rand_vecs(1, 8)[0], k=5, where={"tag": {"$eq": "x"}}
            )
    finally:
        b.close()


# ----------------------------------------------------------------------
# Snapshot / restore
# ----------------------------------------------------------------------


def test_snapshot_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    bundle = tmp_path / "bundle"
    dst = tmp_path / "dst"
    b = ChromaBackend(path=str(src), collection_name="soma_test", dim=8)
    try:
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand_vecs(5, 8, seed=42)
        b.add(ids, vecs)
        bundle.mkdir()
        b.snapshot(bundle)
    finally:
        b.close()

    b2 = ChromaBackend(path=str(dst), collection_name="soma_test", dim=8)
    try:
        b2.restore(bundle)
        assert b2.ntotal == 5
        got = b2.get_vectors(["id-3"])
        np.testing.assert_allclose(got[0], vecs[3], atol=1e-5)
    finally:
        b2.close()


def test_snapshot_writes_sidecar_meta(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["a"], _rand_vecs(1, 8))
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        b.snapshot(bundle)
        meta_path = bundle / "backend.json"
        assert meta_path.exists()
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["backend"] == "chroma"
        assert meta["dim"] == 8
        assert meta["collection_name"] == "soma_test"
    finally:
        b.close()


def test_restore_without_sidecar_is_noop(tmp_path: Path) -> None:
    b = _make(tmp_path)
    try:
        b.add(["a"], _rand_vecs(1, 8))
        empty = tmp_path / "empty"
        empty.mkdir()
        b.restore(empty)
        assert b.ntotal == 1
    finally:
        b.close()


def test_reopen_same_path_sees_existing_rows(tmp_path: Path) -> None:
    """PersistentClient auto-persists; reopening the same path must
    see the prior rows."""
    path = tmp_path / "chroma"
    b = ChromaBackend(path=str(path), collection_name="soma_test", dim=8)
    try:
        b.add([f"id-{i}" for i in range(3)], _rand_vecs(3, 8))
        assert b.ntotal == 3
    finally:
        b.close()

    b2 = ChromaBackend(path=str(path), collection_name="soma_test", dim=8)
    try:
        assert b2.ntotal == 3
    finally:
        b2.close()


# ----------------------------------------------------------------------
# Optional-dep error path
# ----------------------------------------------------------------------


def test_import_error_when_chromadb_missing(tmp_path: Path, monkeypatch) -> None:
    """Simulate chromadb being absent: instantiation must fail with a
    pip install hint so users know which extra to add."""
    import soma.memory.backends.chroma as chroma_module

    monkeypatch.setattr(chroma_module, "_HAS_CHROMA", False)
    with pytest.raises(ImportError, match=r"soma\[chroma\]"):
        ChromaBackend(
            path=str(tmp_path / "x"),
            collection_name="soma_test",
            dim=8,
        )


# ----------------------------------------------------------------------
# MemoryLayer end-to-end
# ----------------------------------------------------------------------


def test_memory_layer_end_to_end(tmp_path: Path) -> None:
    """Store, retrieve, forget, save, load — the operator contract for
    a Chroma-backed MemoryLayer."""
    dim = 16

    def _embed(text: str) -> torch.Tensor:
        h = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return torch.from_numpy(vec)

    backend = ChromaBackend(
        path=str(tmp_path / "mem_chroma"),
        collection_name="soma_mem",
        dim=dim,
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
        assert len(mem) == 2

        save_dir = tmp_path / "bundle_out"
        mem.save(save_dir)
    finally:
        mem.close()

    reopened = MemoryLayer.load(save_dir, embed_fn=_embed)
    try:
        hits = reopened.retrieve("apples", k=5)
        returned_ids = {h.node_id for h in hits}
        assert ida in returned_ids
        assert idc in returned_ids
        assert idb not in returned_ids
    finally:
        reopened.close()


def test_memory_layer_where_pushdown_on_chroma(tmp_path: Path) -> None:
    """`retrieve(where=...)` over Chroma: MemoryLayer does not pass
    metadata through ``backend.add``, so the Chroma collection has no
    metadata to filter on. The adapter raises
    FilterPushdownUnsupported and MemoryLayer falls back to Python
    pre-filter + ``search_subset``. Observable contract: only entries
    matching the filter come back."""
    dim = 16

    def _embed(text: str) -> torch.Tensor:
        h = hash(text) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return torch.from_numpy(vec)

    backend = ChromaBackend(
        path=str(tmp_path / "ml_filter"),
        collection_name="soma_filter",
        dim=dim,
    )
    mem = MemoryLayer(embed_fn=_embed, embed_dim=dim, backend=backend)
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
