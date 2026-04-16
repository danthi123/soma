"""Tests for QdrantBackend — in-memory + local-file modes.

Skipped entirely when qdrant-client isn't installed, so the base
test suite stays green without the optional dep. When installed,
both modes round-trip add/search/remove through real qdrant calls
(the in-memory mode embeds a full qdrant core in-process, no network).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from soma.memory.backend import VectorBackend  # noqa: E402
from soma.memory.backends.qdrant import QdrantBackend  # noqa: E402


def _rand_vecs(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_protocol_isinstance() -> None:
    b = QdrantBackend(mode="memory", dim=8)
    assert isinstance(b, VectorBackend)


def test_add_search_roundtrip_in_memory_mode() -> None:
    b = QdrantBackend(mode="memory", dim=8)
    ids = [f"id-{i}" for i in range(10)]
    vecs = _rand_vecs(10, 8)
    b.add(ids, vecs)
    assert b.ntotal == 10
    hits = b.search(vecs[0], k=3)
    assert len(hits) == 3
    assert hits[0][0] == "id-0"  # self-match on top


def test_add_search_roundtrip_local_file_mode(tmp_path) -> None:
    path = tmp_path / "qdrant_local"
    b = QdrantBackend(mode="local", dim=8, path=str(path))
    ids = [f"id-{i}" for i in range(5)]
    vecs = _rand_vecs(5, 8, seed=7)
    b.add(ids, vecs)
    assert b.ntotal == 5
    b.close()

    b2 = QdrantBackend(
        mode="local", dim=8, path=str(path), collection=b._collection
    )
    # Re-opening the same on-disk collection should see the five rows.
    assert b2.ntotal == 5
    hits = b2.search(vecs[2], k=3)
    assert hits[0][0] == "id-2"


def test_remove_omits_from_search() -> None:
    b = QdrantBackend(mode="memory", dim=8)
    ids = [f"id-{i}" for i in range(6)]
    vecs = _rand_vecs(6, 8)
    b.add(ids, vecs)
    b.remove(["id-3"])
    assert b.ntotal == 5
    hits = b.search(vecs[0], k=10)
    returned = {nid for nid, _ in hits}
    assert "id-3" not in returned


def test_get_vectors_roundtrips_stored_vectors() -> None:
    b = QdrantBackend(mode="memory", dim=8)
    ids = [f"id-{i}" for i in range(4)]
    vecs = _rand_vecs(4, 8, seed=11)
    b.add(ids, vecs)
    got = b.get_vectors(["id-2", "id-0"])
    assert got.shape == (2, 8)
    # Qdrant may renormalize under COSINE distance; compare via cosine
    # similarity of the returned row against the stored row.
    for i, nid in enumerate(["id-2", "id-0"]):
        original = vecs[int(nid.split("-")[1])]
        dot = float(np.dot(got[i], original))
        norms = float(np.linalg.norm(got[i]) * np.linalg.norm(original))
        sim = dot / (norms + 1e-12)
        assert sim > 0.99, f"expected cosine ~1.0 for {nid}, got {sim}"


def test_search_subset_restricts_to_candidates() -> None:
    b = QdrantBackend(mode="memory", dim=8)
    ids = [f"id-{i}" for i in range(8)]
    vecs = _rand_vecs(8, 8, seed=3)
    b.add(ids, vecs)
    subset = ["id-2", "id-5", "id-7"]
    hits = b.search_subset(vecs[0], subset, k=10)
    returned = {nid for nid, _ in hits}
    assert returned <= set(subset)
    assert 0 < len(hits) <= 3


def test_supports_filter_pushdown_true() -> None:
    assert QdrantBackend(mode="memory", dim=8).supports_filter_pushdown is True


def test_name_is_qdrant() -> None:
    assert QdrantBackend(mode="memory", dim=8).name == "qdrant"


def test_collection_per_bundle_name() -> None:
    b = QdrantBackend(
        mode="memory", dim=8, collection="bundle_abc"
    )
    assert b._collection == "bundle_abc"


def test_local_mode_warns_above_20k(tmp_path, monkeypatch) -> None:
    """Inserting past the 20K ceiling in local mode should raise
    an UserWarning so operators know to migrate to HTTP before the
    disk-backed embedded core stops performing.

    Monkeypatches the threshold down to 5 so we don't have to
    actually write 20K rows through Qdrant's disk core in a unit
    test — the semantics we care about are "fires once, after the
    threshold, only in local mode".
    """
    from soma.memory.backends import qdrant as qdrant_module

    monkeypatch.setattr(
        qdrant_module, "LOCAL_MODE_ENTRY_WARNING_THRESHOLD", 5
    )

    path = tmp_path / "cap_qdrant"
    b = QdrantBackend(mode="local", dim=4, path=str(path))
    b.add([f"id-{i}" for i in range(4)], _rand_vecs(4, 4, seed=0))
    # Still under the cap — no warning yet.
    with warnings.catch_warnings(record=True) as caught1:
        warnings.simplefilter("always")
        b.add([f"id-{i}" for i in range(4, 5)], _rand_vecs(1, 4, seed=1))
    assert not [
        w for w in caught1 if issubclass(w.category, UserWarning)
    ]
    # Cross the cap — warning must fire.
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        b.add([f"id-{i}" for i in range(5, 8)], _rand_vecs(3, 4, seed=2))
    msgs = [
        str(w.message)
        for w in caught2
        if issubclass(w.category, UserWarning)
    ]
    assert any("local" in m for m in msgs), msgs
    # Second crossing — warning should not repeat (we set the flag).
    with warnings.catch_warnings(record=True) as caught3:
        warnings.simplefilter("always")
        b.add([f"id-{i}" for i in range(8, 10)], _rand_vecs(2, 4, seed=3))
    assert not [
        w for w in caught3 if issubclass(w.category, UserWarning)
    ]


def test_clear_empties_store() -> None:
    b = QdrantBackend(mode="memory", dim=8)
    b.add(["a", "b"], _rand_vecs(2, 8))
    b.clear()
    assert b.ntotal == 0


def test_dim_matches_constructor() -> None:
    assert QdrantBackend(mode="memory", dim=13).dim == 13


def test_qdrant_backend_raises_clear_error_when_dep_missing(monkeypatch) -> None:
    """Simulate qdrant-client being absent: instantiation must fail
    with a pip install hint so users know what extra to add."""
    import soma.memory.backends.qdrant as qdrant_module

    def _fake_ensure():
        raise ImportError(
            "QdrantBackend requires qdrant-client. "
            "Install with: pip install 'soma[qdrant]'"
        )

    monkeypatch.setattr(qdrant_module, "_ensure_qdrant_available", _fake_ensure)
    with pytest.raises(ImportError, match="soma\\[qdrant\\]"):
        QdrantBackend(mode="memory", dim=8)
