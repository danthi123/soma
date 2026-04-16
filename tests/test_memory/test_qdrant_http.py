"""HTTP-mode tests for QdrantBackend.

Skipped unless ``SOMA_QDRANT_TEST_URL`` is set in the environment —
those tests require a real Qdrant HTTP endpoint (cloud, docker,
localhost). CI machines without Qdrant stay green.

To run these locally::

    docker run -p 6333:6333 qdrant/qdrant:v1.10.0
    SOMA_QDRANT_TEST_URL=http://localhost:6333 \\
      pytest tests/test_memory/test_qdrant_http.py -v

Tests cover:
- Basic add/search/get_vectors over HTTP
- Snapshot writes a ``backend.json`` sidecar with the URL+collection
- Restore re-points a new client at the same collection and sees
  existing ntotal without re-ingesting
- Server-side filter pushdown round-trips unchanged
"""

from __future__ import annotations

import json
import os
import uuid

import numpy as np
import pytest

qdrant_client = pytest.importorskip("qdrant_client")

_QDRANT_URL = os.environ.get("SOMA_QDRANT_TEST_URL")

pytestmark = pytest.mark.skipif(
    not _QDRANT_URL,
    reason="Set SOMA_QDRANT_TEST_URL to a running Qdrant server to run these.",
)

from soma.memory.backends.qdrant import QdrantBackend  # noqa: E402


def _unique_collection() -> str:
    return f"soma_http_test_{uuid.uuid4().hex[:12]}"


def _rand(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_http_mode_basic_roundtrip() -> None:
    coll = _unique_collection()
    b = QdrantBackend(
        mode="http", dim=16, url=_QDRANT_URL, collection=coll, recreate=True
    )
    try:
        ids = [f"id-{i}" for i in range(12)]
        vecs = _rand(12, 16, seed=5)
        b.add(ids, vecs)
        assert b.ntotal == 12
        hits = b.search(vecs[0], k=5)
        assert hits[0][0] == "id-0"
        got = b.get_vectors(["id-3", "id-7"])
        assert got.shape == (2, 16)
    finally:
        # Drop the test collection.
        assert b._client is not None
        try:
            b._client.delete_collection(coll)
        finally:
            b.close()


def test_http_snapshot_produces_sidecar_json(tmp_path) -> None:
    coll = _unique_collection()
    b = QdrantBackend(
        mode="http", dim=8, url=_QDRANT_URL, collection=coll, recreate=True
    )
    try:
        b.add(["x"], _rand(1, 8))
        b.snapshot(tmp_path)
        sidecar = tmp_path / "backend.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["backend"] == "qdrant"
        assert data["mode"] == "http"
        assert data["url"] == _QDRANT_URL
        assert data["collection"] == coll
        assert "qdrant_version" in data
    finally:
        if b._client is not None:
            try:
                b._client.delete_collection(coll)
            finally:
                b.close()


def test_http_restore_repoints_without_reingest(tmp_path) -> None:
    coll = _unique_collection()
    original = QdrantBackend(
        mode="http", dim=8, url=_QDRANT_URL, collection=coll, recreate=True
    )
    try:
        original.add([f"p-{i}" for i in range(5)], _rand(5, 8, seed=9))
        original.snapshot(tmp_path)
        original.close()

        # Fresh backend pointed at the same server + collection.
        restored = QdrantBackend(
            mode="http", dim=8, url=_QDRANT_URL, collection=coll
        )
        try:
            restored.restore(tmp_path)
            # Collection still has the rows; restore rebuilt the id map.
            assert restored.ntotal == 5
            hits = restored.search(_rand(1, 8, seed=42).reshape(-1), k=3)
            assert len(hits) == 3
        finally:
            if restored._client is not None:
                try:
                    restored._client.delete_collection(coll)
                finally:
                    restored.close()
    except Exception:
        if original._client is not None:
            try:
                original._client.delete_collection(coll)
            except Exception:
                pass
            original.close()
        raise


def test_http_filter_pushdown_matches_payload() -> None:
    """Filter pushdown round-trips the Chroma-style where through
    Qdrant's server-side filter engine."""
    from qdrant_client.http import models as qm

    coll = _unique_collection()
    b = QdrantBackend(
        mode="http", dim=4, url=_QDRANT_URL, collection=coll, recreate=True
    )
    try:
        # Manually upsert payloads (QdrantBackend.add only writes
        # node_id today; we inject extras to exercise filter pushdown).
        ids = [f"id-{i}" for i in range(5)]
        vecs = _rand(5, 4, seed=0)
        b.add(ids, vecs)
        assert b._client is not None
        for nid, tag in zip(ids, ["a", "b", "a", "c", "a"], strict=True):
            pid = b._id_to_point[nid]
            b._client.set_payload(
                collection_name=coll,
                payload={"tag": tag},
                points=[pid],
            )
        hits = b.search(vecs[0], k=10, where={"tag": "a"})
        ids_returned = {nid for nid, _ in hits}
        # Three ids tagged "a" in the corpus.
        assert ids_returned <= {"id-0", "id-2", "id-4"}
    finally:
        if b._client is not None:
            try:
                b._client.delete_collection(coll)
            finally:
                b.close()
