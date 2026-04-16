"""Cross-version Qdrant snapshot/restore compatibility tests.

Phase 24 — gated slow matrix. The default ``pytest`` run skips these
entirely. To exercise the matrix locally::

    pip install -e ".[qdrant-test]"
    SOMA_QDRANT_VERSION_MATRIX=1 \\
      pytest tests/test_memory/test_qdrant_version_compat.py -q

Requires a working Docker daemon (Docker Desktop on Windows/macOS,
dockerd on Linux). Each test spins a real Qdrant container via
``testcontainers-python``. Matrix:

- 3 smoke tests — one per pinned Qdrant version — round-trip
  add/search through :class:`QdrantBackend` pointed at the running
  container.
- 9 cross-version tests — every (src, tgt) pair over the same three
  versions — write a snapshot against ``src``, tear it down, bring up
  ``tgt``, recover the snapshot, and assert the top-1 search hit is
  preserved.

SOMA's :class:`QdrantBackend.snapshot` / ``.restore`` for HTTP mode
only re-points a client at a live collection; it does not export
vector data. The cross-version tests therefore drive Qdrant's native
REST snapshot API (``/collections/{c}/snapshots``) directly — that
IS the durable on-wire format operators depend on for backup and
migration, so this is exactly what we want to pin down.

Snapshot format stability was broken between Qdrant 1.10 and 1.11;
the matrix is deliberately pinned to 1.11+ to stay inside the
supported compatibility window. See the plan document at
``docs/plans/2026-04-16-phase-24-qdrant-version-compat.md``.

TODO(Phase 24 follow-up): once the operator decides between GitHub
Actions (cloud) and a self-hosted Gitea runner (Unraid Intel box),
wire a weekly CI job that runs this matrix with
``SOMA_QDRANT_VERSION_MATRIX=1`` set. Suggested shape lives in the
plan doc linked above.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guards — module must be importable even when testcontainers and/or
# qdrant-client are not installed. The module-level skip then hides every
# test at collection time so the default ``pytest`` run stays green.
# ---------------------------------------------------------------------------
try:
    from testcontainers.qdrant import QdrantContainer  # type: ignore[import-not-found]

    _HAS_TC = True
except ImportError:  # pragma: no cover - env-dependent
    QdrantContainer = None  # type: ignore[assignment,misc]
    _HAS_TC = False

try:
    import qdrant_client  # noqa: F401  # type: ignore[import-not-found]

    _HAS_QDRANT_CLIENT = True
except ImportError:  # pragma: no cover - env-dependent
    _HAS_QDRANT_CLIENT = False

try:
    import requests  # type: ignore[import-not-found]

    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover - env-dependent
    requests = None  # type: ignore[assignment]
    _HAS_REQUESTS = False


_GATE_ENV = "SOMA_QDRANT_VERSION_MATRIX"


def _gate_active() -> bool:
    return os.environ.get(_GATE_ENV) == "1"


# Module-level guard. If ANY dep is missing or the gate env isn't set,
# skip collection entirely so this file is a no-op in default runs.
if not _HAS_TC:
    pytestmark = pytest.mark.skip(reason="testcontainers not installed")
elif not _HAS_QDRANT_CLIENT:
    pytestmark = pytest.mark.skip(reason="qdrant-client not installed")
elif not _HAS_REQUESTS:
    pytestmark = pytest.mark.skip(reason="requests not installed")
elif not _gate_active():
    pytestmark = pytest.mark.skip(
        reason=f"set {_GATE_ENV}=1 to run the slow Qdrant version matrix"
    )
else:
    pytestmark = pytest.mark.slow_qdrant


# ---------------------------------------------------------------------------
# Matrix config
# ---------------------------------------------------------------------------
# Pinned to 1.11+ — snapshot format was reworked between 1.10 and 1.11
# (REST endpoint layout for recover-from-snapshot). Staying inside the
# post-1.11 window keeps the adapter-level test honest without having
# to special-case two snapshot dialects. See the plan doc for context.
_QDRANT_VERSIONS: list[str] = ["1.11.3", "1.12.4", "1.13.5"]

_DIM = 32


def _rand_vectors(n: int, dim: int = _DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _qdrant_url(container: Any) -> str:
    host = container.get_container_host_ip()
    port = container.get_exposed_port(6333)
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def qdrant_at_version(request: pytest.FixtureRequest) -> Iterator[Any]:
    """Spin a Qdrant container at a given version, yield it, stop it.

    Parametrized indirectly: the test sets the version via
    ``@pytest.mark.parametrize("qdrant_at_version", _QDRANT_VERSIONS,
    indirect=True)``.
    """
    version: str = request.param
    container = QdrantContainer(image=f"qdrant/qdrant:v{version}")
    container.start()
    try:
        yield container
    finally:
        container.stop()


# ---------------------------------------------------------------------------
# Task 1 — per-version smoke (3 tests)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qdrant_at_version", _QDRANT_VERSIONS, indirect=True)
def test_qdrant_smoke_per_version(qdrant_at_version: Any) -> None:
    """Each pinned Qdrant version accepts adds + serves searches through
    :class:`QdrantBackend` in HTTP mode.
    """
    from soma.memory.backends.qdrant import QdrantBackend

    url = _qdrant_url(qdrant_at_version)
    collection = f"soma_smoke_{uuid.uuid4().hex[:8]}"
    backend = QdrantBackend(
        mode="http",
        dim=_DIM,
        url=url,
        collection=collection,
        recreate=True,
    )
    try:
        ids = [f"id-{i}" for i in range(10)]
        vecs = _rand_vectors(10, seed=1)
        backend.add(ids, vecs)
        assert backend.ntotal == 10
        hits = backend.search(vecs[0], k=3)
        assert len(hits) == 3
        # Cosine self-match is the strongest hit.
        assert hits[0][0] == "id-0"
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Task 2 — 3x3 cross-version snapshot/restore matrix (9 tests)
# ---------------------------------------------------------------------------
def _create_and_download_snapshot(
    url: str, collection: str, out_path: Path
) -> None:
    """Drive Qdrant's REST snapshot API to produce a portable file.

    Uses the documented endpoints from the Qdrant REST API:

    - ``POST /collections/{c}/snapshots`` — create a snapshot on the
      server. Returns a descriptor with the snapshot ``name``.
    - ``GET /collections/{c}/snapshots/{name}`` — stream the snapshot
      bytes back to the client.
    """
    resp = requests.post(f"{url}/collections/{collection}/snapshots", timeout=60)
    resp.raise_for_status()
    name = resp.json()["result"]["name"]
    download = requests.get(
        f"{url}/collections/{collection}/snapshots/{name}",
        stream=True,
        timeout=120,
    )
    download.raise_for_status()
    with out_path.open("wb") as fh:
        for chunk in download.iter_content(chunk_size=1 << 16):
            if chunk:
                fh.write(chunk)


def _upload_and_recover_snapshot(
    url: str, collection: str, snapshot_path: Path
) -> None:
    """Upload ``snapshot_path`` to the target Qdrant and recover into
    ``collection``.

    Uses ``POST /collections/{c}/snapshots/upload`` which accepts a
    multipart-encoded snapshot file and restores it in-place. This is
    the supported cross-version migration path per the Qdrant docs.
    """
    with snapshot_path.open("rb") as fh:
        files = {"snapshot": (snapshot_path.name, fh, "application/octet-stream")}
        resp = requests.post(
            f"{url}/collections/{collection}/snapshots/upload",
            files=files,
            params={"priority": "snapshot"},
            timeout=180,
        )
    resp.raise_for_status()


def _wait_collection_ready(url: str, collection: str, timeout: float = 30.0) -> None:
    """Poll ``GET /collections/{c}`` until status==green or timeout."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(f"{url}/collections/{collection}", timeout=10)
            if resp.status_code == 200:
                status = resp.json().get("result", {}).get("status")
                if status in ("green", "yellow"):
                    return
        except Exception as exc:  # pragma: no cover - transient
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(
        f"Collection {collection!r} at {url} not ready in {timeout}s "
        f"(last_err={last_err!r})"
    )


@pytest.mark.parametrize("tgt_version", _QDRANT_VERSIONS)
@pytest.mark.parametrize("src_version", _QDRANT_VERSIONS)
def test_snapshot_restore_cross_version(
    src_version: str,
    tgt_version: str,
    tmp_path: Path,
) -> None:
    """Cross-version snapshot export + restore preserves top-1 search.

    Strategy:
      1. Spin a Qdrant at ``src_version``, ingest 10 deterministic vectors
         through :class:`QdrantBackend`, remember the expected top-1 id.
      2. Drive Qdrant's REST snapshot API to download a portable
         snapshot file.
      3. Tear down the src container.
      4. Spin a Qdrant at ``tgt_version`` and upload+recover the snapshot
         into the same collection name.
      5. Open a fresh :class:`QdrantBackend` against the target, invoke
         its ``.restore`` to re-hydrate the id map from the recovered
         collection, and assert search still returns the known top-1.
    """
    from soma.memory.backends.qdrant import QdrantBackend

    collection = f"xver_{uuid.uuid4().hex[:8]}"
    snapshot_path = tmp_path / "snapshot.bin"

    # The known-similar query is vector 0 itself — cosine self-match is
    # deterministic even across HNSW rebuilds, so we can assert an exact
    # top-1 id regardless of how the target version rebuilds its index.
    ids = [f"node-{i}" for i in range(10)]
    vecs = _rand_vectors(10, seed=42)
    expected_top = ids[0]

    # --- Phase 1: src ---
    src = QdrantContainer(image=f"qdrant/qdrant:v{src_version}")
    src.start()
    try:
        src_url = _qdrant_url(src)
        src_backend = QdrantBackend(
            mode="http",
            dim=_DIM,
            url=src_url,
            collection=collection,
            recreate=True,
        )
        try:
            src_backend.add(ids, vecs)
            assert src_backend.ntotal == 10
            # Also snapshot through the adapter — verifies the sidecar
            # metadata flows unchanged. The cross-version data transfer
            # itself goes through Qdrant's native REST snapshot API below.
            src_backend.snapshot(tmp_path / "bundle")
        finally:
            src_backend.close()
        _create_and_download_snapshot(src_url, collection, snapshot_path)
        assert snapshot_path.exists()
        assert snapshot_path.stat().st_size > 0
    finally:
        src.stop()

    # --- Phase 2: tgt ---
    tgt = QdrantContainer(image=f"qdrant/qdrant:v{tgt_version}")
    tgt.start()
    try:
        tgt_url = _qdrant_url(tgt)
        _upload_and_recover_snapshot(tgt_url, collection, snapshot_path)
        _wait_collection_ready(tgt_url, collection)

        tgt_backend = QdrantBackend(
            mode="http",
            dim=_DIM,
            url=tgt_url,
            collection=collection,
        )
        try:
            # ``restore`` re-reads the bundle's backend.json sidecar
            # written in phase 1 so the adapter's view of mode/url/
            # collection matches what's actually on the target.
            tgt_backend.restore(tmp_path / "bundle")
            # Even if the sidecar were absent, open() would have
            # rehydrated the id map from the recovered collection.
            assert tgt_backend.ntotal == 10, (
                f"cross-version restore lost rows: src={src_version} "
                f"tgt={tgt_version} ntotal={tgt_backend.ntotal}"
            )
            hits = tgt_backend.search(vecs[0], k=3)
            assert len(hits) == 3
            assert hits[0][0] == expected_top, (
                f"cross-version restore changed top-1: src={src_version} "
                f"tgt={tgt_version} got={hits[0][0]!r} want={expected_top!r}"
            )
            # Score should be near 1.0 for a cosine self-match.
            assert hits[0][1] > 0.99, (
                f"cross-version restore degraded top-1 score: "
                f"src={src_version} tgt={tgt_version} score={hits[0][1]}"
            )
        finally:
            tgt_backend.close()
    finally:
        tgt.stop()
