"""Gated real-GCS integration tests for :class:`GCSObjectStore`.

Phase 32 — mirrors the Phase 24 Qdrant / Phase 29 pgvector / Phase 31
S3 integration gating pattern. The default ``pytest`` run skips these
entirely; to exercise them locally::

    pip install -e ".[gcs]"
    # Any of: set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json,
    # or run `gcloud auth application-default login`,
    # or (on GCE/GKE/Cloud Run) rely on the metadata server.
    export SOMA_GCS_INTEGRATION_BUCKET=my-sandbox-bucket
    pytest tests/test_storage/test_gcs_integration.py -q

Optional env vars::

    export SOMA_GCS_INTEGRATION_PROJECT=my-project
    export SOMA_GCS_INTEGRATION_PREFIX=soma-ci/some-unique-tag

The tests exercise the round-trip shapes that the emulator can't
easily fake:
* real PUT atomicity under network retries,
* server-side pagination past 1000 objects,
* streaming put_stream against live network throughput,
* live ADC credential resolution.

Each test scopes itself to a unique per-run key prefix
(``SOMA_GCS_INTEGRATION_PREFIX`` or an auto-generated uuid) so
concurrent runs from different branches don't collide. Cleanup runs
in a fixture teardown regardless of test outcome.
"""

from __future__ import annotations

import contextlib
import io
import os
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soma.storage.gcs import GCSObjectStore

# ---------------------------------------------------------------------------
# Import guards — this module must be importable without
# google-cloud-storage so the default pytest run (which skips it
# entirely) doesn't crash at collection time.
# ---------------------------------------------------------------------------
try:
    from google.cloud import storage as _gcs  # noqa: F401

    _HAS_GCS = True
except ImportError:  # pragma: no cover - env-dependent
    _HAS_GCS = False


_BUCKET_ENV = "SOMA_GCS_INTEGRATION_BUCKET"
_PROJECT_ENV = "SOMA_GCS_INTEGRATION_PROJECT"
_PREFIX_ENV = "SOMA_GCS_INTEGRATION_PREFIX"


def _bucket() -> str | None:
    return os.environ.get(_BUCKET_ENV)


# Module-level guard. If google-cloud-storage is missing or the gate
# env isn't set, skip collection entirely so this file is a no-op in
# default runs.
if not _HAS_GCS:
    pytestmark = pytest.mark.skip(reason="google-cloud-storage not installed")
elif not _bucket():
    pytestmark = pytest.mark.skip(
        reason=f"set {_BUCKET_ENV}=<bucket> to run the real-GCS integration tests"
    )
else:
    pytestmark = pytest.mark.slow_gcs


@pytest.fixture
def gcs_store() -> Iterator[GCSObjectStore]:  # type: ignore[name-defined]  # noqa: F821
    """Yield a live :class:`GCSObjectStore` scoped to a unique prefix.

    Cleans up every object under the prefix in teardown regardless of
    test outcome — paranoid about bucket clutter on real GCS (and the
    storage bill that implies).
    """
    from soma.storage.gcs import GCSObjectStore

    bucket = _bucket()
    assert bucket is not None  # guarded by the module-level skip
    base = os.environ.get(_PREFIX_ENV, f"soma-ci/{uuid.uuid4().hex}")
    store = GCSObjectStore(
        bucket=bucket,
        prefix=base,
        project=os.environ.get(_PROJECT_ENV),
    )
    try:
        yield store
    finally:
        # Delete everything under our per-run prefix. ``list_prefix``
        # paginates, so we don't stop at 1000 even if a test left that
        # many objects behind.
        for key in list(store.list_prefix("")):
            with contextlib.suppress(Exception):  # pragma: no cover - best-effort
                store.delete(key)
        store.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_put_get_round_trip(gcs_store: GCSObjectStore) -> None:  # noqa: F821
    """Smoke: put a small payload, read it back. Proves ADC credentials +
    bucket access + the adapter wiring works end-to-end against real
    GCS."""
    gcs_store.put_bytes("smoke.bin", b"hello, gcs")
    assert gcs_store.get_bytes("smoke.bin") == b"hello, gcs"


def test_exists_and_delete(gcs_store: GCSObjectStore) -> None:  # noqa: F821
    """``exists`` and ``delete`` semantics against a live GCS — crucially,
    the adapter's idempotent-delete contract (swallowing NotFound on
    missing keys) needs real-GCS exercise to catch any subtle 404 vs
    403 confusion against IAM-restricted buckets."""
    assert gcs_store.exists("never.bin") is False
    gcs_store.put_bytes("here.bin", b"x")
    assert gcs_store.exists("here.bin") is True
    gcs_store.delete("here.bin")
    assert gcs_store.exists("here.bin") is False
    # Idempotent — no error on second delete.
    gcs_store.delete("here.bin")


def test_stream_put_large(gcs_store: GCSObjectStore) -> None:  # noqa: F821
    """Streaming upload of a 2 MB payload. Real-GCS exercises the
    resumable-upload machinery that the emulator fakes trivially; if
    our ``put_stream`` is wrong against live GCS this is where it
    surfaces."""
    payload = b"\x42" * (2 * 1024 * 1024)
    gcs_store.put_stream("big.bin", io.BytesIO(payload))
    assert gcs_store.get_bytes("big.bin") == payload


def test_list_prefix_paginates_past_1000(gcs_store: GCSObjectStore) -> None:  # noqa: F821
    """Real GCS caps list_blobs at 1000 items per page by default. We
    put 1020 and verify the auto-paginating iterator returns all of
    them."""
    # Keep this light — 1020 is enough to force a page break without
    # burning budget on API calls. Tests that want a bigger scale
    # should bump via SOMA_GCS_INTEGRATION_SCALE=.
    target = int(os.environ.get("SOMA_GCS_INTEGRATION_SCALE", "1020"))
    for i in range(target):
        gcs_store.put_bytes(f"bulk/{i:04d}.bin", b"x")
    keys = list(gcs_store.list_prefix("bulk/"))
    assert len(keys) == target


def test_local_root_round_trip(gcs_store: GCSObjectStore) -> None:  # noqa: F821
    """``local_root`` download-on-open, upload-on-close against live
    GCS. The flow MemoryLayer.load relies on when it needs the WAL
    sidecar + ``bundle.lock`` as real files."""
    gcs_store.put_bytes("seed.bin", b"seed")
    root = gcs_store.local_root
    assert (root / "seed.bin").read_bytes() == b"seed"
    # Mutate.
    (root / "new.bin").write_bytes(b"new")
    (root / "seed.bin").write_bytes(b"modified")
    gcs_store.close()
    # Re-open a fresh store to read back.
    from soma.storage.gcs import GCSObjectStore

    after = GCSObjectStore(
        bucket=gcs_store._bucket_name,
        prefix=gcs_store._prefix,
        project=os.environ.get(_PROJECT_ENV),
    )
    try:
        assert after.get_bytes("new.bin") == b"new"
        assert after.get_bytes("seed.bin") == b"modified"
    finally:
        # Best-effort cleanup of the objects this test dropped.
        for key in ("new.bin", "seed.bin"):
            with contextlib.suppress(Exception):  # pragma: no cover
                after.delete(key)
        after.close()
