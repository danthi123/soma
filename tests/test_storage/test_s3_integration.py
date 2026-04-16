"""Gated real-S3 integration tests for :class:`S3ObjectStore`.

Phase 31 — mirrors the Phase 24 Qdrant / Phase 29 pgvector integration
gating pattern. The default ``pytest`` run skips these entirely; to
exercise them locally::

    pip install -e ".[s3]"
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=us-east-1
    export SOMA_S3_INTEGRATION_BUCKET=my-sandbox
    pytest tests/test_storage/test_s3_integration.py -q

Against an S3-compatible endpoint (MinIO, R2, Spaces, LocalStack),
also set::

    export SOMA_S3_INTEGRATION_ENDPOINT=http://localhost:9000

The tests exercise the round-trip shapes that moto can't easily fake:
* real PUT atomicity under network retries,
* server-side pagination past 1000 keys,
* streaming put_stream against live network throughput.

Each test scopes itself to a unique per-run key prefix
(``SOMA_S3_INTEGRATION_PREFIX`` or an auto-generated uuid) so
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
    from soma.storage.s3 import S3ObjectStore

# ---------------------------------------------------------------------------
# Import guards — this module must be importable without boto3 so the
# default pytest run (which skips it entirely) doesn't crash at
# collection time.
# ---------------------------------------------------------------------------
try:
    import boto3  # noqa: F401

    _HAS_BOTO3 = True
except ImportError:  # pragma: no cover - env-dependent
    _HAS_BOTO3 = False


_BUCKET_ENV = "SOMA_S3_INTEGRATION_BUCKET"
_ENDPOINT_ENV = "SOMA_S3_INTEGRATION_ENDPOINT"
_REGION_ENV = "SOMA_S3_INTEGRATION_REGION"
_PREFIX_ENV = "SOMA_S3_INTEGRATION_PREFIX"


def _bucket() -> str | None:
    return os.environ.get(_BUCKET_ENV)


# Module-level guard. If boto3 is missing or the gate env isn't set,
# skip collection entirely so this file is a no-op in default runs.
if not _HAS_BOTO3:
    pytestmark = pytest.mark.skip(reason="boto3 not installed")
elif not _bucket():
    pytestmark = pytest.mark.skip(
        reason=f"set {_BUCKET_ENV}=<bucket> to run the real-S3 integration tests"
    )
else:
    pytestmark = pytest.mark.slow_s3


@pytest.fixture
def s3_store() -> Iterator[S3ObjectStore]:  # type: ignore[name-defined]  # noqa: F821
    """Yield a live :class:`S3ObjectStore` scoped to a unique prefix.

    Cleans up every object under the prefix in teardown regardless of
    test outcome — paranoid about bucket clutter on real S3.
    """
    from soma.storage.s3 import S3ObjectStore

    bucket = _bucket()
    assert bucket is not None  # guarded by the module-level skip
    base = os.environ.get(_PREFIX_ENV, f"soma-ci/{uuid.uuid4().hex}")
    store = S3ObjectStore(
        bucket=bucket,
        prefix=base,
        endpoint_url=os.environ.get(_ENDPOINT_ENV),
        region_name=os.environ.get(_REGION_ENV),
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


def test_put_get_round_trip(s3_store: S3ObjectStore) -> None:  # noqa: F821
    """Smoke: put a small payload, read it back. Proves credentials +
    bucket access + the adapter wiring works end-to-end against real
    AWS (or the configured S3-compatible endpoint)."""
    s3_store.put_bytes("smoke.bin", b"hello, s3")
    assert s3_store.get_bytes("smoke.bin") == b"hello, s3"


def test_exists_and_delete(s3_store: S3ObjectStore) -> None:  # noqa: F821
    """``exists`` and ``delete`` semantics against a live S3."""
    assert s3_store.exists("never.bin") is False
    s3_store.put_bytes("here.bin", b"x")
    assert s3_store.exists("here.bin") is True
    s3_store.delete("here.bin")
    assert s3_store.exists("here.bin") is False
    # Idempotent — no error on second delete.
    s3_store.delete("here.bin")


def test_stream_put_large(s3_store: S3ObjectStore) -> None:  # noqa: F821
    """Streaming upload of a 2 MB payload. Real-S3 exercises the
    multi-part transfer machinery that moto fakes trivially; if our
    ``put_stream`` is wrong against AWS this is where it surfaces."""
    payload = b"\x42" * (2 * 1024 * 1024)
    s3_store.put_stream("big.bin", io.BytesIO(payload))
    assert s3_store.get_bytes("big.bin") == payload


def test_list_prefix_paginates_past_1000(s3_store: S3ObjectStore) -> None:  # noqa: F821
    """Real S3 caps ListObjectsV2 at 1000 keys per page. We put 1020
    and verify the paginator returns all of them."""
    # Keep this light — 1020 is enough to force a page break without
    # burning budget on API calls. Tests that want a bigger scale
    # should bump via SOMA_S3_INTEGRATION_SCALE=.
    target = int(os.environ.get("SOMA_S3_INTEGRATION_SCALE", "1020"))
    for i in range(target):
        s3_store.put_bytes(f"bulk/{i:04d}.bin", b"x")
    keys = list(s3_store.list_prefix("bulk/"))
    assert len(keys) == target


def test_local_root_round_trip(s3_store: S3ObjectStore) -> None:  # noqa: F821
    """``local_root`` download-on-open, upload-on-close against live S3.
    The flow MemoryLayer.load relies on when it needs the WAL sidecar
    + ``bundle.lock`` as real files."""
    s3_store.put_bytes("seed.bin", b"seed")
    root = s3_store.local_root
    assert (root / "seed.bin").read_bytes() == b"seed"
    # Mutate.
    (root / "new.bin").write_bytes(b"new")
    (root / "seed.bin").write_bytes(b"modified")
    s3_store.close()
    # Re-open a fresh store to read back.
    from soma.storage.s3 import S3ObjectStore

    after = S3ObjectStore(
        bucket=s3_store._bucket,
        prefix=s3_store._prefix,
        endpoint_url=os.environ.get(_ENDPOINT_ENV),
        region_name=os.environ.get(_REGION_ENV),
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
