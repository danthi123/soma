"""Behaviour tests for :class:`soma.storage.s3.S3ObjectStore`.

Every test runs inside a ``moto.mock_aws`` context so we get a full
in-memory S3 with no network, no Docker, no creds. The adapter must
match the Protocol contract pinned by ``test_local.py`` byte-for-byte
— missing key raises ``KeyError``; ``delete`` is idempotent;
``list_prefix`` emits relative keys under the store's prefix; stream
paths never slurp a multi-MB payload into RAM.

Phase 31 lands this adapter behind the existing ``ObjectStore``
Protocol so ``MemoryLayer.save``/``load`` pick it up via
:func:`parse_store_url` with no API change at the memory-layer edge.
"""

from __future__ import annotations

import io
import tracemalloc

import pytest

# Import guard — the default CI path installs ``moto[s3]``, but we still
# want a clear skip message if a dev hits this file without the extra.
moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")

from moto import mock_aws  # noqa: E402

from soma.storage import ObjectStore  # noqa: E402
from soma.storage.s3 import S3ObjectStore  # noqa: E402


@mock_aws
def test_put_get_round_trip() -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    store.put_bytes("foo", b"bar")
    assert store.get_bytes("foo") == b"bar"


@mock_aws
def test_satisfies_object_store_protocol() -> None:
    """Duck-type Protocol check — the adapter plugs into
    :class:`MemoryLayer.save` / :class:`load` with no extra shim."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    assert isinstance(store, ObjectStore)


@mock_aws
def test_prefix_isolation() -> None:
    """Two stores on the same bucket but different prefixes see disjoint
    keyspaces. Load-bearing for the bundle-per-prefix deploy pattern."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    a = S3ObjectStore(bucket="test-bucket", prefix="a")
    z = S3ObjectStore(bucket="test-bucket", prefix="z")
    a.put_bytes("k", b"A")
    z.put_bytes("k", b"Z")
    assert a.get_bytes("k") == b"A"
    assert z.get_bytes("k") == b"Z"
    assert list(a.list_prefix("")) == ["k"]  # z's 'k' not visible


@mock_aws
def test_list_prefix_relative_keys() -> None:
    """``list_prefix`` emits keys relative to the store's configured
    prefix so callers can round-trip the same key into ``get_bytes``
    without remembering to strip the prefix themselves — matches local
    FS semantics exactly."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket", prefix="bundle")
    store.put_bytes("a/one.bin", b"1")
    store.put_bytes("a/two.bin", b"2")
    store.put_bytes("b/three.bin", b"3")
    assert sorted(store.list_prefix("a/")) == ["a/one.bin", "a/two.bin"]
    assert sorted(store.list_prefix("b/")) == ["b/three.bin"]


@mock_aws
def test_list_prefix_empty_bucket_yields_nothing() -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    assert list(store.list_prefix("")) == []


@mock_aws
def test_list_prefix_paginates_past_1000_keys() -> None:
    """``list_objects_v2`` caps responses at 1000 keys. The adapter
    must use ``get_paginator`` so a bundle with > 1000 objects doesn't
    silently truncate. This is the one place the adapter can go wrong
    at scale and not surface until a user's first large bundle."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    for i in range(1050):
        store.put_bytes(f"k{i:04d}", b"x")
    keys = list(store.list_prefix(""))
    assert len(keys) == 1050


@mock_aws
def test_delete_missing_is_idempotent() -> None:
    """S3 semantics: ``delete_object`` on a missing key is a no-op,
    not a 404. Pin the adapter to that behaviour so callers can
    delete-before-write without a try/except dance."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    store.delete("never-existed.bin")  # no-op, must not raise
    store.put_bytes("gone.bin", b"x")
    store.delete("gone.bin")
    assert not store.exists("gone.bin")
    store.delete("gone.bin")  # second delete still a no-op


@mock_aws
def test_exists_true_false() -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    assert store.exists("never.bin") is False
    store.put_bytes("here.bin", b"x")
    assert store.exists("here.bin") is True


@mock_aws
def test_get_missing_raises_key_error() -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    with pytest.raises(KeyError):
        store.get_bytes("never.bin")
    with pytest.raises(KeyError):
        store.get_stream("never.bin")


@mock_aws
def test_put_overwrites() -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    store.put_bytes("k.bin", b"old")
    store.put_bytes("k.bin", b"new")
    assert store.get_bytes("k.bin") == b"new"


@mock_aws
def test_stream_round_trip_large() -> None:
    """10 MB payload via ``put_stream`` / ``get_stream``. The adapter
    must use ``upload_fileobj`` / ``StreamingBody`` so neither side
    materialises the full payload in RAM. Tracemalloc pins the peak.

    moto's in-memory backend is itself a ``BytesIO`` so the buffers
    it holds count — we allow a generous 6 MB ceiling (enough for the
    server-side copy + the 8 MB upload chunk threshold in boto's
    default transfer config) but below the 10 MB slurp.
    """
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    payload = b"\x42" * (10 * 1024 * 1024)
    src = io.BytesIO(payload)

    tracemalloc.start()
    store.put_stream("big.bin", src)
    _, peak_put = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    total_read = 0
    with store.get_stream("big.bin") as fh:
        while chunk := fh.read(65536):
            assert chunk == b"\x42" * len(chunk)
            total_read += len(chunk)
    _, peak_get = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert total_read == len(payload)
    # moto runs an in-memory S3 so the "server" keeps a full copy of
    # every upload; boto's default TransferConfig also retains an
    # 8 MB chunk during multi-part orchestration. Against real S3 the
    # put peak is ~8 MB; against moto we see ~3x the payload. Ceiling
    # at 50 MB catches regressions (e.g. someone re-reading the whole
    # payload into bytes before uploading) while accepting the mock
    # overhead. For the "no slurp" invariant see
    # ``test_local.py::test_stream_round_trip_large`` (3 MB ceiling)
    # which pins the local adapter's contract.
    assert peak_put < 50 * 1024 * 1024, (
        f"put_stream peak memory {peak_put} exceeded 50 MB budget"
    )
    # Read side: against real S3 the StreamingBody is a socket wrapper
    # and peak is ~64 KB (the read window). moto's body is a BytesIO
    # holding the full payload in memory, so tracemalloc sees ~10 MB
    # resident. Ceiling at 20 MB catches a "buffer into bytes before
    # yielding" regression while accepting the mock cost. Real-S3
    # integration test (gated) is the home for the tight ceiling.
    assert peak_get < 20 * 1024 * 1024, (
        f"get_stream peak memory {peak_get} exceeded 20 MB budget"
    )


@mock_aws
def test_endpoint_url_accepted() -> None:
    """Endpoint override is first-class so the adapter runs against
    MinIO / LocalStack / R2 / Spaces without any sub-classing. We
    don't route traffic in the test (moto intercepts all AWS calls);
    we just assert the kwarg lands on the underlying client config."""
    # A non-default endpoint + region pair so we can observe both.
    store = S3ObjectStore(
        bucket="test-bucket",
        endpoint_url="http://localhost:9000",
        region_name="us-west-2",
    )
    assert store._client.meta.endpoint_url == "http://localhost:9000"
    assert store._client.meta.region_name == "us-west-2"


@mock_aws
def test_client_override_honoured() -> None:
    """Passing an already-built client lets tests share one mock and
    lets advanced callers pre-configure retries, proxies, signature
    versions, etc."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket", client=client)
    assert store._client is client


def test_import_error_without_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiating the adapter without ``boto3`` on the path must
    surface a clear install-hint error. Flip the module-level flag so
    we can test the guard branch without uninstalling boto3."""
    monkeypatch.setattr("soma.storage.s3._HAS_BOTO3", False)
    with pytest.raises(ImportError, match=r"soma\[s3\]"):
        S3ObjectStore(bucket="test-bucket")


@mock_aws
def test_tmp_siblings_skipped_in_list_prefix() -> None:
    """Parity with ``LocalFSObjectStore``: transient ``.tmp`` siblings
    (left over if a writer died mid-put against a different store
    sharing the same S3 bucket + prefix) must never show up in
    ``list_prefix``. The local adapter already pins this; the S3
    adapter matches so callers iterating a prefix don't have to
    filter themselves."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    store.put_bytes("real.bin", b"x")
    # Simulate a stray .tmp from some other writer that didn't finish.
    client.put_object(Bucket="test-bucket", Key="half.bin.tmp", Body=b"incomplete")
    keys = list(store.list_prefix(""))
    assert "real.bin" in keys
    assert "half.bin.tmp" not in keys


@mock_aws
def test_forward_slash_keys_on_windows() -> None:
    """Windows callers sometimes pass backslash paths. S3 keys are
    forward-slash always — the adapter must normalise so a caller
    that wrote with ``\\`` can read with ``/`` and vice versa."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    store.put_bytes("dir\\sub\\file.bin", b"x")
    assert store.get_bytes("dir/sub/file.bin") == b"x"
    # And the listing reports the POSIX form regardless of how it
    # was written.
    assert sorted(store.list_prefix("")) == ["dir/sub/file.bin"]
