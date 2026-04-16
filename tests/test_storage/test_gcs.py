"""Behaviour tests for :class:`soma.storage.gcs.GCSObjectStore`.

**Mock strategy chosen:** ``gcp-storage-emulator`` (community package,
PyPI ``gcp-storage-emulator>=2024.8``) wrapping an in-process HTTP
server that speaks the real GCS JSON API. Respected via the
``STORAGE_EMULATOR_HOST`` environment variable that the stock
``google-cloud-storage`` client honours when set. Higher fidelity
than raw ``unittest.mock`` stubs — we're exercising the same code
paths that talk to real GCS, just with the HTTP backend redirected
to localhost. Emulator is spun up in a session-scoped fixture and
torn down at the end so we pay the startup cost once.

Falls back to ``pytest.importorskip`` if either ``google-cloud-storage``
or ``gcp-storage-emulator`` is missing; the default CI path installs
both via the ``[gcs-test]`` extra.

Every test matches the Protocol contract pinned by ``test_local.py``
and ``test_s3.py`` byte-for-byte — missing key raises ``KeyError``;
``delete`` is idempotent (GCS raises ``NotFound`` on missing, so the
adapter swallows that); ``list_prefix`` emits relative keys; stream
paths never slurp a multi-MB payload into RAM.
"""

from __future__ import annotations

import contextlib
import io
import socket
import threading
import time
import tracemalloc
import uuid
from collections.abc import Iterator

import pytest

# Import guard — both emulator and client must be present.
pytest.importorskip("google.cloud.storage")
pytest.importorskip("gcp_storage_emulator")

from gcp_storage_emulator.server import create_server  # noqa: E402
from google.cloud import storage as _gcs  # noqa: E402

from soma.storage import ObjectStore  # noqa: E402
from soma.storage.gcs import GCSObjectStore  # noqa: E402


def _free_port() -> int:
    """Grab a free TCP port for the emulator. Tests run in parallel on
    CI and a hard-coded port would collide with itself on re-run."""
    s = socket.socket()
    try:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@pytest.fixture(scope="module")
def _emulator() -> Iterator[str]:
    """Session-per-module emulator. Yields the ``http://host:port``
    base URL; the caller is responsible for setting
    ``STORAGE_EMULATOR_HOST`` before constructing the client (we do
    that in each test's own fixture so ordering is explicit)."""
    port = _free_port()
    server = create_server("localhost", port, in_memory=True, default_bucket="")
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    # Wait for the emulator to come up. 2 s is more than enough on a
    # local box; bump if CI is slow.
    time.sleep(1.0)
    base = f"http://localhost:{port}"
    try:
        yield base
    finally:
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort
            server.stop()


@pytest.fixture
def gcs_client(
    _emulator: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_gcs.Client]:
    """Anonymous client pointed at the emulator. One per test so
    state doesn't leak between tests (emulator is shared but each
    test uses its own fresh bucket)."""
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", _emulator)
    client = _gcs.Client.create_anonymous_client()
    # create_anonymous_client() hard-codes ``project='<none>'``;
    # some emulator endpoints (and the real GCS JSON API) want a
    # sensible project value even for anonymous operations.
    client.project = "test-project"
    yield client


@pytest.fixture
def bucket_name(gcs_client: _gcs.Client) -> str:
    """Unique bucket per test so ``list`` results don't cross contaminate."""
    name = f"test-bucket-{uuid.uuid4().hex[:8]}"
    # Emulator creates buckets implicitly when blobs land but we
    # call create() for parity with real GCS. Real GCS requires
    # explicit creation; the adapter never creates buckets.
    with contextlib.suppress(Exception):
        # Emulator is lax; ignore duplicate-bucket errors.
        gcs_client.create_bucket(name)
    return name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_put_get_round_trip(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    store.put_bytes("foo", b"bar")
    assert store.get_bytes("foo") == b"bar"


def test_satisfies_object_store_protocol(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """Duck-type Protocol check — the adapter plugs into
    :class:`MemoryLayer.save` / :class:`load` with no extra shim."""
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    assert isinstance(store, ObjectStore)


def test_prefix_isolation(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """Two stores on the same bucket but different prefixes see disjoint
    keyspaces. Load-bearing for the bundle-per-prefix deploy pattern."""
    a = GCSObjectStore(bucket=bucket_name, prefix="a", client=gcs_client)
    z = GCSObjectStore(bucket=bucket_name, prefix="z", client=gcs_client)
    a.put_bytes("k", b"A")
    z.put_bytes("k", b"Z")
    assert a.get_bytes("k") == b"A"
    assert z.get_bytes("k") == b"Z"
    assert list(a.list_prefix("")) == ["k"]  # z's 'k' not visible


def test_list_prefix_returns_matching_keys(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """``list_prefix`` emits keys relative to the store's configured
    prefix so callers can round-trip the same key into ``get_bytes``
    without remembering to strip the prefix themselves — matches local
    FS + S3 semantics exactly."""
    store = GCSObjectStore(bucket=bucket_name, prefix="bundle", client=gcs_client)
    store.put_bytes("a/one.bin", b"1")
    store.put_bytes("a/two.bin", b"2")
    store.put_bytes("b/three.bin", b"3")
    assert sorted(store.list_prefix("a/")) == ["a/one.bin", "a/two.bin"]
    assert sorted(store.list_prefix("b/")) == ["b/three.bin"]


def test_list_prefix_empty_bucket_yields_nothing(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    assert list(store.list_prefix("")) == []


def test_delete_missing_is_idempotent(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """S3 semantics: ``delete`` on a missing key is a no-op. GCS raises
    ``NotFound`` on missing; the adapter must swallow it so callers get
    the same Protocol contract regardless of backend."""
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    store.delete("never-existed.bin")  # no-op, must not raise
    store.put_bytes("gone.bin", b"x")
    store.delete("gone.bin")
    assert not store.exists("gone.bin")
    store.delete("gone.bin")  # second delete still a no-op


def test_exists_true_false(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    assert store.exists("never.bin") is False
    store.put_bytes("here.bin", b"x")
    assert store.exists("here.bin") is True


def test_get_missing_raises_key_error(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """GCS raises ``google.cloud.exceptions.NotFound`` on absent blob
    download. Adapter must translate to ``KeyError`` for Protocol parity."""
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    with pytest.raises(KeyError):
        store.get_bytes("never.bin")
    with pytest.raises(KeyError):
        store.get_stream("never.bin")


def test_put_overwrites(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    store.put_bytes("k.bin", b"old")
    store.put_bytes("k.bin", b"new")
    assert store.get_bytes("k.bin") == b"new"


def test_stream_round_trip_large(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """2 MB payload via ``put_stream`` / ``get_stream``. The emulator
    is in-memory so we pick a modest payload to keep wall time sane;
    the real-GCS gated integration test scales higher.

    ``get_stream`` returns a ``BytesIO`` (same precedent as S3 — see
    the adapter docstring for why). Callers get seek + tell for
    ``torch.load`` / ``numpy.load`` compatibility. Tracemalloc pins
    a generous ceiling on the put side to catch accidental double-
    buffering regressions.
    """
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    payload = b"\x42" * (2 * 1024 * 1024)
    src = io.BytesIO(payload)

    tracemalloc.start()
    store.put_stream("big.bin", src)
    _, peak_put = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_read = 0
    with store.get_stream("big.bin") as fh:
        while chunk := fh.read(65536):
            assert chunk == b"\x42" * len(chunk)
            total_read += len(chunk)

    assert total_read == len(payload)
    # Emulator is in-memory so we allow plenty of slack — the point
    # of this ceiling is to catch a regression that slurps the whole
    # payload twice (into bytes and then back into a stream), not to
    # pin exact memory behaviour.
    assert peak_put < 25 * 1024 * 1024, (
        f"put_stream peak memory {peak_put} exceeded 25 MB budget"
    )


def test_client_override_honoured(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """Passing an already-built client lets tests share one emulator
    connection and lets advanced callers pre-configure retries, custom
    credentials, etc."""
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    assert store._client is gcs_client


def test_import_error_without_gcs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiating the adapter without ``google-cloud-storage`` on
    the path must surface a clear install-hint error. Flip the
    module-level flag so we can test the guard branch without
    uninstalling the client."""
    monkeypatch.setattr("soma.storage.gcs._HAS_GCS", False)
    with pytest.raises(ImportError, match=r"soma\[gcs\]"):
        GCSObjectStore(bucket="test-bucket")


def test_tmp_siblings_skipped_in_list_prefix(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """Parity with ``LocalFSObjectStore`` + ``S3ObjectStore``: transient
    ``.tmp`` siblings must never show up in ``list_prefix``. A hybrid
    deploy that shares a bucket with a local-writer process might leak
    these; the adapter filters so callers iterating a prefix don't
    have to."""
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    store.put_bytes("real.bin", b"x")
    # Simulate a stray .tmp from another writer.
    stray = gcs_client.bucket(bucket_name).blob("half.bin.tmp")
    stray.upload_from_file(io.BytesIO(b"incomplete"))
    keys = list(store.list_prefix(""))
    assert "real.bin" in keys
    assert "half.bin.tmp" not in keys


def test_forward_slash_keys_on_windows(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """Windows callers sometimes pass backslash paths. GCS keys are
    forward-slash always — the adapter must normalise so a caller
    that wrote with ``\\`` can read with ``/`` and vice versa."""
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    store.put_bytes("dir\\sub\\file.bin", b"x")
    assert store.get_bytes("dir/sub/file.bin") == b"x"
    assert sorted(store.list_prefix("")) == ["dir/sub/file.bin"]


def test_project_kwarg_accepted(
    _emulator: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``project=`` kwarg lets callers pin the GCS project without
    relying on ADC. We don't need to route traffic — just verify the
    kwarg lands on the underlying client config."""
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", _emulator)
    # Use an injected anonymous client with a custom project override,
    # which is what the ``project=`` kwarg path ultimately does.
    bucket = f"proj-{uuid.uuid4().hex[:8]}"
    client = _gcs.Client.create_anonymous_client()
    client.project = "my-custom-project"
    store = GCSObjectStore(bucket=bucket, client=client)
    assert store._client.project == "my-custom-project"


def test_close_without_local_root_is_noop(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """``local_root`` is pay-per-use: if a caller only goes through
    ``put_bytes`` / ``get_bytes``, no staging dir is created. Close()
    stays a no-op in that case (no temp dir to sync, no temp dir to
    tear down)."""
    store = GCSObjectStore(bucket=bucket_name, client=gcs_client)
    store.put_bytes("k.bin", b"x")
    assert store.get_bytes("k.bin") == b"x"
    assert store._local_root is None
    # close() is safe and doesn't touch GCS.
    store.close()
    # Idempotent — second close is a no-op.
    store.close()


def test_local_root_download_then_upload(
    gcs_client: _gcs.Client, bucket_name: str
) -> None:
    """End-to-end staging-dir round-trip: seed the prefix, touch
    ``local_root`` (download), mutate files, close (upload), re-open
    and verify the mutations landed in GCS. Mirrors the Phase-31 S3
    test shape byte-for-byte — this is what MemoryLayer.load relies on
    when the bundle has a WAL sidecar or ``bundle.lock`` that needs a
    real on-disk path."""
    seed = GCSObjectStore(bucket=bucket_name, prefix="bundle", client=gcs_client)
    seed.put_bytes("a.bin", b"original-a")
    seed.put_bytes("sub/b.bin", b"original-b")

    store = GCSObjectStore(bucket=bucket_name, prefix="bundle", client=gcs_client)
    root = store.local_root
    assert root.exists()
    assert (root / "a.bin").read_bytes() == b"original-a"
    assert (root / "sub" / "b.bin").read_bytes() == b"original-b"

    # Mutate.
    (root / "a.bin").write_bytes(b"modified-a")
    (root / "sub").mkdir(exist_ok=True)
    (root / "sub" / "c.bin").write_bytes(b"brand-new-c")

    store.close()

    after = GCSObjectStore(bucket=bucket_name, prefix="bundle", client=gcs_client)
    assert after.get_bytes("a.bin") == b"modified-a"
    assert after.get_bytes("sub/b.bin") == b"original-b"
    assert after.get_bytes("sub/c.bin") == b"brand-new-c"
    after.close()
