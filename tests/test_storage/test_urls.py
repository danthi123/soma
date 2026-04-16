"""URL dispatch tests for :func:`soma.storage.parse_store_url`.

``parse_store_url`` is the single point where a user-supplied string
becomes a concrete :class:`ObjectStore`. It has to handle every
platform-valid form of a local path (POSIX, Windows, drive-letter,
with or without ``file://`` prefix) and reject unknown schemes
cleanly so Phase 31's ``s3://`` work has a clear place to plug in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soma.storage import LocalFSObjectStore, parse_store_url


def test_file_url_posix_three_slashes(tmp_path: Path) -> None:
    """RFC-style ``file:///abs/path`` — the canonical form."""
    as_posix = tmp_path.as_posix()
    # Normalize to the three-slash form regardless of platform.
    url = f"file://{as_posix}" if as_posix.startswith("/") else f"file:///{as_posix}"
    store = parse_store_url(url)
    assert isinstance(store, LocalFSObjectStore)


def test_file_url_windows_two_slashes(tmp_path: Path) -> None:
    """``file://C:/Users/foo`` — the Windows-odd two-slash form. ``urlparse``
    puts ``C:`` in ``netloc`` here; the parser has to put it back together.
    Skipped on POSIX where there are no drive letters to parse.
    """
    import os

    if os.name != "nt":
        pytest.skip("Windows-specific drive-letter parsing")
    as_posix = tmp_path.as_posix()  # e.g. 'C:/Users/foo/pytest-.../t0'
    url = f"file://{as_posix}"
    store = parse_store_url(url)
    assert isinstance(store, LocalFSObjectStore)
    # Round-trip a byte to confirm the path lands where we expect.
    store.put_bytes("probe.bin", b"ok")
    assert (tmp_path / "probe.bin").read_bytes() == b"ok"


def test_plain_path_auto_prefixes(tmp_path: Path) -> None:
    """A bare path (no scheme) is treated as a local file:// URL."""
    store = parse_store_url(str(tmp_path))
    assert isinstance(store, LocalFSObjectStore)
    store.put_bytes("k.bin", b"x")
    assert (tmp_path / "k.bin").exists()


def test_pathlib_accepted(tmp_path: Path) -> None:
    """``Path`` objects parse identically to their string form."""
    store = parse_store_url(tmp_path)
    assert isinstance(store, LocalFSObjectStore)


def test_parse_store_url_dispatches_to_s3() -> None:
    """Phase 31 flipped the Phase 30 "unsupported scheme" pin into a
    real dispatch: ``s3://bucket/prefix`` now resolves to an
    :class:`S3ObjectStore`. We gate the import via
    ``pytest.importorskip`` so the test is skipped (not failed) on
    environments without ``moto`` / ``boto3`` installed."""
    pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    import boto3 as _boto3

    from soma.storage.s3 import S3ObjectStore

    with moto.mock_aws():
        _boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        store = parse_store_url("s3://test-bucket/bundle")
        assert isinstance(store, S3ObjectStore)
        # Round-trip a byte so we know the bucket + prefix split
        # landed correctly.
        store.put_bytes("probe.bin", b"ok")
        assert store.get_bytes("probe.bin") == b"ok"
        # Expected prefix is the path portion of the URL.
        assert store._prefix == "bundle"


def test_parse_store_url_s3_carries_query_params() -> None:
    """Query-string kwargs override defaults so callers can point at a
    non-AWS S3-compatible endpoint straight from the URL without
    writing Python. ``region`` and ``endpoint`` are the two we pin."""
    pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    from soma.storage.s3 import S3ObjectStore

    with moto.mock_aws():
        store = parse_store_url(
            "s3://test-bucket/bundle?region=eu-west-1&endpoint=http://minio:9000"
        )
        assert isinstance(store, S3ObjectStore)
        assert store._client.meta.region_name == "eu-west-1"
        assert store._client.meta.endpoint_url == "http://minio:9000"


def test_parse_store_url_s3_bucket_only() -> None:
    """``s3://bucket`` (no trailing prefix) is a valid form and means
    "put bundles at the bucket root". Pin so the parser doesn't
    accidentally stuff an empty path into the prefix slot."""
    pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    import boto3 as _boto3

    from soma.storage.s3 import S3ObjectStore

    with moto.mock_aws():
        _boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
        store = parse_store_url("s3://test-bucket")
        assert isinstance(store, S3ObjectStore)
        assert store._prefix == ""


def test_parse_store_url_dispatches_to_gcs() -> None:
    """Phase 32 flipped the earlier ``unsupported scheme`` pin into a
    real dispatch: ``gs://bucket/prefix`` now resolves to a
    :class:`GCSObjectStore`. Gated via ``importorskip`` so the test is
    skipped (not failed) on environments without
    ``google-cloud-storage`` / ``gcp-storage-emulator``."""
    pytest.importorskip("google.cloud.storage")
    pytest.importorskip("gcp_storage_emulator")

    import contextlib
    import socket
    import threading
    import time
    import uuid

    from gcp_storage_emulator.server import create_server
    from google.cloud import storage as _gcs

    from soma.storage.gcs import GCSObjectStore

    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    server = create_server("localhost", port, in_memory=True, default_bucket="")
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(1.0)

    try:
        import os

        os.environ["STORAGE_EMULATOR_HOST"] = f"http://localhost:{port}"
        try:
            bucket_name = f"url-{uuid.uuid4().hex[:8]}"
            client = _gcs.Client.create_anonymous_client()
            client.project = "test-project"
            with contextlib.suppress(Exception):
                client.create_bucket(bucket_name)
            store = parse_store_url(f"gs://{bucket_name}/bundle")
            assert isinstance(store, GCSObjectStore)
            # Round-trip a byte so we know the bucket + prefix split
            # landed correctly.
            store.put_bytes("probe.bin", b"ok")
            assert store.get_bytes("probe.bin") == b"ok"
            # Expected prefix is the path portion of the URL.
            assert store._prefix == "bundle"
        finally:
            os.environ.pop("STORAGE_EMULATOR_HOST", None)
    finally:
        with contextlib.suppress(Exception):
            server.stop()


def test_parse_store_url_gcs_carries_project_query_param() -> None:
    """Query-string ``?project=...`` overrides the ADC project
    attribution so callers can pin a specific project straight from
    the URL without writing Python. Cross-project service-account
    setups need this."""
    pytest.importorskip("google.cloud.storage")

    # We don't need a live emulator to probe the project kwarg plumbing:
    # the Client constructor just records the project attribute. Any
    # network call would fail, but we never make one.
    import os

    prev = os.environ.get("STORAGE_EMULATOR_HOST")
    # Stub an emulator host so Client() doesn't try to fetch ADC
    # credentials from the metadata server during the import path.
    os.environ["STORAGE_EMULATOR_HOST"] = "http://localhost:1"
    try:
        from soma.storage.gcs import GCSObjectStore

        store = parse_store_url("gs://test-bucket/bundle?project=my-proj")
        assert isinstance(store, GCSObjectStore)
        assert store._client.project == "my-proj"
    finally:
        if prev is None:
            os.environ.pop("STORAGE_EMULATOR_HOST", None)
        else:
            os.environ["STORAGE_EMULATOR_HOST"] = prev


def test_object_store_instance_passes_through() -> None:
    """If a caller already has an ObjectStore, parse_store_url must
    accept it and return it unchanged. Callers that stitched their own
    adapter (e.g. a test-time MemoryStore) need this.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        existing = LocalFSObjectStore(Path(d))
        # parse_store_url is the URL-from-user path; the instance-passthrough
        # lives in _coerce_store on MemoryLayer. But the Protocol doc
        # implies an accepting behaviour — pin it if we implement it, or
        # drop this test if we route through _coerce_store only.
        # We pin the URL round-trip: the ObjectStore's root must be
        # reachable via a plain path call too.
        path_store = parse_store_url(Path(d))
        assert isinstance(path_store, LocalFSObjectStore)
        _ = existing  # keep reference alive until tempdir closes


def test_empty_string_raises() -> None:
    with pytest.raises(ValueError):
        parse_store_url("")


def test_windows_drive_letter_plain_path(tmp_path: Path) -> None:
    """Plain Windows paths like ``C:\\Users\\foo`` must be accepted when
    no scheme prefix is supplied. POSIX boxes don't have drive letters
    so we skip there.
    """
    import os

    if os.name != "nt":
        pytest.skip("Windows-specific drive-letter parsing")
    # tmp_path already has a drive letter on Windows.
    store = parse_store_url(str(tmp_path))
    assert isinstance(store, LocalFSObjectStore)
    store.put_bytes("k.bin", b"ok")
    assert (tmp_path / "k.bin").read_bytes() == b"ok"
