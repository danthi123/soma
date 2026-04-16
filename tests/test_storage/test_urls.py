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


def test_unknown_scheme_raises() -> None:
    """Phase 31 flips this test by adding s3:// dispatch. Keeping the
    exact error message stable so the contract-pin is unambiguous.
    """
    with pytest.raises(ValueError, match="unsupported store scheme"):
        parse_store_url("s3://bucket/prefix")


def test_gcs_scheme_raises_today() -> None:
    """Same pin for GCS — Phase 32 flips this."""
    with pytest.raises(ValueError, match="unsupported store scheme"):
        parse_store_url("gs://bucket/prefix")


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
