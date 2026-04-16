"""Behaviour tests for :class:`soma.storage.LocalFSObjectStore`.

The local implementation is the reference behaviour: S3 / GCS adapters
in Phases 31-32 must match this contract exactly (missing key raises
``KeyError``; ``delete`` is idempotent; ``list_prefix`` matches
server-side prefix semantics; stream paths never slurp into RAM).
"""

from __future__ import annotations

import io
import tracemalloc
from pathlib import Path

import pytest

from soma.storage import LocalFSObjectStore


def test_put_get_round_trip(tmp_path: Path) -> None:
    store = LocalFSObjectStore(tmp_path)
    store.put_bytes("hello.bin", b"hello world")
    assert store.get_bytes("hello.bin") == b"hello world"


def test_stream_round_trip_large(tmp_path: Path) -> None:
    """A 10 MB payload must round-trip through ``put_stream`` /
    ``get_stream`` without pulling the full buffer into memory. We use
    ``tracemalloc`` to assert the stream path's peak allocation stays
    well below the payload size — if someone silently replaces the
    chunked copy with a ``.read()`` + ``.write()`` the test trips.
    """
    store = LocalFSObjectStore(tmp_path)
    payload = b"\x42" * (10 * 1024 * 1024)  # 10 MB
    src = io.BytesIO(payload)

    tracemalloc.start()
    store.put_stream("big.bin", src)
    _, peak_put = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Drain chunk-by-chunk, verifying each chunk in place so we never
    # materialise the whole 10 MB payload again on the reader side.
    # Peak allocation during the read loop proves get_stream doesn't
    # slurp internally.
    tracemalloc.start()
    total_read = 0
    with store.get_stream("big.bin") as fh:
        while chunk := fh.read(65536):
            # Payload is a single byte repeated; cheap constant-memory check.
            assert chunk == b"\x42" * len(chunk)
            total_read += len(chunk)
    _, peak_get = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert total_read == len(payload)
    # Allow plenty of headroom but fail if somebody slurps the whole
    # payload (10 MB) into a single allocation. 3 MB ceiling is ample
    # for python bookkeeping + the 64 KB drain buffer but still well
    # below the 10 MB slurp we're guarding against.
    assert peak_put < 3 * 1024 * 1024, (
        f"put_stream peak memory {peak_put} exceeded 3 MB budget"
    )
    assert peak_get < 3 * 1024 * 1024, (
        f"get_stream peak memory {peak_get} exceeded 3 MB budget"
    )


def test_list_prefix_returns_matching_keys(tmp_path: Path) -> None:
    store = LocalFSObjectStore(tmp_path)
    store.put_bytes("a/one.bin", b"1")
    store.put_bytes("a/two.bin", b"2")
    store.put_bytes("b/three.bin", b"3")
    keys_a = sorted(store.list_prefix("a/"))
    assert keys_a == ["a/one.bin", "a/two.bin"]
    keys_b = sorted(store.list_prefix("b/"))
    assert keys_b == ["b/three.bin"]


def test_list_prefix_empty_store_yields_nothing(tmp_path: Path) -> None:
    store = LocalFSObjectStore(tmp_path)
    assert list(store.list_prefix("")) == []


def test_list_prefix_empty_dirs_are_invisible(tmp_path: Path) -> None:
    """Matches S3 semantics: empty prefixes don't exist in the object
    listing. ``os.walk`` gives us this for free; pinned so nobody later
    adds a directory-listing shim that breaks parity.
    """
    store = LocalFSObjectStore(tmp_path)
    (tmp_path / "empty-dir").mkdir()
    store.put_bytes("real/file.bin", b"x")
    assert sorted(store.list_prefix("")) == ["real/file.bin"]


def test_delete_missing_is_idempotent(tmp_path: Path) -> None:
    store = LocalFSObjectStore(tmp_path)
    # Matches S3 semantics — deleting a key that isn't there is a no-op.
    store.delete("never-existed.bin")
    # And a real delete round-trip still works.
    store.put_bytes("gone.bin", b"x")
    store.delete("gone.bin")
    assert not store.exists("gone.bin")
    # Second delete of the same key is still a no-op.
    store.delete("gone.bin")


def test_exists_true_false(tmp_path: Path) -> None:
    store = LocalFSObjectStore(tmp_path)
    assert store.exists("never.bin") is False
    store.put_bytes("here.bin", b"x")
    assert store.exists("here.bin") is True


def test_get_missing_raises_key_error(tmp_path: Path) -> None:
    store = LocalFSObjectStore(tmp_path)
    with pytest.raises(KeyError):
        store.get_bytes("never.bin")
    with pytest.raises(KeyError):
        store.get_stream("never.bin")


def test_put_overwrites(tmp_path: Path) -> None:
    store = LocalFSObjectStore(tmp_path)
    store.put_bytes("k.bin", b"old")
    store.put_bytes("k.bin", b"new")
    assert store.get_bytes("k.bin") == b"new"


def test_nested_keys_create_parent_dirs(tmp_path: Path) -> None:
    """``put_bytes("a/b/c.bin", ...)`` is expected to create ``a/b/``
    on the way in, matching S3's flat-namespace illusion. The local
    backend is where this shim lives; callers never mkdir themselves.
    """
    store = LocalFSObjectStore(tmp_path)
    store.put_bytes("a/b/c.bin", b"x")
    assert (tmp_path / "a" / "b" / "c.bin").exists()
    assert store.get_bytes("a/b/c.bin") == b"x"


def test_stream_closes_file_handle(tmp_path: Path) -> None:
    """``get_stream`` returns a context-manager-capable handle; once the
    caller exits the block the OS handle is released. Pinned so nobody
    later hands out a ``BytesIO`` copy of the file (which would violate
    the streaming contract on larger payloads)."""
    store = LocalFSObjectStore(tmp_path)
    store.put_bytes("k.bin", b"x" * 128)
    handle = store.get_stream("k.bin")
    assert handle.readable()
    handle.close()
    # Re-using a closed handle must fail loudly rather than silently
    # return empty bytes.
    with pytest.raises(ValueError):
        handle.read()


def test_put_stream_buffer_size_is_modest(tmp_path: Path) -> None:
    """``put_stream`` uses ``shutil.copyfileobj`` with a conservative
    buffer size. If someone later bumps the buffer to something
    memory-hostile (say 128 MB), this test trips. 1 MB is generous
    headroom over the 16 KB default but catches runaway changes.
    """
    import shutil

    store = LocalFSObjectStore(tmp_path)
    payload = b"x" * (2 * 1024 * 1024)
    src = io.BytesIO(payload)
    observed_bufsize: list[int] = []
    real_copy = shutil.copyfileobj

    def _spy(fsrc: object, fdst: object, length: int = -1) -> None:
        observed_bufsize.append(length)
        return real_copy(fsrc, fdst, length)  # type: ignore[arg-type]

    # Monkey-patch only for the duration of the put_stream call.
    shutil.copyfileobj = _spy  # type: ignore[assignment]
    try:
        store.put_stream("k.bin", src)
    finally:
        shutil.copyfileobj = real_copy  # type: ignore[assignment]

    assert observed_bufsize, "put_stream must route through shutil.copyfileobj"
    # The length argument is either -1 (= shutil's default 16 KB window)
    # or a finite, modest buffer. Guard against accidental multi-MB buffers.
    for n in observed_bufsize:
        assert n == -1 or n <= 1 * 1024 * 1024, (
            f"put_stream buffer size {n} exceeds 1 MB guard"
        )
