"""Local-filesystem implementation of :class:`soma.storage.ObjectStore`.

Thin wrapper over ``pathlib`` + ``shutil`` that mirrors the S3 object-
store surface: keys are forward-slash paths, nested directories are
created on demand, ``delete`` is idempotent, listing an empty "prefix"
yields nothing.

Writes through :meth:`LocalFSObjectStore.put_bytes` and
:meth:`put_stream` are *atomic*: they land first in a sibling ``.tmp``
file and then ``os.replace`` into the final key. A crash mid-put
leaves the previous (consistent) object intact. This is how
:meth:`MemoryLayer.save` keeps its "never observe a half-written
bundle" guarantee after Phase 30 routed it through the store API.

Subsequent S3 / GCS adapters (Phases 31 / 32) must match this
behaviour byte-for-byte — ``test_local.py`` is the contract pin.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


class LocalFSObjectStore:
    """Store bytes as files under ``root``.

    Keys are treated as POSIX-style relative paths; ``/`` in the key
    becomes a directory separator on disk. ``put_bytes("a/b.bin", ...)``
    creates ``root/a/b.bin`` (including the ``a/`` subdirectory) so
    callers never have to ``mkdir`` themselves — matches the way S3
    hides prefix hierarchy.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root: Path = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """The local directory backing this store.

        Exposed so callers that specifically need a local path (WAL
        replay, ``portalocker`` file lock) can still reach it after
        resolving via :func:`parse_store_url`.
        """
        return self._root

    # ------------------------------------------------------------------
    # Byte-oriented methods
    # ------------------------------------------------------------------
    def get_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise KeyError(key) from exc

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            # Write → fsync → os.replace, so crashes either leave the
            # previous object intact or the new one visible — never a
            # half-written mix. Matches the invariant MemoryLayer.save
            # used to own inline before Phase 30 routed it through the
            # store API.
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(str(tmp), str(path))
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    # ------------------------------------------------------------------
    # Stream-oriented methods — preferred for large payloads
    # ------------------------------------------------------------------
    def get_stream(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        try:
            return path.open("rb")
        except FileNotFoundError as exc:
            raise KeyError(key) from exc

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("wb") as dst:
                # shutil.copyfileobj defaults to a 16 KB window; that's
                # the conservative buffer we want on the streaming path
                # (no memory blow-up on hundred-MB payloads). Tests pin
                # this against accidental multi-MB bumps.
                shutil.copyfileobj(stream, dst)
                dst.flush()
                with contextlib.suppress(OSError):
                    os.fsync(dst.fileno())
            os.replace(str(tmp), str(path))
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

    # ------------------------------------------------------------------
    # Directory-style helpers
    # ------------------------------------------------------------------
    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield keys that start with ``prefix``.

        Local-FS implementation walks every file under ``root`` and
        emits its forward-slash relative path. Empty directories are
        skipped so the listing matches S3's "prefixes don't exist
        without at least one object under them" semantics.
        """
        for dirpath, _dirnames, filenames in os.walk(self._root):
            for fname in filenames:
                # Skip the transient .tmp sibling files that
                # :meth:`put_bytes` / :meth:`put_stream` leave in
                # place while a write is in flight. A reader listing
                # mid-write must only see committed objects.
                if fname.endswith(".tmp"):
                    continue
                abs_path = Path(dirpath) / fname
                rel = abs_path.relative_to(self._root)
                # Normalise to POSIX-style keys regardless of host OS
                # (on Windows, ``relative_to`` still yields backslashes).
                rel_str = rel.as_posix()
                if rel_str.startswith(prefix):
                    yield rel_str

    def delete(self, key: str) -> None:
        """Remove ``key``. Idempotent — S3-style no-op on a missing key."""
        path = self._resolve(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve(self, key: str) -> Path:
        """Turn a ``/``-separated key into an absolute Path under root."""
        # Normalise backslashes (Windows callers sometimes leak them
        # through) and reject absolute keys — those would escape the
        # store root and break prefix semantics.
        normalised = key.replace("\\", "/")
        if normalised.startswith("/"):
            normalised = normalised.lstrip("/")
        return self._root / normalised
