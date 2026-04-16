"""Abstract :class:`ObjectStore` surface.

The Protocol is the seam between ``MemoryLayer.save/load`` and any
concrete byte-moving backend — local filesystem today
(:class:`soma.storage.LocalFSObjectStore`), S3 / GCS in Phases 31-32.

Keeping this as a ``runtime_checkable`` Protocol rather than an ABC
means callers (and tests) can branch on "is this a store?" without
importing every adapter, and third-party adapters don't have to
subclass a SOMA-specific base. Structural typing FTW.

Contract:
  * ``get_*`` on a missing key raises ``KeyError``.
  * ``delete`` on a missing key is a no-op (S3-compatible semantics).
  * ``list_prefix`` yields keys in the store-native order; callers that
    need sorted output sort themselves. Empty "prefixes" (dirs with
    no files on local FS) don't appear in the listing, matching S3.
  * Streams returned by ``get_stream`` are owned by the caller — they
    must be closed (use ``with``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO, Protocol, runtime_checkable


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal byte-moving surface that MemoryLayer save/load talks to.

    Every concrete store (local FS, S3, GCS) implements this exact
    surface. The Protocol is streaming-first so bundles with hundred-MB
    vectors don't need to fit in RAM during transfer — callers can pipe
    through :meth:`get_stream` / :meth:`put_stream` and only the
    small-fixture paths fall back to :meth:`get_bytes` / :meth:`put_bytes`.
    """

    def get_bytes(self, key: str) -> bytes:
        """Return the object at ``key``. Raises :class:`KeyError` if absent."""
        ...

    def put_bytes(self, key: str, data: bytes) -> None:
        """Upload ``data`` to ``key``. Overwrites an existing object."""
        ...

    def get_stream(self, key: str) -> BinaryIO:
        """Return a readable binary stream for ``key``.

        The caller owns the stream and is responsible for closing it
        (use ``with``). Raises :class:`KeyError` if absent.
        """
        ...

    def put_stream(self, key: str, stream: BinaryIO) -> None:
        """Upload from a readable binary stream to ``key``.

        Drains ``stream`` into the store without slurping the whole
        payload into memory. Overwrites an existing object.
        """
        ...

    def list_prefix(self, prefix: str) -> Iterator[str]:
        """Yield keys beginning with ``prefix`` in store-native order."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key``. Idempotent — missing keys are a no-op."""
        ...

    def exists(self, key: str) -> bool:
        """True if ``key`` is present in the store."""
        ...
