"""Bundle-storage layer.

Foundation for the S3 / GCS bundle-backend track (Phases 30-33).
``MemoryLayer.save`` / ``MemoryLayer.load`` route every byte through
this abstraction so swapping in a remote object store is a matter of
adding an adapter, not patching the memory layer.

Public surface:

  * :class:`ObjectStore` — the Protocol every backend satisfies.
  * :class:`LocalFSObjectStore` — wraps ``pathlib`` + ``shutil``;
    identical semantics to S3-but-local.
  * :func:`parse_store_url` — single dispatch point from a user
    string / Path to a concrete store instance.
"""

from __future__ import annotations

from soma.storage.base import ObjectStore
from soma.storage.local import LocalFSObjectStore
from soma.storage.urls import parse_store_url

__all__ = ["ObjectStore", "LocalFSObjectStore", "parse_store_url"]
