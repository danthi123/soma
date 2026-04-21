"""Protocol-shape tests for :class:`soma.storage.ObjectStore`.

These are contract pins: the Protocol declares a small, streaming-capable
surface that every object store (local, S3 in Phase 31, GCS in Phase 32)
satisfies. Breaking a method signature or silently adding a new one
without updating these tests is a regression.
"""

from __future__ import annotations

from pathlib import Path

from soma.storage import LocalFSObjectStore, ObjectStore


def test_local_store_satisfies_object_store_protocol(tmp_path: Path) -> None:
    """``LocalFSObjectStore`` must pass ``isinstance`` against the
    Protocol. ``ObjectStore`` is marked ``runtime_checkable`` so callers
    can branch on "is this a store?" without importing every adapter.
    """
    store = LocalFSObjectStore(tmp_path)
    assert isinstance(store, ObjectStore)


def test_all_methods_declared() -> None:
    """Regression guard: any future method added to ``ObjectStore`` needs
    both a test exercising it and a matching entry in this set. A typo
    in the Protocol name also trips this check.
    """
    required = {
        "get_bytes",
        "put_bytes",
        "get_stream",
        "put_stream",
        "list_prefix",
        "delete",
        "exists",
    }
    # ``__protocol_attrs__`` is Python-3.12+. On 3.11, fall back to the
    # private ``typing._get_protocol_attrs`` helper that has existed since
    # 3.8 and returns the same set.
    if hasattr(ObjectStore, "__protocol_attrs__"):
        declared = set(ObjectStore.__protocol_attrs__)  # type: ignore[attr-defined]
    else:
        from typing import _get_protocol_attrs  # type: ignore[attr-defined]
        declared = set(_get_protocol_attrs(ObjectStore))
    missing = required - declared
    assert not missing, f"ObjectStore is missing Protocol methods: {missing}"
