"""Pins the ``dim=`` / ``embed_dim=`` alias on :class:`ChromaBackend`.

Ergonomic edge #2 from the 0.2 release audit: the rest of SOMA
uses ``embed_dim=`` everywhere (matching :class:`MemoryLayer`'s
constructor), but :class:`ChromaBackend` originally named its
dimension kwarg ``dim=``. That's confusing at call sites that
mix-and-match backends. 0.2 accepts both, with ``embed_dim=`` as
the canonical name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from soma.memory.backends.chroma import ChromaBackend  # noqa: E402


def test_embed_dim_kwarg_accepted(tmp_path: Path) -> None:
    """``embed_dim=`` is the new canonical spelling."""
    b = ChromaBackend(path=str(tmp_path / "c"), collection_name="soma_t", embed_dim=8)
    try:
        assert b._dim == 8
    finally:
        b.close()


def test_dim_kwarg_still_accepted_for_back_compat(tmp_path: Path) -> None:
    """``dim=`` keeps working for callers predating 0.2."""
    b = ChromaBackend(path=str(tmp_path / "c"), collection_name="soma_t", dim=8)
    try:
        assert b._dim == 8
    finally:
        b.close()


def test_embed_dim_and_dim_conflict_raises(tmp_path: Path) -> None:
    """Supplying both kwargs with different values is a clear-error.

    Matching values are permitted — they're unambiguous — but a
    mismatch signals a caller-side bug we'd rather surface up front.
    """
    with pytest.raises(ValueError, match="embed_dim"):
        ChromaBackend(
            path=str(tmp_path / "c"),
            collection_name="soma_t",
            embed_dim=8,
            dim=16,
        )


def test_embed_dim_and_dim_matching_ok(tmp_path: Path) -> None:
    """Matching values pass through — no false-positive on
    defensive double-spelling."""
    b = ChromaBackend(
        path=str(tmp_path / "c"),
        collection_name="soma_t",
        embed_dim=8,
        dim=8,
    )
    try:
        assert b._dim == 8
    finally:
        b.close()
