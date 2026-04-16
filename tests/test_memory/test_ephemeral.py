"""Tests for :meth:`MemoryLayer.ephemeral` — no-WAL convenience factory.

Pins the Phase 19 "in-RAM by default, persist on demand" ergonomics
from ``docs/plans/2026-04-16-phase-19-ephemeral-mode.md``. Every test
asserts the ephemeral MemoryLayer leaves zero on-disk footprint until
the caller explicitly invokes :meth:`save`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.memory import MemoryLayer


def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(8, generator=g)


def test_ephemeral_no_wal_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ephemeral mode never touches disk — not even the WAL sidecar.

    Chdir the process into ``tmp_path`` so any stray relative-path writes
    land where we can see them. Post-store, the dir must still be empty.
    """
    monkeypatch.chdir(tmp_path)
    mem = MemoryLayer.ephemeral(embed_fn=_stub_embed, embed_dim=8)
    assert mem._bundle_path is None
    assert mem._wal is None

    nid = mem.store("hello")
    assert nid in mem

    # Full recursive sweep — nothing ephemeral should have materialized
    # on disk. Including the cwd we just chdir'd into.
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert leftovers == [], f"ephemeral mode wrote files: {leftovers}"


def test_ephemeral_save_then_load_round_trip(tmp_path: Path) -> None:
    """``save(path)`` on an ephemeral layer yields a normally-loadable bundle."""
    mem = MemoryLayer.ephemeral(embed_fn=_stub_embed, embed_dim=8)
    nid = mem.store("hello world")
    extra = mem.store("second entry", metadata={"tag": "x"})

    bundle = tmp_path / "b"
    mem.save(bundle)
    assert bundle.is_dir()
    assert (bundle / "memory_index.json").exists()

    mem2 = MemoryLayer.load(bundle, embed_fn=_stub_embed)
    assert nid in mem2
    assert extra in mem2
    assert mem2.get(extra).metadata == {"tag": "x"}


def test_ephemeral_rejects_bundle_path_kwarg() -> None:
    """``bundle_path=`` is reserved — the whole point is no-disk mode."""
    with pytest.raises(TypeError, match="bundle_path"):
        MemoryLayer.ephemeral(
            embed_fn=_stub_embed,
            embed_dim=8,
            bundle_path="./should-not-be-here",
        )


def test_ephemeral_requires_an_embedder() -> None:
    """Must pass either (embed_fn, embed_dim) or sbert_model=."""
    with pytest.raises(ValueError, match="embed_fn"):
        MemoryLayer.ephemeral()


def test_ephemeral_sbert_and_embed_fn_mutually_exclusive() -> None:
    """Mixing sbert_model with embed_fn/embed_dim is a clear-error."""
    with pytest.raises(ValueError, match="not both"):
        MemoryLayer.ephemeral(
            sbert_model="all-MiniLM-L6-v2",
            embed_fn=_stub_embed,
            embed_dim=8,
        )
