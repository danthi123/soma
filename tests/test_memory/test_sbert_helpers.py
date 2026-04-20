"""Tests for the sbert convenience helpers on :class:`MemoryLayer`.

Pins the ``with_sbert`` / ``load_with_sbert`` symmetry added in 0.2
(ergonomic edge #1 from the release audit): callers who built a
MemoryLayer with :meth:`with_sbert` should be able to round-trip the
bundle via :meth:`save` + :meth:`load_with_sbert` without having to
hand-rebuild the sbert embed closure.

These tests require ``sentence-transformers`` and pull the
``all-MiniLM-L6-v2`` model on first run. Skipped cleanly when the
extra isn't installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma.memory import MemoryLayer

pytest.importorskip("sentence_transformers")


def test_with_sbert_records_model_name() -> None:
    """``with_sbert`` must remember which model it used so
    ``save`` can persist it."""
    mem = MemoryLayer.with_sbert()
    assert mem._sbert_model_name == "all-MiniLM-L6-v2"


def test_with_sbert_records_custom_model_name() -> None:
    """Explicit model-name arg propagates to the instance attr."""
    # Reuse the default model so the test stays fast (no second
    # download); we only care that the attr reflects the arg.
    mem = MemoryLayer.with_sbert("all-MiniLM-L6-v2")
    assert mem._sbert_model_name == "all-MiniLM-L6-v2"


def test_save_persists_sbert_model_name(tmp_path: Path) -> None:
    """After ``save`` the bundle's ``memory_index.json`` carries the
    sbert model name for the matching load helper to pick up."""
    mem = MemoryLayer.with_sbert()
    mem.store("hello world")
    mem.save(tmp_path / "bundle")
    idx = json.loads((tmp_path / "bundle" / "memory_index.json").read_text())
    assert idx["sbert_model_name"] == "all-MiniLM-L6-v2"


def test_load_with_sbert_round_trip(tmp_path: Path) -> None:
    """Store → save → ``load_with_sbert`` recovers the same content
    without the caller having to rebuild the embedder."""
    bundle = tmp_path / "bundle"
    mem = MemoryLayer.with_sbert()
    mem.store("the cat sat on the mat")
    mem.store("a peaceful afternoon in the park")
    mem.store("quantum mechanics is weird")
    mem.save(bundle)

    restored = MemoryLayer.load_with_sbert(bundle)
    assert len(restored) == 3
    top = restored.retrieve("feline on a rug", k=1)[0]
    assert "cat" in top.text


def test_load_with_sbert_without_model_arg_picks_persisted(tmp_path: Path) -> None:
    """When ``model_name`` is omitted, the helper reads it from the
    bundle's ``memory_index.json`` (written by ``save``)."""
    bundle = tmp_path / "bundle"
    mem = MemoryLayer.with_sbert()
    mem.store("one fact")
    mem.save(bundle)

    restored = MemoryLayer.load_with_sbert(bundle)
    assert restored._sbert_model_name == "all-MiniLM-L6-v2"


def test_load_with_sbert_explicit_model_name_wins(tmp_path: Path) -> None:
    """Explicit ``model_name=`` overrides whatever's persisted."""
    bundle = tmp_path / "bundle"
    mem = MemoryLayer.with_sbert("all-MiniLM-L6-v2")
    mem.store("one fact")
    mem.save(bundle)

    restored = MemoryLayer.load_with_sbert(bundle, model_name="all-MiniLM-L6-v2")
    assert restored._sbert_model_name == "all-MiniLM-L6-v2"


def test_load_with_sbert_falls_back_to_default_when_field_missing(
    tmp_path: Path,
) -> None:
    """Bundles saved before 0.2 don't have ``sbert_model_name`` in
    their index. The helper must still work — it falls back to the
    :meth:`with_sbert` default (``all-MiniLM-L6-v2``)."""
    bundle = tmp_path / "bundle"
    mem = MemoryLayer.with_sbert()
    mem.store("legacy fact")
    mem.save(bundle)

    # Simulate a pre-0.2 bundle by stripping the new field.
    idx_path = bundle / "memory_index.json"
    idx = json.loads(idx_path.read_text())
    del idx["sbert_model_name"]
    idx_path.write_text(json.dumps(idx))

    restored = MemoryLayer.load_with_sbert(bundle)
    assert restored._sbert_model_name == "all-MiniLM-L6-v2"
    assert len(restored) == 1
