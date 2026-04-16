"""Tests for the memory-inspect demo subcommands."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

from scripts.demo_memory_inspect import (
    _parse_where,
    cmd_dump,
    cmd_filter,
    cmd_recent,
    cmd_search,
    cmd_stats,
)
from soma.memory import MemoryLayer


def _stub_embed(text: str) -> torch.Tensor:
    """Tiny deterministic embed for tests — hash → 8-d vector."""
    h = hash(text)
    return torch.tensor(
        [
            (h >> i) & 0xF for i in range(0, 32, 4)
        ],
        dtype=torch.float32,
    )


def _make_bundle(tmp_path: Path) -> tuple[MemoryLayer, Path]:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    mem.store("Alex lives in Portland", metadata={"source": "seed", "role": "fact"})
    mem.store("Alex has a dog named Luna", metadata={"source": "seed", "role": "fact"})
    mem.store("Alex works at ArcMotion", metadata={"source": "seed", "role": "fact"})
    mem.store("user asked about dinner", metadata={"source": "chat", "role": "user"})
    mem.store("assistant suggested pasta", metadata={"source": "chat", "role": "assistant"})
    bundle = tmp_path / "bundle"
    mem.save(bundle)
    return mem, bundle


def test_parse_where_splits_on_first_equals() -> None:
    assert _parse_where(["k=v"]) == [("k", "v")]
    assert _parse_where(["k=v=more"]) == [("k", "v=more")]
    assert _parse_where([]) == []


def test_parse_where_rejects_missing_equals() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_where(["bad"])


def test_stats_reports_count_and_metadata_keys(tmp_path: Path) -> None:
    mem, bundle = _make_bundle(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_stats(mem, bundle)
    out = buf.getvalue()
    assert "entries: 5" in out
    assert "embed_dim: 8" in out
    assert "source: 5" in out
    assert "role: 5" in out


def test_stats_handles_empty_bundle(tmp_path: Path) -> None:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    bundle = tmp_path / "empty"
    mem.save(bundle)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_stats(mem, bundle)
    out = buf.getvalue()
    assert "entries: 0" in out


def test_recent_lists_newest_first(tmp_path: Path) -> None:
    mem, _ = _make_bundle(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_recent(mem, n=3)
    out = buf.getvalue()
    # Newest = "assistant suggested pasta", appears before older items.
    assert out.index("assistant suggested pasta") < out.index("user asked about dinner")
    assert "Alex lives in Portland" not in out  # older than top-3


def test_filter_matches_metadata(tmp_path: Path) -> None:
    mem, _ = _make_bundle(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_filter(mem, where=[("source", "chat")], limit=10)
    out = buf.getvalue()
    assert "user asked about dinner" in out
    assert "assistant suggested pasta" in out
    assert "Alex lives in Portland" not in out


def test_filter_combines_with_and(tmp_path: Path) -> None:
    mem, _ = _make_bundle(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_filter(
            mem,
            where=[("source", "chat"), ("role", "assistant")],
            limit=10,
        )
    out = buf.getvalue()
    assert "assistant suggested pasta" in out
    assert "user asked about dinner" not in out


def test_filter_no_matches_prints_message(tmp_path: Path) -> None:
    mem, _ = _make_bundle(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_filter(mem, where=[("source", "missing")], limit=10)
    assert "(no matches)" in buf.getvalue()


def test_search_returns_top_k(tmp_path: Path) -> None:
    mem, _ = _make_bundle(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_search(mem, query="dinner", k=2)
    out = buf.getvalue()
    # Top-k headers present (lines like "[1] score=...").
    assert "[1] score=" in out
    assert "[2] score=" in out


def test_dump_emits_jsonl(tmp_path: Path) -> None:
    mem, _ = _make_bundle(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_dump(mem)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 5
    for ln in lines:
        rec = json.loads(ln)
        assert {"id", "text", "metadata", "timestamp_step"} <= set(rec.keys())
