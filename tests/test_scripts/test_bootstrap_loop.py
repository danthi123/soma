"""Tests for scripts/bootstrap_loop.py — idempotent directory + file scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.bootstrap_loop import (
    FIXED_PROMPTS,
    build_heldout_split,
    build_token_freq,
    ensure_state_dirs,
    seed_current_yaml,
    seed_fixed_prompts,
)


def test_fixed_prompts_count() -> None:
    assert len(FIXED_PROMPTS) == 20
    # prompt 16 (index 15) is empty per design Appendix C
    assert FIXED_PROMPTS[15] == ""
    # last prompt is long
    assert len(FIXED_PROMPTS[-1]) > 20


def test_ensure_state_dirs(tmp_path: Path) -> None:
    ensure_state_dirs(tmp_path)
    for rel in [
        ".soma-loop/signals",
        ".soma-loop/state/ticks",
        ".soma-loop/metrics",
        ".soma-loop/reports/chat",
        ".soma-loop/logs",
        ".soma-loop/pid",
    ]:
        assert (tmp_path / rel).is_dir(), f"missing: {rel}"
    for rel in [
        ".soma-loop/state/change_log.jsonl",
        ".soma-loop/state/approval_queue.jsonl",
        ".soma-loop/state/consecutive_failures.json",
    ]:
        assert (tmp_path / rel).is_file(), f"missing: {rel}"
    cf = json.loads((tmp_path / ".soma-loop/state/consecutive_failures.json").read_text())
    assert cf == {"count": 0, "last_reset_ts": None, "last_failure_ts": None}


def test_seed_current_yaml_copies_default(tmp_path: Path) -> None:
    default = tmp_path / "configs/default.yaml"
    current = tmp_path / "configs/current.yaml"
    default.parent.mkdir(parents=True)
    default.write_text("vocab_size: 512\ntext_embed_dim: 64\n", encoding="utf-8")
    seed_current_yaml(default, current, force=False)
    assert current.exists()
    assert current.read_text() == default.read_text()
    # Never overwrite without force
    current.write_text("vocab_size: 999\n", encoding="utf-8")
    seed_current_yaml(default, current, force=False)
    assert "999" in current.read_text(), "must not be overwritten without --force"
    # --force still does NOT overwrite current.yaml per G64 safety
    seed_current_yaml(default, current, force=True)
    assert "999" in current.read_text(), "even --force preserves user-edited current.yaml"


def test_seed_fixed_prompts_preserves_user_edits(tmp_path: Path) -> None:
    path = tmp_path / "data/fixed_prompts.txt"
    path.parent.mkdir(parents=True)
    seed_fixed_prompts(path, force=False)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    # Operator edits — must be preserved on rerun (with OR without --force)
    path.write_text("custom prompt\n", encoding="utf-8")
    seed_fixed_prompts(path, force=True)  # G64: never overwritten
    assert path.read_text() == "custom prompt\n"


def test_build_heldout_split_deterministic(tmp_path: Path) -> None:
    corpus = tmp_path / "data/corpus.txt"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("\n".join(f"line {i}" for i in range(100)) + "\n", encoding="utf-8")
    out_a = tmp_path / "data/heldout_a.txt"
    out_b = tmp_path / "data/heldout_b.txt"
    build_heldout_split(corpus, out_a, ratio=0.1, seed=42)
    build_heldout_split(corpus, out_b, ratio=0.1, seed=42)
    assert out_a.read_text() == out_b.read_text()
    # Roughly 10% of 100 lines
    assert 5 <= out_a.read_text().count("\n") <= 15


def test_build_token_freq_json(tmp_path: Path) -> None:
    corpus = tmp_path / "data/corpus.txt"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("hello world\nhello there\n", encoding="utf-8")
    out = tmp_path / "data/corpus_token_freq.json"
    build_token_freq(corpus, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
    assert "hello" in data
    assert data["hello"] == 2
