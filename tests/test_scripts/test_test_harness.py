"""Tests for scripts/test_harness.py — B-tier evaluation."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.test_harness import (
    compute_ema,
    compute_health_flags,
    compute_kl_divergence,
    summarize_growth,
)


def test_compute_kl_divergence_identical_is_zero() -> None:
    ref = {"a": 3, "b": 2, "c": 5}
    out = {"a": 30, "b": 20, "c": 50}  # same proportions, different magnitudes
    kl = compute_kl_divergence(out, ref)
    assert kl == pytest.approx(0.0, abs=1e-6)


def test_compute_kl_divergence_non_negative() -> None:
    ref = {"a": 1, "b": 1, "c": 1}
    out = {"a": 10, "b": 1, "c": 1}
    assert compute_kl_divergence(out, ref) > 0.0


def test_compute_kl_divergence_handles_missing_keys() -> None:
    # Out mentions a key not in reference — smoothing must keep KL finite.
    ref = {"a": 1, "b": 1}
    out = {"a": 1, "c": 1}
    kl = compute_kl_divergence(out, ref)
    assert math.isfinite(kl)
    assert kl >= 0.0


def test_compute_kl_divergence_empty_inputs() -> None:
    assert compute_kl_divergence({}, {"a": 1}) == 0.0
    assert compute_kl_divergence({"a": 1}, {}) == 0.0


def test_compute_ema_constant_series() -> None:
    assert compute_ema([1.0] * 10, alpha=0.1) == pytest.approx(1.0)


def test_compute_ema_empty_is_nan() -> None:
    assert math.isnan(compute_ema([], alpha=0.1))


def test_compute_ema_weights_recent_more() -> None:
    # After a jump the EMA should lag but move toward the new value.
    old = compute_ema([1.0] * 100 + [10.0], alpha=0.1)
    assert 1.0 < old < 10.0
    # Final value dominated by recent push.
    assert old > 1.5


def test_summarize_growth_counts_events() -> None:
    records = [
        {"step": 100, "num_nodes": 10, "num_edges": 20},
        {"step": 200, "num_nodes": 12, "num_edges": 22},  # +2 nodes, +2 edges -> neuro
        {"step": 300, "num_nodes": 12, "num_edges": 30},  # +0 nodes, +8 edges -> syn
        {"step": 400, "num_nodes": 11, "num_edges": 28},  # -1 nodes -> prune
        {"step": 500, "num_nodes": 11, "num_edges": 28},  # no change
    ]
    summary = summarize_growth(records, lookback_steps=500)
    assert summary["neuro"] == 1
    assert summary["syn"] == 1
    assert summary["prune"] == 1


def test_summarize_growth_respects_lookback() -> None:
    records = [
        {"step": 100, "num_nodes": 10, "num_edges": 20},
        {"step": 200, "num_nodes": 12, "num_edges": 22},
        {"step": 1500, "num_nodes": 12, "num_edges": 22},  # no change
    ]
    # lookback 1000 steps from max_step=1500 → window starts at 500 → only last record inside
    summary = summarize_growth(records, lookback_steps=1000)
    assert summary["neuro"] == 0  # the neurogenesis at step 200 is outside the window


def test_summarize_growth_empty() -> None:
    summary = summarize_growth([], lookback_steps=1000)
    assert summary == {"syn": 0, "neuro": 0, "prune": 0}


def test_health_flags_nan_detected() -> None:
    flags = compute_health_flags(
        {"heldout_loss_mean": float("nan"), "graph": {"nodes": 100, "edges": 200}}
    )
    assert "nan_detected" in flags


def test_health_flags_graph_collapsed() -> None:
    flags = compute_health_flags(
        {"heldout_loss_mean": 1.0, "graph": {"nodes": 2, "edges": 1}}
    )
    assert "graph_collapsed" in flags


def test_health_flags_clean_snapshot_empty() -> None:
    flags = compute_health_flags(
        {
            "heldout_loss_mean": 0.5,
            "graph": {"nodes": 100, "edges": 400},
            "memory": {"wm_occupancy": 0.5, "episodic_fill_pct": 0.3},
        }
    )
    assert flags == []


def test_health_flags_wm_pinned_high() -> None:
    flags = compute_health_flags(
        {
            "heldout_loss_mean": 0.5,
            "graph": {"nodes": 100, "edges": 400},
            "memory": {"wm_occupancy": 0.98, "episodic_fill_pct": 0.3},
        }
    )
    assert "wm_pinned_high" in flags


@pytest.mark.slow
def test_test_harness_produces_valid_report(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[2]
    ckpt = repo_root / "checkpoints/current.pt"
    if not ckpt.exists():
        pytest.skip("no live checkpoint for harness test")

    out_path = tmp_path / "report.json"
    chat_path = tmp_path / "chat.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/test_harness.py"),
            "--out",
            str(out_path),
            "--chat-out",
            str(chat_path),
            "--corpus",
            str(repo_root / "data/tinyshakespeare.txt"),
            "--heldout",
            str(repo_root / "data/heldout.txt"),
            "--fixed-prompts",
            str(repo_root / "data/fixed_prompts.txt"),
            "--device",
            "cpu",
            "--heldout-max-lines",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "global_step" in data
    assert "heldout_loss_mean" in data
    assert "graph" in data and "nodes" in data["graph"]
    assert chat_path.exists()
    chat_lines = chat_path.read_text(encoding="utf-8").splitlines()
    assert len(chat_lines) == 20  # one per fixed prompt
