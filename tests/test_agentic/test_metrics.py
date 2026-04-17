"""Tests for metrics formatting."""

from __future__ import annotations

from benchmarks.agentic.metrics import TaskResult, format_comparison_table


def test_task_result_defaults() -> None:
    tr = TaskResult()
    assert tr.completion is False
    assert tr.accuracy == 0.0
    assert tr.steps == 0


def test_format_empty() -> None:
    assert "_No results._" in format_comparison_table({})


def test_format_single_model_single_task() -> None:
    results = {
        "Qwen3.5-4B": {
            "fact_recall": TaskResult(completion=True, accuracy=80.0, steps=100),
        }
    }
    md = format_comparison_table(results)
    assert "Qwen3.5-4B" in md
    assert "fact_recall" in md
    assert "80%" in md
    assert "pass" in md


def test_format_multiple_models() -> None:
    results = {
        "ModelA": {
            "t1": TaskResult(completion=True, accuracy=90.0, steps=5),
            "t2": TaskResult(completion=False, accuracy=30.0, steps=10),
        },
        "ModelB": {
            "t1": TaskResult(completion=False, accuracy=50.0, steps=8),
        },
    }
    md = format_comparison_table(results)
    lines = md.strip().split("\n")
    # header + separator + 2 data rows
    assert len(lines) == 4
    # ModelB missing t2 -> should show "-"
    assert "-" in lines[3]
