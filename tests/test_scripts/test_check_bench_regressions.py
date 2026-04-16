"""Tests for ``scripts/check_bench_regressions.py``.

Covers the four paths the CI workflow needs to trust:

1. Under tolerance -> exit 0, no regression.
2. Over tolerance -> exit 1, diff table mentions the regressing metric.
3. Missing golden file -> auto-create from current, exit 0.
4. New metric in current (absent from golden) -> recorded as ``new``,
   never counted as a regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_bench_regressions import (
    Diff,
    check_pair,
    format_diff_table,
    main,
)


def _write_json(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_regression_under_tolerance_passes(tmp_path: Path) -> None:
    golden = tmp_path / "golden.json"
    current = tmp_path / "current.json"
    _write_json(
        golden,
        [
            {
                "system": "soma",
                "n": 1000,
                "retrieve_avg_ms": 10.0,
                "recall_at_k": 0.90,
            }
        ],
    )
    # +10% latency, -1% recall — both well under defaults (20% / 5%).
    _write_json(
        current,
        [
            {
                "system": "soma",
                "n": 1000,
                "retrieve_avg_ms": 11.0,
                "recall_at_k": 0.891,
            }
        ],
    )
    diffs, created = check_pair(
        current_path=current,
        golden_path=golden,
        tolerance_latency=0.20,
        tolerance_recall=0.05,
    )
    assert not created
    assert all(d.status == "pass" for d in diffs), [
        (d.metric, d.status, d.delta_pct) for d in diffs
    ]


def test_regression_above_tolerance_fails(tmp_path: Path) -> None:
    golden = tmp_path / "golden.json"
    current = tmp_path / "current.json"
    _write_json(
        golden,
        [
            {
                "system": "soma",
                "n": 1000,
                "retrieve_avg_ms": 10.0,
                "recall_at_k": 0.90,
            }
        ],
    )
    # Latency +50% (over 20%), recall -20% (over 5%) -> both regress.
    _write_json(
        current,
        [
            {
                "system": "soma",
                "n": 1000,
                "retrieve_avg_ms": 15.0,
                "recall_at_k": 0.72,
            }
        ],
    )
    diffs, _ = check_pair(
        current_path=current,
        golden_path=golden,
        tolerance_latency=0.20,
        tolerance_recall=0.05,
    )
    regressions = {d.metric for d in diffs if d.status == "regression"}
    assert "retrieve_avg_ms" in regressions
    assert "recall_at_k" in regressions

    # The table rendering picks up the regressing rows.
    table = format_diff_table(diffs)
    assert "regression" in table
    assert "retrieve_avg_ms" in table

    # And the main() entrypoint exits 1.
    rc = main(
        [
            "--current",
            str(current),
            "--golden",
            str(golden),
        ]
    )
    assert rc == 1


def test_missing_golden_file_creates_it(tmp_path: Path, capsys) -> None:
    golden = tmp_path / "golden" / "new_bench.json"  # does not exist
    current = tmp_path / "current.json"
    payload = [
        {
            "system": "soma",
            "n": 1000,
            "retrieve_avg_ms": 5.5,
            "recall_at_k": 0.923,
        }
    ]
    _write_json(current, payload)
    assert not golden.exists()

    diffs, created = check_pair(
        current_path=current,
        golden_path=golden,
        tolerance_latency=0.20,
        tolerance_recall=0.05,
    )
    assert created is True
    assert diffs == []
    assert golden.exists()
    assert json.loads(golden.read_text(encoding="utf-8")) == payload

    # main() also returns 0 on the bootstrap path.
    rc = main(
        [
            "--current",
            str(current),
            "--golden",
            str(tmp_path / "golden" / "another_new_bench.json"),
        ]
    )
    assert rc == 0


def test_new_metric_in_current_is_recorded(tmp_path: Path) -> None:
    golden = tmp_path / "golden.json"
    current = tmp_path / "current.json"
    _write_json(
        golden,
        [
            {
                "system": "soma",
                "n": 1000,
                "retrieve_avg_ms": 10.0,
            }
        ],
    )
    # Current adds a new field ``disk_mb`` absent from golden.
    _write_json(
        current,
        [
            {
                "system": "soma",
                "n": 1000,
                "retrieve_avg_ms": 10.5,
                "disk_mb": 1.2,
            }
        ],
    )
    diffs, _ = check_pair(
        current_path=current,
        golden_path=golden,
        tolerance_latency=0.20,
        tolerance_recall=0.05,
    )
    new_metrics = {d.metric: d for d in diffs if d.status == "new"}
    assert "disk_mb" in new_metrics
    assert new_metrics["disk_mb"].current == pytest.approx(1.2)
    assert new_metrics["disk_mb"].golden is None

    # No regressions recorded — exit 0.
    rc = main(
        [
            "--current",
            str(current),
            "--golden",
            str(golden),
        ]
    )
    assert rc == 0


def test_diff_dataclass_round_trip() -> None:
    # Light sanity check that the Diff dataclass is importable and
    # honours fields we rely on in format_diff_table.
    d = Diff(
        dataset="x",
        row_key="k",
        metric="retrieve_avg_ms",
        golden=1.0,
        current=2.0,
        delta_pct=1.0,
        status="regression",
    )
    table = format_diff_table([d])
    assert "retrieve_avg_ms" in table
    assert "regression" in table
