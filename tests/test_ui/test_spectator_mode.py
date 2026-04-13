"""Tests for spectator-mode UI behaviour when the autonomous-loop flag is present."""

from __future__ import annotations

import json
import time
from pathlib import Path

from soma.ui.panels.graph_panel import load_snapshot_from_disk
from soma.ui.panels.metrics_panel import read_metrics_tail
from soma.ui.state import UIState, detect_autonomous_mode

# ---- detect_autonomous_mode -----------------------------------------------


def test_detect_autonomous_mode_missing(tmp_path: Path) -> None:
    assert detect_autonomous_mode(tmp_path / ".soma-loop") is False


def test_detect_autonomous_mode_present(tmp_path: Path) -> None:
    flag_dir = tmp_path / ".soma-loop/state"
    flag_dir.mkdir(parents=True)
    (flag_dir / "autonomous_mode.flag").touch()
    assert detect_autonomous_mode(tmp_path / ".soma-loop") is True


# ---- UIState.autonomous_mode -----------------------------------------------


def test_uistate_default_autonomous_mode_false() -> None:
    from soma.core.config import SOMAConfig
    from soma.ui.bus import DataBus
    from soma.ui.training_controller import TrainingController

    bus = DataBus()
    state = UIState(
        config=SOMAConfig(),
        bus=bus,
        controller=TrainingController(bus),
    )
    assert state.autonomous_mode is False


def test_uistate_autonomous_mode_when_set() -> None:
    from soma.core.config import SOMAConfig
    from soma.ui.bus import DataBus
    from soma.ui.training_controller import TrainingController

    bus = DataBus()
    state = UIState(
        config=SOMAConfig(),
        bus=bus,
        controller=TrainingController(bus),
        autonomous_mode=True,
    )
    assert state.autonomous_mode is True


# ---- metrics tail reader --------------------------------------------------


def test_read_metrics_tail_empty(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    assert read_metrics_tail(path, max_records=500) == []


def test_read_metrics_tail_returns_last_n(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(json.dumps({"step": i, "loss": 1.0 / (i + 1)}) + "\n")
    tail = read_metrics_tail(path, max_records=3)
    assert len(tail) == 3
    assert [r["step"] for r in tail] == [7, 8, 9]


def test_read_metrics_tail_ignores_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    path.write_text(
        json.dumps({"step": 1, "loss": 1.0})
        + "\ngarbage\n"
        + json.dumps({"step": 2, "loss": 0.5})
        + "\n",
        encoding="utf-8",
    )
    tail = read_metrics_tail(path, max_records=5)
    assert [r["step"] for r in tail] == [1, 2]


def test_read_metrics_tail_missing_file(tmp_path: Path) -> None:
    assert read_metrics_tail(tmp_path / "no.jsonl", max_records=10) == []


# ---- graph snapshot reader -----------------------------------------------


def test_load_snapshot_from_disk_missing_dir(tmp_path: Path) -> None:
    assert load_snapshot_from_disk(tmp_path / "missing") is None


def test_load_snapshot_from_disk_picks_newest(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    older = reports / "graph_snapshot.json"
    older.write_text(
        json.dumps({"global_step": 100, "nodes": [], "edges": []}),
        encoding="utf-8",
    )
    # Wait one second so mtime definitely differs, then write a newer snapshot.
    time.sleep(0.05)
    newest = reports / "graph_snapshot.json"
    newest.write_text(
        json.dumps({"global_step": 200, "nodes": [{"id": "n0"}], "edges": []}),
        encoding="utf-8",
    )
    snap = load_snapshot_from_disk(reports)
    assert snap is not None
    assert snap["global_step"] == 200


def test_load_snapshot_from_disk_bad_json_returns_none(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "graph_snapshot.json").write_text("not json", encoding="utf-8")
    assert load_snapshot_from_disk(reports) is None


# ---- app builder autonomous_mode propagation -----------------------------


def test_build_app_for_tests_detects_autonomous_flag(tmp_path: Path, monkeypatch) -> None:
    flag_dir = tmp_path / ".soma-loop/state"
    flag_dir.mkdir(parents=True)
    (flag_dir / "autonomous_mode.flag").touch()

    from soma.ui.app import AppOptions, build_app_for_tests

    # AppOptions supports passing an explicit soma_loop_dir.
    state, _ = build_app_for_tests(AppOptions(soma_loop_dir=tmp_path / ".soma-loop"))
    assert state.autonomous_mode is True


def test_build_app_for_tests_without_flag_is_not_autonomous(tmp_path: Path) -> None:
    from soma.ui.app import AppOptions, build_app_for_tests

    state, _ = build_app_for_tests(AppOptions(soma_loop_dir=tmp_path / ".soma-loop"))
    assert state.autonomous_mode is False
