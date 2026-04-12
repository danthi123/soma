"""Smoke tests for the top-level DPG app.

We can't reliably spin up a real DPG viewport in CI, so these tests
only exercise the code paths that don't require an OpenGL context:
- Building the UIState + panel list via ``build_app_for_tests``.
- Panel registration coverage.
- Lightweight checks on the graph_panel's snapshot truncation.
"""

from __future__ import annotations

import pytest

from soma.ui.app import AppOptions, build_app_for_tests
from soma.ui.panels.graph_panel import _truncate_snapshot
from soma.ui.registry import default_registry


def _snapshot(num: int) -> dict:
    return {
        "global_step": 0,
        "nodes": [
            {"id": f"n{i}", "type": "associator", "activation_ema": float(i)} for i in range(num)
        ],
        "edges": [
            {"source": f"n{i}", "target": f"n{i + 1}", "weight": 0.5} for i in range(num - 1)
        ],
        "sensor_nodes": {"text": "n0"},
        "output_nodes": {"text": f"n{num - 1}"},
    }


class TestBuildForTests:
    def test_build_returns_state_and_panels(self) -> None:
        state, panels_list = build_app_for_tests(AppOptions())
        assert state.config is not None
        assert state.bus is not None
        assert state.controller is not None
        # Every built-in panel should be present.
        names = {n for n, _ in panels_list}
        assert {"config", "controls", "metrics", "graph", "chat"}.issubset(names)

    def test_every_registered_panel_has_name_and_label(self) -> None:
        for name in default_registry.names():
            panel = default_registry.build(name)
            assert panel.name == name
            assert isinstance(panel.display_label, str) and panel.display_label


class TestTruncateSnapshot:
    def test_under_cap_unchanged(self) -> None:
        snap = _snapshot(5)
        assert _truncate_snapshot(snap, cap=10) is snap

    def test_over_cap_trims_to_most_active_plus_boundaries(self) -> None:
        snap = _snapshot(20)
        trimmed = _truncate_snapshot(snap, cap=5)
        assert len(trimmed["nodes"]) == 5
        # Boundary nodes n0 (sensor) and n19 (output) must survive.
        ids = {n["id"] for n in trimmed["nodes"]}
        assert "n0" in ids
        assert "n19" in ids
        # All edges kept should reference only kept nodes.
        for edge in trimmed["edges"]:
            assert edge["source"] in ids
            assert edge["target"] in ids

    def test_over_cap_keeps_both_boundaries_even_if_inactive(self) -> None:
        # Zero-activation sensor + output. Force them into the set.
        snap = {
            "global_step": 0,
            "nodes": [
                {"id": "s", "type": "sensor", "activation_ema": 0.0},
                {"id": "o", "type": "output", "activation_ema": 0.0},
                {"id": "a", "type": "associator", "activation_ema": 0.9},
                {"id": "b", "type": "associator", "activation_ema": 0.8},
            ],
            "edges": [],
            "sensor_nodes": {"text": "s"},
            "output_nodes": {"text": "o"},
        }
        trimmed = _truncate_snapshot(snap, cap=2)
        ids = {n["id"] for n in trimmed["nodes"]}
        assert "s" in ids
        assert "o" in ids


class TestPanelInstantiation:
    """Every registered panel must be constructible without a DPG context."""

    @pytest.mark.parametrize("name", sorted(default_registry.names()))
    def test_panel_instantiates(self, name: str) -> None:
        panel = default_registry.build(name)
        assert panel.name == name
