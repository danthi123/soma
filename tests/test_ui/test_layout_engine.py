"""Tests for the 2D force-directed layout engine."""

from __future__ import annotations

import pytest

from soma.ui.graph_view.layout_engine import ForceDirectedLayout


def _snapshot(nodes: list[str], edges: list[tuple[str, str]], step: int = 0) -> dict:
    return {
        "global_step": step,
        "nodes": [{"id": nid, "type": "associator", "activation_ema": 0.1} for nid in nodes],
        "edges": [{"source": s, "target": t, "weight": 0.5} for s, t in edges],
    }


class TestForceDirectedLayout:
    def test_empty_snapshot_returns_empty_positions(self) -> None:
        pytest.importorskip("networkx")
        layout = ForceDirectedLayout(iterations=5)
        result = layout.compute(_snapshot([], []))
        assert result.positions == {}

    def test_computes_positions_for_all_nodes(self) -> None:
        pytest.importorskip("networkx")
        layout = ForceDirectedLayout(iterations=10)
        result = layout.compute(_snapshot(["a", "b", "c"], [("a", "b"), ("b", "c")]))
        assert set(result.positions) == {"a", "b", "c"}
        for xy in result.positions.values():
            assert isinstance(xy, tuple)
            assert len(xy) == 2

    def test_positions_stay_bounded_across_updates(self) -> None:
        """After re-solving with a topology change, existing nodes should
        still end up in the roughly-unit-range coordinate space (i.e., the
        layout remains bounded, not drifting off to infinity).

        We deliberately don't assert "positions barely moved" because
        spring_layout's rescaling can mirror the cluster; the important
        property for the UI is that positions stay in range so the
        renderer doesn't need to autoscale wildly.
        """
        pytest.importorskip("networkx")
        layout = ForceDirectedLayout(iterations=30, seed=0)
        layout.compute(_snapshot(["a", "b", "c"], [("a", "b")]))
        second = layout.compute(_snapshot(["a", "b", "c", "d"], [("a", "b"), ("c", "d")]))
        # Spring-layout outputs are rescaled to roughly [-1, 1].
        for nid, (x, y) in second.positions.items():
            assert -1.5 <= x <= 1.5, f"{nid} x={x} out of range"
            assert -1.5 <= y <= 1.5, f"{nid} y={y} out of range"

    def test_reset_clears_cache(self) -> None:
        pytest.importorskip("networkx")
        layout = ForceDirectedLayout(iterations=5)
        layout.compute(_snapshot(["a"], []))
        assert layout.last() is not None
        layout.reset()
        assert layout.last() is None

    def test_rejects_non_positive_iterations(self) -> None:
        with pytest.raises(ValueError, match="iterations"):
            ForceDirectedLayout(iterations=0)
