"""Graph panel — live 2D force-directed view of the SOMA graph.

Tuning knobs live in ``UIState.graph_view``:
- ``enabled``: if False, rendering + layout computation are skipped entirely.
- ``recompute_every_steps``: how many global_steps between layout recomputes.
- ``max_render_nodes``: cap render complexity on large graphs.

Controls:
- "Enabled" checkbox toggles rendering.
- "Recompute every" int input.
- "Max render nodes" int input.
- "Recompute now" button triggers a layout pass on the next update.

Rendering uses a DPG drawlist so we're not bound by the `mvPlot` type
discipline. Node colors map from NodeType; size scales from
``activation_ema``; edge thickness scales from abs(weight).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    dpg = None
    DPG_AVAILABLE = False

from soma.ui.graph_view.layout_engine import ForceDirectedLayout
from soma.ui.registry import register_panel
from soma.ui.state import UIState
from soma.ui.training_controller import CHANNEL_GRAPH_SNAPSHOT


def load_snapshot_from_disk(reports_dir: Path) -> dict[str, Any] | None:
    """Return the most recent graph snapshot JSON from ``reports_dir``.

    Autonomous-loop mode: the training service is expected to periodically
    write ``reports/graph_snapshot.json``; if it doesn't exist yet (or is
    malformed) we return None so the panel can show a placeholder instead
    of stale data. Only files literally named ``graph_snapshot.json`` are
    considered — other JSON artefacts in ``reports/`` (tick reports,
    chat logs) are ignored.
    """
    if not reports_dir.exists():
        return None
    snap_path = reports_dir / "graph_snapshot.json"
    if not snap_path.exists():
        return None
    try:
        data = json.loads(snap_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data

_NODE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "sensor": (80, 140, 230, 255),
    "associator": (235, 215, 130, 255),
    "integrator": (120, 220, 150, 255),
    "output": (230, 120, 120, 255),
}

_CANVAS_W = 720
_CANVAS_H = 560
_CANVAS_PADDING = 30


@register_panel("graph")
class GraphPanel:
    name = "graph"
    display_label = "Graph view"

    def __init__(self) -> None:
        self._root: int | str | None = None
        self._layout = ForceDirectedLayout()
        self._snapshot: dict[str, Any] | None = None
        self._force_recompute = False
        self._lock = threading.RLock()
        self._last_rendered_step: int = -1

    # ------------------------------------------------------------------
    def build(self, parent: int | str, state: UIState) -> int | str:
        if not DPG_AVAILABLE:  # pragma: no cover
            raise RuntimeError("DearPyGUI not installed; install soma[ui]")

        self._root = dpg.add_group(parent=parent, tag="panel_graph_root")
        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_checkbox(
                label="Enabled",
                default_value=state.graph_view.enabled,
                tag="graph_enabled",
                callback=self._on_enabled,
                user_data=state,
            )
            dpg.add_input_int(
                label="Recompute every (steps)",
                default_value=state.graph_view.recompute_every_steps,
                min_value=1,
                min_clamped=True,
                step=0,
                width=120,
                tag="graph_recompute_every",
                callback=self._on_recompute_every,
                user_data=state,
            )
            dpg.add_input_int(
                label="Max nodes",
                default_value=state.graph_view.max_render_nodes,
                min_value=1,
                min_clamped=True,
                step=0,
                width=100,
                tag="graph_max_nodes",
                callback=self._on_max_nodes,
                user_data=state,
            )
            dpg.add_button(
                label="Recompute now",
                tag="graph_recompute_now",
                callback=self._on_recompute_now,
                user_data=state,
            )

        with dpg.drawlist(
            width=_CANVAS_W,
            height=_CANVAS_H,
            tag="graph_drawlist",
            parent=self._root,
        ):
            pass

        dpg.add_text("no snapshot yet", tag="graph_status", parent=self._root)

        # Subscribe to snapshots *after* drawlist exists so the first payload
        # can trigger a redraw.
        state.bus.subscribe(CHANNEL_GRAPH_SNAPSHOT, self._on_snapshot)
        return self._root

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------
    def _on_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = snapshot

    # ------------------------------------------------------------------
    # Button / widget callbacks
    # ------------------------------------------------------------------
    def _on_enabled(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        user_data.graph_view.enabled = bool(app_data)

    def _on_recompute_every(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        value = max(1, int(app_data))
        user_data.graph_view.recompute_every_steps = value
        # Push to the worker too: the render cadence is bottlenecked by how
        # often snapshots arrive, so wire this setting directly to the
        # worker's publish interval. Without this, lowering "Recompute every"
        # below SessionSpec.snapshot_every has no visible effect.
        user_data.controller.set_snapshot_every(value)

    def _on_max_nodes(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        user_data.graph_view.max_render_nodes = max(1, int(app_data))

    def _on_recompute_now(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        self._force_recompute = True

    # ------------------------------------------------------------------
    # Update (DPG main thread)
    # ------------------------------------------------------------------
    def update(self, state: UIState) -> None:
        if not DPG_AVAILABLE:  # pragma: no cover
            return
        if not state.graph_view.enabled:
            return
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            return
        step = int(snapshot.get("global_step", 0))
        # Render each distinct snapshot the worker publishes. The publish
        # cadence (``SessionSpec.snapshot_every``) is what the "Recompute
        # every" control drives via ``_on_recompute_every``, so any extra
        # filter here would be redundant. "Recompute now" still forces a
        # re-render of the current snapshot.
        if not self._force_recompute and step == self._last_rendered_step:
            return
        self._force_recompute = False
        # Truncate to the most-active nodes if over the cap.
        cap = state.graph_view.max_render_nodes
        if len(snapshot.get("nodes", [])) > cap:
            truncated = _truncate_snapshot(snapshot, cap)
        else:
            truncated = snapshot
        result = self._layout.compute(truncated)
        self._draw(truncated, result.positions, state)
        self._last_rendered_step = step
        dpg.set_value(
            "graph_status",
            f"step={step}  rendered {result.num_nodes}/{len(snapshot.get('nodes', []))} nodes  "
            f"{result.num_edges}/{len(snapshot.get('edges', []))} edges",
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw(
        self,
        snapshot: dict[str, Any],
        positions: dict[str, tuple[float, float]],
        state: UIState,
    ) -> None:
        # Clear previous drawing.
        dpg.delete_item("graph_drawlist", children_only=True)

        if not positions:
            return

        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = max(1e-6, x_max - x_min)
        y_span = max(1e-6, y_max - y_min)

        def project(pt: tuple[float, float]) -> tuple[float, float]:
            px = _CANVAS_PADDING + (pt[0] - x_min) / x_span * (_CANVAS_W - 2 * _CANVAS_PADDING)
            py = _CANVAS_PADDING + (pt[1] - y_min) / y_span * (_CANVAS_H - 2 * _CANVAS_PADDING)
            return (px, py)

        nodes_by_id = {n["id"]: n for n in snapshot.get("nodes", [])}

        # Edges first so they sit behind nodes.
        if state.graph_view.show_edges:
            for edge in snapshot.get("edges", []):
                if edge["source"] not in positions or edge["target"] not in positions:
                    continue
                src_pos = project(positions[edge["source"]])
                tgt_pos = project(positions[edge["target"]])
                weight = float(edge.get("weight", 0.0))
                thickness = max(1.0, min(4.0, abs(weight) * 3.0))
                alpha = int(max(60, min(255, 80 + abs(weight) * 120)))
                dpg.draw_line(
                    src_pos,
                    tgt_pos,
                    color=(180, 180, 200, alpha),
                    thickness=thickness,
                    parent="graph_drawlist",
                )

        # Nodes.
        radius = state.graph_view.node_radius
        for nid, pos in positions.items():
            node = nodes_by_id.get(nid)
            if node is None:
                continue
            px, py = project(pos)
            color = _NODE_COLORS.get(node["type"], (200, 200, 200, 255))
            ema = float(node.get("activation_ema", 0.0))
            size_scale = 1.0 + min(2.0, ema * 2.0)
            r = radius * size_scale
            dpg.draw_circle(
                (px, py),
                r,
                color=(40, 40, 40, 255),
                fill=color,
                thickness=1.0,
                parent="graph_drawlist",
            )


# ----------------------------------------------------------------------
# Snapshot truncation
# ----------------------------------------------------------------------
def _truncate_snapshot(snapshot: dict[str, Any], cap: int) -> dict[str, Any]:
    """Keep the ``cap`` most-active nodes + any boundary sensors/outputs.

    Edges whose endpoints both survive are kept; all others are dropped.
    """
    nodes = list(snapshot.get("nodes", []))
    if len(nodes) <= cap:
        return snapshot
    sensor_ids = set((snapshot.get("sensor_nodes") or {}).values())
    output_ids = set((snapshot.get("output_nodes") or {}).values())
    boundary_ids = sensor_ids | output_ids
    # Sort by activation_ema desc.
    nodes_sorted = sorted(nodes, key=lambda n: float(n.get("activation_ema", 0.0)), reverse=True)
    keep_ids: set[str] = set(boundary_ids)
    for node in nodes_sorted:
        if len(keep_ids) >= cap:
            break
        keep_ids.add(node["id"])
    kept_nodes = [n for n in nodes if n["id"] in keep_ids]
    kept_edges = [
        e for e in snapshot.get("edges", []) if e["source"] in keep_ids and e["target"] in keep_ids
    ]
    return {
        **snapshot,
        "nodes": kept_nodes,
        "edges": kept_edges,
    }
