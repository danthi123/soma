"""Live metrics panel — rolling plots of loss / curiosity / LR / memory.

Subscribes to the ``metrics`` and ``growth_event`` bus channels and
updates DPG line-series widgets on each DPG frame via ``update``. We
don't pull from a deque inside the subscriber (running on the worker
thread would force DPG writes from the wrong thread); instead, the
subscriber captures the last N records and ``update`` drains them.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    dpg = None
    DPG_AVAILABLE = False

from soma.ui.registry import register_panel
from soma.ui.state import UIState
from soma.ui.training_controller import CHANNEL_GROWTH_EVENT, CHANNEL_METRICS

# How many recent points each plot retains.
_WINDOW = 500


def read_metrics_tail(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    """Return the last ``max_records`` valid JSON records from a JSONL file.

    Used by spectator mode (autonomous loop) when the metrics panel reads
    from disk instead of subscribing to the in-process bus. Missing files
    return an empty list. Malformed lines are silently skipped.
    """
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for raw in lines[-max_records * 2 :]:
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-max_records:]

_GROWTH_COLORS = {
    "synaptogenesis": (80, 200, 255, 200),
    "neurogenesis": (100, 255, 100, 200),
    "pruning": (255, 140, 60, 200),
}


@register_panel("metrics")
class MetricsPanel:
    name = "metrics"
    display_label = "Live metrics"

    def __init__(self) -> None:
        self._root: int | str | None = None
        self._lock = threading.RLock()
        self._metrics_queue: deque[dict[str, Any]] = deque(maxlen=_WINDOW * 2)
        self._growth_queue: deque[dict[str, Any]] = deque(maxlen=_WINDOW)
        self._steps: list[float] = []
        self._loss: list[float] = []
        self._curiosity: list[float] = []
        self._lr_mul: list[float] = []
        self._num_nodes: list[float] = []
        self._num_edges: list[float] = []
        self._wm_occ: list[float] = []
        self._autonomous_metrics_file: Path | None = None
        self._last_disk_step: int = -1

    # ------------------------------------------------------------------
    def build(self, parent: int | str, state: UIState) -> int | str:
        if not DPG_AVAILABLE:  # pragma: no cover
            raise RuntimeError("DearPyGUI not installed; install soma[ui]")

        self._root = dpg.add_group(parent=parent, tag="panel_metrics_root")

        def _plot(title: str, y_label: str, tag_prefix: str, height: int = 130) -> None:
            with dpg.plot(label=title, height=height, width=-1, parent=self._root):
                dpg.add_plot_axis(dpg.mvXAxis, label="global_step", tag=f"{tag_prefix}_x")
                with dpg.plot_axis(dpg.mvYAxis, label=y_label, tag=f"{tag_prefix}_y"):
                    dpg.add_line_series([], [], tag=f"{tag_prefix}_series")

        _plot("Loss (rolling)", "loss", "metrics_loss")
        _plot("Curiosity", "curiosity", "metrics_curiosity")
        _plot("LR multiplier", "lr_mul", "metrics_lrmul")
        with dpg.plot(
            label="Graph size",
            height=130,
            width=-1,
            parent=self._root,
            tag="metrics_size_plot",
        ):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="global_step", tag="metrics_size_x")
            with dpg.plot_axis(dpg.mvYAxis, label="count", tag="metrics_size_y"):
                dpg.add_line_series([], [], tag="metrics_size_nodes", label="nodes")
                dpg.add_line_series([], [], tag="metrics_size_edges", label="edges")
        _plot("WM occupancy", "fraction", "metrics_wm")

        # Growth event strip plot — render as a scatter with colored markers.
        with dpg.plot(label="Growth events", height=130, width=-1, parent=self._root):
            dpg.add_plot_axis(dpg.mvXAxis, label="global_step", tag="growth_x")
            with dpg.plot_axis(dpg.mvYAxis, label="type", no_tick_labels=True, tag="growth_y"):
                dpg.add_scatter_series([], [], tag="growth_series_syn", label="syn")
                dpg.add_scatter_series([], [], tag="growth_series_neuro", label="neuro")
                dpg.add_scatter_series([], [], tag="growth_series_prune", label="prune")
            dpg.set_axis_limits("growth_y", -0.5, 2.5)

        if state.autonomous_mode:
            # In spectator mode the training service owns SOMA and writes
            # metrics.current.jsonl; update() tails that file instead of
            # listening for bus messages.
            self._autonomous_metrics_file = (
                state.soma_loop_dir / "metrics" / "metrics.current.jsonl"
            )
        else:
            state.bus.subscribe(CHANNEL_METRICS, self._on_metrics)
            state.bus.subscribe(CHANNEL_GROWTH_EVENT, self._on_growth)
        return self._root

    # ------------------------------------------------------------------
    # Subscribers (worker thread) — just buffer, let update() draw.
    # ------------------------------------------------------------------
    def _on_metrics(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._metrics_queue.append(record)

    def _on_growth(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._growth_queue.append(record)

    def _pull_from_disk(self) -> None:
        """Tail the autonomous-loop metrics file and enqueue new records."""
        assert self._autonomous_metrics_file is not None
        records = read_metrics_tail(self._autonomous_metrics_file, max_records=_WINDOW)
        # The disk file stores ``step`` while the in-process bus uses
        # ``global_step``. Map and dedupe against what we've already consumed.
        with self._lock:
            for rec in records:
                step = int(rec.get("step", rec.get("global_step", 0)))
                if step <= self._last_disk_step:
                    continue
                self._last_disk_step = step
                self._metrics_queue.append(
                    {
                        "global_step": step,
                        "loss": rec.get("loss"),
                        "curiosity": rec.get("curiosity", 0.0),
                        "lr_multiplier": rec.get("lr_multiplier", 1.0),
                        "num_nodes": rec.get("num_nodes", 0),
                        "num_edges": rec.get("num_edges", 0),
                        "wm_occupancy": rec.get("wm_occupancy", 0.0),
                    }
                )

    # ------------------------------------------------------------------
    # Per-frame update (DPG main thread)
    # ------------------------------------------------------------------
    def update(self, state: UIState) -> None:
        if not DPG_AVAILABLE:  # pragma: no cover
            return
        if self._autonomous_metrics_file is not None:
            self._pull_from_disk()
        with self._lock:
            metrics = list(self._metrics_queue)
            growth = list(self._growth_queue)
            self._metrics_queue.clear()
        for rec in metrics:
            self._steps.append(float(rec["global_step"]))
            if rec["loss"] is not None:
                self._loss.append(float(rec["loss"]))
            else:
                self._loss.append(float("nan"))
            self._curiosity.append(float(rec["curiosity"]))
            self._lr_mul.append(float(rec["lr_multiplier"]))
            self._num_nodes.append(float(rec["num_nodes"]))
            self._num_edges.append(float(rec["num_edges"]))
            self._wm_occ.append(float(rec["wm_occupancy"]))
        self._truncate()
        self._redraw_lines()
        self._redraw_growth(growth)

    def _truncate(self) -> None:
        for series in (
            self._steps,
            self._loss,
            self._curiosity,
            self._lr_mul,
            self._num_nodes,
            self._num_edges,
            self._wm_occ,
        ):
            if len(series) > _WINDOW:
                del series[: len(series) - _WINDOW]

    def _redraw_lines(self) -> None:
        if not self._steps:
            return
        dpg.set_value("metrics_loss_series", [self._steps, self._loss])
        dpg.set_value("metrics_curiosity_series", [self._steps, self._curiosity])
        dpg.set_value("metrics_lrmul_series", [self._steps, self._lr_mul])
        dpg.set_value("metrics_size_nodes", [self._steps, self._num_nodes])
        dpg.set_value("metrics_size_edges", [self._steps, self._num_edges])
        dpg.set_value("metrics_wm_series", [self._steps, self._wm_occ])
        # Autoscale X to the full window.
        x_min = self._steps[0]
        x_max = self._steps[-1]
        dpg.fit_axis_data("metrics_loss_x")
        dpg.fit_axis_data("metrics_loss_y")
        dpg.fit_axis_data("metrics_curiosity_x")
        dpg.fit_axis_data("metrics_curiosity_y")
        dpg.fit_axis_data("metrics_lrmul_x")
        dpg.fit_axis_data("metrics_lrmul_y")
        dpg.fit_axis_data("metrics_size_x")
        dpg.fit_axis_data("metrics_size_y")
        dpg.fit_axis_data("metrics_wm_x")
        dpg.fit_axis_data("metrics_wm_y")
        _ = x_min, x_max

    def _redraw_growth(self, recent: list[dict[str, Any]]) -> None:
        # Bucket by type.
        categories = {"synaptogenesis": 0, "neurogenesis": 1, "pruning": 2}
        x_by: dict[str, list[float]] = {"synaptogenesis": [], "neurogenesis": [], "pruning": []}
        y_by: dict[str, list[float]] = {"synaptogenesis": [], "neurogenesis": [], "pruning": []}
        # Keep all events from the persistent queue.
        with self._lock:
            persistent = list(self._growth_queue)
        for rec in persistent:
            t = rec.get("type")
            if t not in categories:
                continue
            x_by[t].append(float(rec["global_step"]))
            y_by[t].append(float(categories[t]))
        dpg.set_value("growth_series_syn", [x_by["synaptogenesis"], y_by["synaptogenesis"]])
        dpg.set_value("growth_series_neuro", [x_by["neurogenesis"], y_by["neurogenesis"]])
        dpg.set_value("growth_series_prune", [x_by["pruning"], y_by["pruning"]])
        if persistent:
            dpg.fit_axis_data("growth_x")
