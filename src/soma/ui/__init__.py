"""SOMA desktop UI package.

A DearPyGUI-based control center that wraps :class:`soma.system.SOMA` with:
- a training controller (start / stop / pause / reset on a worker thread),
- live metric plots (loss, curiosity, LR multiplier, memory, growth events),
- a 2D force-directed graph viewer with tunable recompute intervals,
- an interactive chat panel,
- a config-editor panel that reads and writes :class:`soma.core.config.SOMAConfig`.

The UI is intentionally decoupled from the engine: every source of live
data flows through :class:`soma.ui.bus.DataBus`, and every panel registers
itself via the plugin-style :mod:`soma.ui.registry` so new panels can be
added without editing the main layout.

Importing this package has no side effects beyond pure-Python module
loading — DPG context is only created when :func:`soma.ui.app.run` is
called.
"""

from soma.ui.bus import DataBus, DataChannel
from soma.ui.state import UIState
from soma.ui.training_controller import TrainingCommand, TrainingController, TrainingState

__all__ = [
    "DataBus",
    "DataChannel",
    "TrainingCommand",
    "TrainingController",
    "TrainingState",
    "UIState",
]
