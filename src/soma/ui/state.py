"""Central UI state container — single source of truth for the app.

Everything that multiple panels need to read or mutate lives here (bus,
training controller, graph-view settings, config overrides, etc.). The
main window wires panels by passing the same :class:`UIState` into each
one, so a panel can subscribe to bus channels or read controller status
without importing every other panel.

This mirrors the ``global_gui_state`` pattern in the sim/ project but
with explicit fields instead of a free-form dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from soma.core.config import SOMAConfig
from soma.ui.bus import DataBus
from soma.ui.training_controller import TrainingController

DEFAULT_SOMA_LOOP_DIR = Path(".soma-loop")


def detect_autonomous_mode(soma_loop_dir: Path) -> bool:
    """Return True iff ``<soma_loop_dir>/state/autonomous_mode.flag`` exists.

    The autonomous-loop operator touches this flag when starting the
    scheduled-task loop so the desktop UI knows to go read-only.
    """
    return (soma_loop_dir / "state" / "autonomous_mode.flag").exists()


@dataclass
class GraphViewSettings:
    """Tuning knobs for the live 2D graph view.

    These are adjustable at runtime from the graph panel — they don't
    need to roundtrip through SOMA or the bus.
    """

    enabled: bool = True
    recompute_every_steps: int = 100
    max_render_nodes: int = 100
    node_radius: float = 6.0
    show_edges: bool = True
    layout_iterations: int = 30


@dataclass
class UIState:
    """Top-level container shared by every panel."""

    config: SOMAConfig
    bus: DataBus
    controller: TrainingController
    graph_view: GraphViewSettings = field(default_factory=GraphViewSettings)

    # Session metadata — used by the file-menu panel.
    last_checkpoint_path: str | None = None
    last_corpus_path: str | None = None

    # Spectator mode: when True, controls/config are read-only and metrics/
    # graph/chat pull from .soma-loop state written by the autonomous loop
    # rather than the in-process training controller.
    autonomous_mode: bool = False
    soma_loop_dir: Path = field(default_factory=lambda: DEFAULT_SOMA_LOOP_DIR)

    def reset_session(self) -> None:
        """Forget per-session paths (still keeps config + bus + controller)."""
        self.last_checkpoint_path = None
        self.last_corpus_path = None
