"""Main DearPyGUI application shell.

Wires together config + bus + training controller + all registered
panels. Entry point is :func:`run` (called by ``scripts/ui.py``).

Layout overview (top-down, roughly docked)::

    +------------------------------------------------------------+
    | Menu bar: File / Session / Help                            |
    +---------------------+------------------+-------------------+
    | Config (left)       | Live metrics     | Chat (right)      |
    |                     | (center-top)     |                   |
    |                     +------------------+                   |
    |                     | Graph view       |                   |
    |                     | (center-bottom)  |                   |
    +---------------------+------------------+-------------------+
    | Controls + status bar (bottom)                             |
    +------------------------------------------------------------+

Each panel registers itself via the :mod:`soma.ui.registry`. The app
instantiates them via ``default_registry.build(name)`` so adding a new
panel is just "create a module + register".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    dpg = None
    DPG_AVAILABLE = False

# Importing panels triggers registration.
from soma.core.config import SOMAConfig
from soma.ui import panels  # noqa: F401  (side-effect import for registration)
from soma.ui.bus import DataBus
from soma.ui.registry import Panel, default_registry
from soma.ui.state import UIState
from soma.ui.training_controller import TrainingController

LOGGER = logging.getLogger("soma.ui.app")


@dataclass
class AppOptions:
    config_path: str | None = None
    window_title: str = "SOMA Control Center"
    window_width: int = 1600
    window_height: int = 960


# Named tags so tests / panels can find them.
_MAIN_WINDOW = "soma_main_window"


def build_state(options: AppOptions) -> UIState:
    """Build a fresh UIState (config from YAML if given, else defaults)."""
    config = SOMAConfig.from_yaml(options.config_path) if options.config_path else SOMAConfig()
    bus = DataBus()
    controller = TrainingController(bus)
    return UIState(config=config, bus=bus, controller=controller)


def _build_panels(state: UIState) -> list[tuple[str, Panel]]:
    """Instantiate every registered panel and return [(name, panel), ...] in display order."""
    ordered = [
        "controls",
        "config",
        "metrics",
        "graph",
        "chat",
    ]
    result: list[tuple[str, Panel]] = []
    for name in ordered:
        if name in default_registry:
            result.append((name, default_registry.build(name)))
    # Also pick up any panels registered beyond the built-in set (extensibility).
    for name in default_registry.names():
        if name not in ordered:
            result.append((name, default_registry.build(name)))
    return result


def _build_layout(state: UIState, built_panels: list[tuple[str, Panel]]) -> None:
    """Create the DPG widget tree."""
    assert DPG_AVAILABLE
    with dpg.window(
        label="SOMA",
        tag=_MAIN_WINDOW,
        width=-1,
        height=-1,
        no_close=True,
        no_collapse=True,
    ):
        with dpg.menu_bar(), dpg.menu(label="Session"):
            dpg.add_menu_item(
                label="Start",
                callback=lambda s, a, u: state.controller.start(),
            )
            dpg.add_menu_item(
                label="Stop",
                callback=lambda s, a, u: state.controller.stop(),
            )
            dpg.add_menu_item(
                label="Reset",
                callback=lambda s, a, u: state.controller.reset(),
            )

        # Controls live at the TOP of the window in a fixed-height strip so
        # they're always reachable without scrolling past the (often tall)
        # metrics plots below.
        with dpg.child_window(
            height=260,
            border=True,
            tag="row_top",
        ):
            _build_panel(state, built_panels, "controls", parent="row_top")

        # Three-column layout fills the remaining vertical space.
        with dpg.group(horizontal=True, tag="main_columns"):
            # Left column: config.
            with dpg.child_window(width=360, height=-1, border=True, tag="col_left"):
                _build_panel(state, built_panels, "config", parent="col_left")

            # Center column: metrics / graph tabs.
            with (
                dpg.child_window(width=760, height=-1, border=True, tag="col_center"),
                dpg.tab_bar(tag="center_tabs"),
            ):
                with dpg.tab(label="Live metrics", tag="tab_metrics"):
                    _build_panel(state, built_panels, "metrics", parent="tab_metrics")
                with dpg.tab(label="Graph view", tag="tab_graph"):
                    _build_panel(state, built_panels, "graph", parent="tab_graph")

            # Right column: chat.
            with dpg.child_window(width=-1, height=-1, border=True, tag="col_right"):
                _build_panel(state, built_panels, "chat", parent="col_right")

        # Any custom panels that weren't placed explicitly go into a floating tab bar.
        builtins = {"config", "metrics", "graph", "chat", "controls"}
        extras = [p for name, p in built_panels if name not in builtins]
        if extras:
            with (
                dpg.child_window(border=True, tag="row_extras", height=240),
                dpg.tab_bar(tag="extras_tabs"),
            ):
                for idx, panel in enumerate(extras):
                    tab_tag = f"tab_extra_{idx}"
                    with dpg.tab(label=panel.display_label, tag=tab_tag):
                        panel.build(parent=tab_tag, state=state)


def _build_panel(
    state: UIState,
    built_panels: list[tuple[str, Panel]],
    name: str,
    *,
    parent: int | str,
) -> None:
    for panel_name, panel in built_panels:
        if panel_name == name:
            panel.build(parent=parent, state=state)
            return


def _frame_update(panels_list: list[tuple[str, Panel]], state: UIState) -> None:
    """Per-frame callback: ask every panel to redraw any pending data."""
    for _, panel in panels_list:
        try:
            panel.update(state)
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("panel %r update raised", panel)


def run(options: AppOptions | None = None) -> None:
    """Launch the DPG app. Blocks until the user closes the window."""
    if not DPG_AVAILABLE:
        raise RuntimeError("DearPyGUI not installed; install soma[ui]")
    opts = options or AppOptions()
    state = build_state(opts)
    built = _build_panels(state)

    dpg.create_context()
    try:
        dpg.create_viewport(
            title=opts.window_title,
            width=opts.window_width,
            height=opts.window_height,
        )
        dpg.setup_dearpygui()
        _build_layout(state, built)
        dpg.set_primary_window(_MAIN_WINDOW, True)
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            _frame_update(built, state)
            dpg.render_dearpygui_frame()
    finally:
        state.controller.shutdown()
        dpg.destroy_context()


# Exposed for tests.
def build_app_for_tests(
    options: AppOptions | None = None,
) -> tuple[UIState, list[tuple[str, Panel]]]:
    """Return (state, panels) without creating a DPG context.

    The test suite exercises panel wiring and ``update`` dispatch without
    ever starting the viewport. This keeps CI headless-safe.
    """
    opts = options or AppOptions()
    state = build_state(opts)
    built = _build_panels(state)
    return state, built
