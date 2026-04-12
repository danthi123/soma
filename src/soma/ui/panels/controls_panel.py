"""Controls panel — start / pause / stop / step / save-checkpoint buttons.

Also houses session-level actions: load corpus, rebuild SOMA (applies
config changes), load checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path

try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except ImportError:  # pragma: no cover
    dpg = None
    DPG_AVAILABLE = False

from soma.ui.registry import register_panel
from soma.ui.state import UIState
from soma.ui.training_controller import SessionSpec, TrainingState, load_corpus


@register_panel("controls")
class ControlsPanel:
    name = "controls"
    display_label = "Controls"

    def __init__(self) -> None:
        self._root: int | str | None = None
        # live state/step; overwritten every frame
        self._state_tag = "controls_state_label"
        # last action message; only overwritten by _set_status
        self._status_tag = "controls_status_label"

    # ------------------------------------------------------------------
    def build(self, parent: int | str, state: UIState) -> int | str:
        if not DPG_AVAILABLE:  # pragma: no cover
            raise RuntimeError("DearPyGUI not installed; install soma[ui]")

        self._root = dpg.add_group(parent=parent, tag="panel_controls_root")
        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_button(label="Start", tag="btn_start", callback=self._on_start, user_data=state)
            dpg.add_button(label="Pause", tag="btn_pause", callback=self._on_pause, user_data=state)
            dpg.add_button(
                label="Resume", tag="btn_resume", callback=self._on_resume, user_data=state
            )
            dpg.add_button(label="Stop", tag="btn_stop", callback=self._on_stop, user_data=state)
            dpg.add_button(
                label="Step once",
                tag="btn_step",
                callback=self._on_step,
                user_data=state,
            )
            dpg.add_button(
                label="Reset session",
                tag="btn_reset",
                callback=self._on_reset,
                user_data=state,
            )

        dpg.add_separator(parent=self._root)
        dpg.add_text("Session", parent=self._root)
        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_input_text(
                label="Corpus path",
                tag="controls_corpus_path",
                default_value=state.last_corpus_path or "data/tinyshakespeare.txt",
                width=400,
            )
            dpg.add_input_int(
                label="Vocab size",
                tag="controls_vocab_size",
                default_value=512,
                min_value=16,
                max_value=65_536,
                min_clamped=True,
                max_clamped=True,
                step=0,
                width=150,
            )
        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_input_int(
                label="Step budget (0 = unlimited)",
                tag="controls_step_budget",
                default_value=0,
                min_value=0,
                min_clamped=True,
                step=0,
                width=200,
            )
            dpg.add_input_int(
                label="Snapshot every",
                tag="controls_snapshot_every",
                default_value=100,
                min_value=1,
                min_clamped=True,
                step=0,
                width=200,
            )
            dpg.add_combo(
                label="Device",
                items=_available_devices(),
                default_value=_default_device(),
                tag="controls_device",
                width=100,
            )

        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_button(
                label="Apply & rebuild",
                tag="btn_rebuild",
                callback=self._on_rebuild,
                user_data=state,
            )

        dpg.add_separator(parent=self._root)
        dpg.add_text("Checkpoint", parent=self._root)
        with dpg.group(parent=self._root, horizontal=True):
            dpg.add_input_text(
                label="Path",
                tag="controls_ckpt_path",
                default_value=state.last_checkpoint_path or "checkpoints/soma_session.pt",
                width=400,
            )
            dpg.add_button(
                label="Save",
                tag="btn_save_ckpt",
                callback=self._on_save,
                user_data=state,
            )
            dpg.add_button(
                label="Load",
                tag="btn_load_ckpt",
                callback=self._on_load,
                user_data=state,
            )

        dpg.add_separator(parent=self._root)
        dpg.add_text("state=idle  step=0", tag=self._state_tag, parent=self._root)
        dpg.add_text("", tag=self._status_tag, parent=self._root)
        return self._root

    # ------------------------------------------------------------------
    def update(self, state: UIState) -> None:
        if not DPG_AVAILABLE:  # pragma: no cover
            return
        ctrl_state = state.controller.state
        soma = state.controller.soma
        step = soma.global_step if soma is not None else 0
        # Only the live state line is refreshed every frame; the _status_tag
        # line preserves the last action message ("session ready", "save failed",
        # etc.) until another action overwrites it.
        dpg.set_value(self._state_tag, f"state={ctrl_state.value}  step={step}")

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def _on_start(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        state = user_data
        # If the controller has never been configured, build a session from
        # the current controls inputs.
        if state.controller._spec is None:
            self._rebuild_session(state)
        state.controller.start()

    def _on_pause(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        user_data.controller.pause()

    def _on_resume(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        user_data.controller.resume()

    def _on_stop(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        user_data.controller.stop()

    def _on_step(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        user_data.controller.step_once()

    def _on_reset(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        user_data.controller.reset()

    def _on_rebuild(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        self._rebuild_session(user_data)

    def _on_save(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        path = dpg.get_value("controls_ckpt_path")
        try:
            user_data.controller.save_checkpoint(Path(path))
            user_data.last_checkpoint_path = path
        except RuntimeError as exc:
            self._set_status(f"save failed: {exc}")

    def _on_load(self, sender, app_data, user_data: UIState) -> None:  # type: ignore[no-untyped-def]
        path = dpg.get_value("controls_ckpt_path")
        try:
            user_data.controller.load_checkpoint(Path(path))
            user_data.last_checkpoint_path = path
        except (RuntimeError, FileNotFoundError) as exc:
            self._set_status(f"load failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rebuild_session(self, state: UIState) -> None:
        corpus_path = Path(dpg.get_value("controls_corpus_path"))
        vocab_size = int(dpg.get_value("controls_vocab_size"))
        step_budget = int(dpg.get_value("controls_step_budget"))
        snapshot_every = int(dpg.get_value("controls_snapshot_every"))
        device = str(dpg.get_value("controls_device"))
        try:
            corpus = load_corpus(corpus_path)
        except (FileNotFoundError, ValueError) as exc:
            self._set_status(f"corpus error: {exc}")
            return
        spec = SessionSpec(
            config=state.config,
            corpus_texts=corpus,
            vocab_size=vocab_size,
            step_budget=step_budget,
            snapshot_every=snapshot_every,
            device=device,
        )
        # Stop first if running; otherwise configure will reject.
        if state.controller.state is TrainingState.RUNNING:
            state.controller.stop()
        state.controller.reset()
        state.controller.configure(spec)
        state.last_corpus_path = str(corpus_path)
        self._set_status(
            f"session ready - corpus={len(corpus)} lines, vocab={vocab_size}, device={device}"
        )

    def _set_status(self, msg: str) -> None:
        if DPG_AVAILABLE:
            timestamp = time.strftime("%H:%M:%S")
            dpg.set_value(self._status_tag, f"[{timestamp}] {msg}")


def _available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:  # pragma: no cover
        pass
    return devices


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"
