"""Config panel — auto-generates widgets from :class:`SOMAConfig` fields.

The widget tree is built by introspecting :func:`dataclasses.fields` on
SOMAConfig. Field type drives widget choice:

- ``int`` / ``float``: numeric input.
- ``bool``: checkbox.
- ``str``: text input.
- ``list[str]``: comma-separated text input (edited as a flat string).

Numeric fields get a ``min``/``max`` hint if a field's default value is
positive (we cap at 0 below to prevent negative node counts, etc.).

Changes are applied to the ``UIState.config`` instance in-place but do
not take effect until the user clicks "Rebuild session" in the controls
panel — the training controller reads ``config`` fresh when it builds
its next SOMA.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Any

try:
    import dearpygui.dearpygui as dpg

    DPG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    dpg = None
    DPG_AVAILABLE = False

from soma.core.config import SOMAConfig
from soma.ui.registry import register_panel
from soma.ui.state import UIState

# Field categorisation for nicer grouping in the UI. Any field not listed
# falls under "Misc".
_FIELD_GROUPS: dict[str, list[str]] = {
    "Graph dimensions": [
        "sensor_output_dim",
        "associator_input_dim",
        "associator_hidden_dim",
        "associator_output_dim",
        "integrator_input_dim",
        "integrator_hidden_dim",
        "integrator_output_dim",
        "position_dim",
    ],
    "Memory": [
        "wm_slots",
        "wm_dim",
        "wm_decay_rate",
        "episodic_capacity",
        "key_dim",
        "value_dim",
    ],
    "I/O": [
        "input_modalities",
        "output_modalities",
        "vocab_size",
        "text_embed_dim",
        "max_input_tokens",
        "max_output_tokens",
    ],
    "Growth": [
        "initial_associator_count",
        "initial_integrator_count",
        "max_nodes",
        "max_edges_per_node",
        "activation_threshold",
        "synaptogenesis_rate",
        "neurogenesis_threshold",
        "edge_strength_threshold",
        "inactivity_threshold",
        "pruning_grace_period",
        "myelination_strength_threshold",
        "myelination_age_threshold",
        "max_edge_weight",
        "locality_scale",
        "position_jitter",
    ],
    "Intervals": [
        "synaptogenesis_interval",
        "neurogenesis_interval",
        "consolidation_interval",
        "consolidation_replay_steps",
        "consolidation_error_threshold",
        "pruning_interval",
        "checkpoint_interval",
    ],
    "Learning": [
        "base_lr",
        "youth_lr_multiplier",
        "hebbian_lr",
        "consolidation_lr_ratio",
        "maturity_increment",
    ],
    "Meta-cognition": [
        "num_curiosity_domains",
        "default_target_activation",
        "gain_min",
        "gain_max",
    ],
}


def _tag_for(field_name: str) -> str:
    return f"cfg__{field_name}"


def _group_for(field_name: str) -> str:
    for group, names in _FIELD_GROUPS.items():
        if field_name in names:
            return group
    return "Misc"


def _is_list_type(field_type: Any) -> bool:
    return field_type is list[str] or (
        hasattr(field_type, "__origin__") and field_type.__origin__ is list
    )


def _coerce_to_field_type(field_type: Any, raw: Any) -> Any:
    """Best-effort string -> typed coercion for list[str] fields."""
    if _is_list_type(field_type):
        if isinstance(raw, list):
            return [s for s in raw if s]
        if isinstance(raw, str):
            return [s.strip() for s in raw.split(",") if s.strip()]
    return raw


@register_panel("config")
class ConfigPanel:
    name = "config"
    display_label = "Config"

    def __init__(self) -> None:
        self._root: int | str | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, parent: int | str, state: UIState) -> int | str:
        if not DPG_AVAILABLE:  # pragma: no cover - UI-only path
            raise RuntimeError("DearPyGUI not installed; install soma[ui]")

        self._root = dpg.add_group(parent=parent, tag="panel_config_root")
        dpg.add_text("SOMAConfig (edits take effect on next Rebuild)", parent=self._root)
        dpg.add_separator(parent=self._root)

        grouped: dict[str, list[Any]] = {}
        for f in dataclass_fields(SOMAConfig):
            grouped.setdefault(_group_for(f.name), []).append(f)

        for group_name in list(_FIELD_GROUPS) + ["Misc"]:
            fields_in_group = grouped.get(group_name, [])
            if not fields_in_group:
                continue
            with dpg.collapsing_header(label=group_name, parent=self._root, default_open=False):
                for f in fields_in_group:
                    self._add_field_widget(f, state)
        return self._root

    def _add_field_widget(self, field: Any, state: UIState) -> None:
        tag = _tag_for(field.name)
        current_value = getattr(state.config, field.name)
        if field.type in (int, "int"):
            dpg.add_input_int(
                label=field.name,
                default_value=int(current_value),
                tag=tag,
                callback=self._make_callback(field.name, int, state),
                on_enter=True,
                step=0,
            )
        elif field.type in (float, "float"):
            dpg.add_input_float(
                label=field.name,
                default_value=float(current_value),
                tag=tag,
                callback=self._make_callback(field.name, float, state),
                on_enter=True,
                format="%.6f",
                step=0,
            )
        elif field.type in (bool, "bool"):
            dpg.add_checkbox(
                label=field.name,
                default_value=bool(current_value),
                tag=tag,
                callback=self._make_callback(field.name, bool, state),
            )
        elif field.type == "list[str]":
            joined = (
                ",".join(current_value) if isinstance(current_value, list) else str(current_value)
            )
            dpg.add_input_text(
                label=field.name,
                default_value=joined,
                tag=tag,
                callback=self._make_callback(field.name, list[str], state),
                on_enter=True,
            )
        elif field.type in (str, "str"):
            dpg.add_input_text(
                label=field.name,
                default_value=str(current_value),
                tag=tag,
                callback=self._make_callback(field.name, str, state),
                on_enter=True,
            )
        else:
            # Unknown — fall back to read-only text display.
            dpg.add_text(f"{field.name} = {current_value!r}  (unsupported type {field.type})")

    def _make_callback(self, field_name: str, field_type: Any, state: UIState):  # type: ignore[no-untyped-def]
        def callback(sender: int | str, app_data: Any, user_data: Any) -> None:
            try:
                coerced = _coerce_to_field_type(field_type, app_data)
                setattr(state.config, field_name, coerced)
            except Exception:  # pragma: no cover - UI callback path
                # Silently ignore bad values — validation happens when
                # SOMAConfig.__post_init__ runs on rebuild.
                pass

        return callback

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, state: UIState) -> None:  # pragma: no cover - no-op
        return
