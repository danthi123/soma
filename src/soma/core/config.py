"""SOMAConfig: central configuration dataclass and YAML loader.

The config holds all hyperparameters for the SOMA system. Values come from:
1. Dataclass defaults (this file) — match whitepaper Section 10.
2. YAML overrides (configs/default.yaml) — loaded via ``SOMAConfig.from_yaml``.

Design notes:
- SOMAConfig is a plain ``dataclass``, not an ``nn.Module``; it has no learnable
  parameters. Pass it into modules that need it.
- ``global_step`` lives on SOMA, not here; SOMAConfig is static per-run.
- Whitepaper uppercase constants (e.g., ``BASE_LR``) map to lowercase attributes
  (``base_lr``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SOMAConfig:
    """Central hyperparameter container for the SOMA system.

    All defaults mirror the whitepaper (Section 10). Override via YAML
    or by constructing with keyword arguments.
    """

    # --- Graph Dimensions -------------------------------------------------
    sensor_output_dim: int = 64
    associator_input_dim: int = 64
    associator_hidden_dim: int = 128
    associator_output_dim: int = 64
    integrator_input_dim: int = 128
    integrator_hidden_dim: int = 256
    integrator_output_dim: int = 128
    position_dim: int = 16

    # --- Memory Systems ---------------------------------------------------
    wm_slots: int = 32
    wm_dim: int = 128
    wm_decay_rate: float = 0.95
    episodic_capacity: int = 10000
    key_dim: int = 128
    value_dim: int = 256

    # --- I/O --------------------------------------------------------------
    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    vocab_size: int = 8192
    text_embed_dim: int = 64

    # --- Growth -----------------------------------------------------------
    initial_associator_count: int = 32
    initial_integrator_count: int = 8
    max_nodes: int = 50000
    max_edges_per_node: float = 20.0

    # --- Learning ---------------------------------------------------------
    base_lr: float = 0.001
    youth_lr_multiplier: float = 3.0
    hebbian_lr: float = 0.0001
    consolidation_lr_ratio: float = 0.1
    maturity_increment: float = 0.0001
    edge_weight_decay: float = 0.9999
    grad_clip_max_norm: float = 1.0
    max_consecutive_skipped_steps: int = 50

    # --- Growth Thresholds ------------------------------------------------
    activation_threshold: float = 0.1
    synaptogenesis_rate: float = 0.01
    neurogenesis_threshold: float = 1.2
    edge_strength_threshold: float = 0.001
    inactivity_threshold: int = 5000
    pruning_grace_period: int = 2000
    myelination_strength_threshold: float = 0.5
    myelination_age_threshold: int = 10000
    max_edge_weight: float = 5.0
    locality_scale: float = 2.0
    position_jitter: float = 0.1

    # --- Intervals --------------------------------------------------------
    synaptogenesis_interval: int = 100
    neurogenesis_interval: int = 500
    consolidation_interval: int = 1000
    consolidation_replay_steps: int = 100
    consolidation_error_threshold: float = 0.5
    pruning_interval: int = 1000
    checkpoint_interval: int = 5000

    # --- Curiosity --------------------------------------------------------
    num_curiosity_domains: int = 8

    # --- Homeostasis ------------------------------------------------------
    default_target_activation: float = 0.5
    gain_min: float = 0.1
    gain_max: float = 10.0

    # --- Sequence Processing ---------------------------------------------
    max_output_tokens: int = 128
    max_input_tokens: int = 256

    # --- Misc -------------------------------------------------------------
    # Size of each node's activation history ring buffer. Used for recent
    # activity statistics, myelination chain detection, etc.
    activation_history_size: int = 64

    # Random seed applied during system construction (None = nondeterministic).
    seed: int | None = None

    # --- Execution path --------------------------------------------------
    # When True (default), ``SOMA.step`` and ``consolidation_cycle`` use the
    # wave-batched :func:`soma.core.execution.execute_graph_batched`; when
    # False they fall back to the sequential :func:`execute_graph`. Flip to
    # False to A/B test or to recover if the batched path ever misbehaves.
    # Both paths are proven per-step equivalent in
    # ``tests/test_core/test_execution_parity.py``.
    use_batched_executor: bool = True

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """Raise ``ValueError`` if invariants are broken.

        We keep this light: only catch values that would break downstream
        modules (e.g., a negative capacity). Out-of-range learning rates
        are permitted — they're user choices.
        """
        positive_ints = [
            "sensor_output_dim",
            "associator_input_dim",
            "associator_hidden_dim",
            "associator_output_dim",
            "integrator_input_dim",
            "integrator_hidden_dim",
            "integrator_output_dim",
            "position_dim",
            "wm_slots",
            "wm_dim",
            "episodic_capacity",
            "key_dim",
            "value_dim",
            "vocab_size",
            "text_embed_dim",
            "max_nodes",
            "synaptogenesis_interval",
            "neurogenesis_interval",
            "consolidation_interval",
            "consolidation_replay_steps",
            "pruning_interval",
            "checkpoint_interval",
            "num_curiosity_domains",
            "max_output_tokens",
            "max_input_tokens",
            "activation_history_size",
        ]
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"SOMAConfig.{name} must be a positive int, got {value!r}")

        non_negative_ints = [
            "initial_associator_count",
            "initial_integrator_count",
            "inactivity_threshold",
            "pruning_grace_period",
            "myelination_age_threshold",
        ]
        for name in non_negative_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"SOMAConfig.{name} must be a non-negative int, got {value!r}")

        if not 0.0 < self.wm_decay_rate <= 1.0:
            raise ValueError(
                f"SOMAConfig.wm_decay_rate must be in (0, 1], got {self.wm_decay_rate!r}"
            )

        if self.gain_min <= 0.0 or self.gain_max <= self.gain_min:
            raise ValueError(
                f"SOMAConfig requires 0 < gain_min ({self.gain_min}) < gain_max ({self.gain_max})"
            )

        if self.max_edges_per_node <= 0.0:
            raise ValueError(
                f"SOMAConfig.max_edges_per_node must be positive, got {self.max_edges_per_node!r}"
            )

        if not self.input_modalities:
            raise ValueError("SOMAConfig.input_modalities must not be empty")
        if not self.output_modalities:
            raise ValueError("SOMAConfig.output_modalities must not be empty")

        if not 0.0 < self.edge_weight_decay <= 1.0:
            raise ValueError(
                f"SOMAConfig.edge_weight_decay must be in (0, 1], got {self.edge_weight_decay!r}"
            )
        if self.grad_clip_max_norm <= 0.0:
            raise ValueError(
                f"SOMAConfig.grad_clip_max_norm must be positive, got {self.grad_clip_max_norm!r}"
            )
        if (
            not isinstance(self.max_consecutive_skipped_steps, int)
            or self.max_consecutive_skipped_steps < 1
        ):
            raise ValueError(
                f"SOMAConfig.max_consecutive_skipped_steps must be >= 1, "
                f"got {self.max_consecutive_skipped_steps!r}"
            )

    # ------------------------------------------------------------------
    # YAML (de)serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> SOMAConfig:
        """Load a config from a YAML file.

        Only keys that match dataclass field names are accepted; unknown
        keys raise ``ValueError`` to catch typos early.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config YAML {path!s} must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SOMAConfig:
        """Build a config from a dict, dropping unknown keys with a warning.

        Tolerates unknown keys (e.g., a config field that an older/newer SOMA
        version doesn't recognize) so that checkpoints load across versions.
        Emits ``UserWarning`` naming the dropped keys.
        """
        import warnings

        known = {f.name for f in fields(cls)}
        unknown = [k for k in data if k not in known]
        if unknown:
            warnings.warn(
                f"Dropping unknown SOMAConfig fields (likely from an older/newer "
                f"checkpoint): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (YAML-friendly)."""
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Copy lists so callers can't mutate our internal state.
            if isinstance(value, list):
                value = list(value)
            result[f.name] = value
        return result

    def to_yaml(self, path: str | Path) -> None:
        """Write the config to a YAML file."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=True)
