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
from typing import Any, Literal

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

    # --- Verbalizer bootstrap (Phase 4) -----------------------------------
    # LR for the verbalizer-bootstrap Adam optimizer. Smaller than base_lr
    # because the verbalizer is one dense projector, not a sparse Hebbian
    # graph, so a few good gradient steps compound quickly.
    verbalizer_lr: float = 1e-4
    verbalizer_checkpoint_interval: int = 500
    bootstrap_sample_tokens: int = 64
    bootstrap_max_steps: int = 5000

    # --- Online verbalizer training (Phase 6) ----------------------------
    # LR for per-turn online updates during chat. 10× smaller than
    # bootstrap's 1e-4 because online trains on just the last turn plus a
    # few replays — noisier signal, so gentler steps.
    online_verbalizer_lr: float = 1e-5
    online_batch_size: int = 4  # 1 latest turn + (N-1) random replays
    replay_buffer_capacity: int = 64
    divergence_window: int = 20
    # If mean(last half of window) - mean(first half) > threshold,
    # online updates freeze until reset_divergence_guard() is called.
    divergence_threshold: float = 1.0

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
    sparse_init_connectivity: float = 1.0  # 1.0 = fully connected (default)

    # --- Intervals --------------------------------------------------------
    synaptogenesis_interval: int = 100
    neurogenesis_interval: int = 500
    consolidation_interval: int = 1000
    consolidation_replay_steps: int = 100
    consolidation_error_threshold: float = 0.5
    pruning_interval: int = 1000
    checkpoint_interval: int = 5000

    # --- Growth trigger mode ---------------------------------------------
    # "interval": fire at fixed step cadence (neurogenesis_interval),
    #   gated by neurogenesis_threshold internally. Historic default.
    # "pe_gated": check every step; fire when the PE trigger ratio
    #   exceeds neurogenesis_threshold AND the cooldown has elapsed
    #   since the last firing. Tracks prediction error spikes rather
    #   than wall-clock cadence. See env_sequence_v05 findings
    #   (2026-04-18) for motivation.
    neurogenesis_mode: Literal["interval", "pe_gated"] = "interval"
    # Minimum gap between neurogenesis events in pe_gated mode.
    neurogenesis_cooldown: int = 200
    # Scale of randn-drawn initial weight for edges wired out of
    # freshly-created neurogenesis nodes. Lower values reduce the
    # immediate disturbance a new node imposes on existing circuitry
    # and let Hebbian updates grow useful weight only where there is
    # co-activation to support it. 0.01 preserves legacy behavior.
    neurogenesis_init_weight_scale: float = 0.01

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

    # --- Deployment ------------------------------------------------------
    # Headroom factor applied to detected CUDA VRAM before tier selection.
    # 0.9 (default) means "reserve 10% for OS/driver/KV-cache jitter": a
    # 23 GB raw RTX 3090 reading becomes 20 effective GB, comfortably
    # picking the ``large`` tier instead of getting unstuck at the
    # 23/24-boundary edge case where a few hundred MB of driver overhead
    # OOMs the model. Set to 1.0 to disable headroom (use full VRAM); set
    # below 0.9 (e.g., 0.7) to be more conservative when SOMA's own
    # memory footprint is unusually large for the run. Must be in (0, 1].
    # Applied at tier-selection time, not at VRAM-report time -- callers
    # using ``detect_cuda_vram`` for their own purposes still see raw GB.
    vram_safety_factor: float = 0.9

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
            "consolidation_replay_steps",
            "checkpoint_interval",
            "num_curiosity_domains",
            "max_output_tokens",
            "max_input_tokens",
            "activation_history_size",
            "verbalizer_checkpoint_interval",
            "bootstrap_sample_tokens",
            "bootstrap_max_steps",
            "online_batch_size",
            "replay_buffer_capacity",
            "divergence_window",
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
            # Growth/consolidation intervals: 0 = disabled.
            "synaptogenesis_interval",
            "neurogenesis_interval",
            "consolidation_interval",
            "pruning_interval",
            "neurogenesis_cooldown",
        ]
        for name in non_negative_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"SOMAConfig.{name} must be a non-negative int, got {value!r}")

        if self.neurogenesis_mode not in ("interval", "pe_gated"):
            raise ValueError(
                f"SOMAConfig.neurogenesis_mode must be 'interval' or 'pe_gated', "
                f"got {self.neurogenesis_mode!r}"
            )

        if self.neurogenesis_init_weight_scale < 0.0:
            raise ValueError(
                f"SOMAConfig.neurogenesis_init_weight_scale must be >= 0, "
                f"got {self.neurogenesis_init_weight_scale!r}"
            )

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

        if not 0.0 < self.vram_safety_factor <= 1.0:
            raise ValueError(
                f"SOMAConfig.vram_safety_factor must be in (0, 1], got {self.vram_safety_factor!r}"
            )

    # ------------------------------------------------------------------
    # YAML (de)serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> SOMAConfig:
        """Load a config from a YAML file.

        Delegates to :meth:`from_dict`. Unknown keys are dropped with a
        ``UserWarning`` instead of raising, so older or newer checkpoints/YAMLs
        that include fields this SOMA version doesn't know about can still load.
        Typos in known fields are still enforced by :class:`SOMAConfig`'s
        dataclass __init__.
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

    @classmethod
    def production(cls, **overrides: Any) -> SOMAConfig:
        """Return a production-tuned config: frozen graph, no growth cycles.

        B3 CL ablations showed that SOMA's anti-forgetting property is
        architectural (the graph acts as a fixed nonlinear feature
        extractor, similar to reservoir computing). Hebbian plasticity,
        consolidation, and critical periods add compute cost without
        measurable benefit in production workloads.

        This factory disables all growth/consolidation intervals (set to
        0) so the graph is constructed once and then used read-only.
        Callers should pass ``eval_mode=True`` to ``SOMA.step()`` to
        also skip per-step Hebbian weight updates (2x faster).

        Any keyword argument overrides the production default, so you
        can still enable specific research features::

            cfg = SOMAConfig.production(consolidation_interval=500)
        """
        defaults: dict[str, Any] = {
            "synaptogenesis_interval": 0,
            "neurogenesis_interval": 0,
            "consolidation_interval": 0,
            "pruning_interval": 0,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def developmental(cls, **overrides: Any) -> SOMAConfig:
        """Return a config tuned for developmental learning.

        Optimised for small graphs (< 50 nodes) that grow through
        text interaction.  Higher synaptogenesis rate, lower
        activation threshold, and shorter growth intervals than the
        whitepaper defaults — which were designed for 50K-node graphs
        and million-step training runs.
        """
        defaults: dict[str, Any] = {
            "vocab_size": 256,
            "text_embed_dim": 128,
            "sensor_output_dim": 128,
            "max_input_tokens": 128,
            "initial_integrator_count": 4,
            "initial_associator_count": 8,
            # Faster growth cycles for interactive use
            "synaptogenesis_interval": 10,
            "synaptogenesis_rate": 2.0,
            "neurogenesis_interval": 25,
            "neurogenesis_threshold": 1.1,
            "pruning_interval": 200,
            # Cap graph size to prevent runaway growth when encoder
            # training creates non-stationary input distributions.
            "max_nodes": 50,
            "consolidation_interval": 50,
            "consolidation_replay_steps": 20,
            # Lower thresholds for developmental graphs
            "activation_threshold": 0.005,
            "sparse_init_connectivity": 0.3,
            "seed": 42,
        }
        defaults.update(overrides)
        return cls(**defaults)

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
