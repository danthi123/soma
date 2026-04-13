"""Homeostatic regulation — global LR dampening + growth gating.

Whitepaper Section 7.2. Every step, the regulator:
- Updates the loss EMA and variance EMA.
- Dampens ``global_lr_multiplier`` on 3-sigma loss spikes (stability).
- Recovers the LR gradually (x1.01 per step, capped at 1.0).
- Publishes ``allow_neurogenesis`` / ``allow_synaptogenesis`` gates
  based on the current graph density.
"""

from __future__ import annotations

import math

from soma.core.config import SOMAConfig
from soma.core.graph import Graph


class HomeostaticRegulator:
    """Tracks global learning signal stability and gates structural growth."""

    def __init__(
        self,
        max_nodes: int = 50_000,
        max_edges_per_node: float = 20.0,
        *,
        spike_sigma: float = 3.0,
        spike_dampen: float = 0.5,
        recovery_factor: float = 1.01,
        ema_decay: float = 0.99,
        warmup_steps: int = 20,
    ) -> None:
        if max_nodes <= 0:
            raise ValueError(f"max_nodes must be positive, got {max_nodes}")
        if max_edges_per_node <= 0.0:
            raise ValueError(f"max_edges_per_node must be positive, got {max_edges_per_node}")
        if spike_sigma <= 0.0:
            raise ValueError(f"spike_sigma must be positive, got {spike_sigma}")
        if not 0.0 < spike_dampen <= 1.0:
            raise ValueError(f"spike_dampen must be in (0, 1], got {spike_dampen}")
        if recovery_factor < 1.0:
            raise ValueError(f"recovery_factor must be >= 1.0, got {recovery_factor}")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        self.spike_sigma = spike_sigma
        self.spike_dampen = spike_dampen
        self.recovery_factor = recovery_factor
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps

        self.loss_ema: float = 0.0
        self.loss_variance_ema: float = 0.0
        self.global_lr_multiplier: float = 1.0
        self.allow_neurogenesis: bool = True
        self.allow_synaptogenesis: bool = True
        self._update_count: int = 0

    @classmethod
    def from_config(cls, config: SOMAConfig) -> HomeostaticRegulator:
        return cls(
            max_nodes=config.max_nodes,
            max_edges_per_node=config.max_edges_per_node,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, graph: Graph, current_loss: float) -> float | None:
        """Update the internal state for this step; return the LR multiplier.

        ``current_loss`` should be a finite non-negative scalar (MSE, cross-
        entropy, etc.). If non-finite, returns ``None`` and leaves all state
        unchanged so the caller can skip the learning step gracefully
        without corrupting the loss EMA / variance / step counter.
        """
        if not math.isfinite(current_loss):
            return None

        # EMA and variance of loss.
        prev_loss_ema = self.loss_ema
        self.loss_ema = self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * current_loss
        deviation_sq = (current_loss - prev_loss_ema) ** 2
        self.loss_variance_ema = (
            self.ema_decay * self.loss_variance_ema + (1.0 - self.ema_decay) * deviation_sq
        )
        std = math.sqrt(self.loss_variance_ema + 1e-8)

        # Spike detection and recovery. During warmup the EMA/variance are
        # still settling, so the spike test is unreliable — we skip it.
        self._update_count += 1
        if self._update_count > self.warmup_steps and current_loss > (
            prev_loss_ema + self.spike_sigma * std
        ):
            self.global_lr_multiplier *= self.spike_dampen
        else:
            self.global_lr_multiplier = min(1.0, self.global_lr_multiplier * self.recovery_factor)

        # Floor the multiplier so repeated spikes can't drive it to 0
        # and freeze learning. Recovery always has something to multiply
        # back up. Ratified from tick-1776108773 proposal that was
        # stashed when safety_gate's test phase failed.
        self.global_lr_multiplier = max(0.01, self.global_lr_multiplier)

        # Density-based growth gating.
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        avg_edges = num_edges / max(num_nodes, 1)
        self.allow_neurogenesis = num_nodes < self.max_nodes
        self.allow_synaptogenesis = avg_edges < self.max_edges_per_node

        return self.global_lr_multiplier

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all tracked state."""
        self.loss_ema = 0.0
        self.loss_variance_ema = 0.0
        self.global_lr_multiplier = 1.0
        self.allow_neurogenesis = True
        self.allow_synaptogenesis = True
        self._update_count = 0

    def state_dict(self) -> dict[str, float | bool | int]:
        """Return a plain dict of current state for serialization."""
        return {
            "loss_ema": self.loss_ema,
            "loss_variance_ema": self.loss_variance_ema,
            "global_lr_multiplier": self.global_lr_multiplier,
            "allow_neurogenesis": self.allow_neurogenesis,
            "allow_synaptogenesis": self.allow_synaptogenesis,
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict[str, float | bool | int]) -> None:
        self.loss_ema = float(state["loss_ema"])
        self.loss_variance_ema = float(state["loss_variance_ema"])
        self.global_lr_multiplier = float(state["global_lr_multiplier"])
        self.allow_neurogenesis = bool(state["allow_neurogenesis"])
        self.allow_synaptogenesis = bool(state["allow_synaptogenesis"])
        self._update_count = int(state.get("update_count", 0))
