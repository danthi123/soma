"""Edge: weighted, optionally-projected connection between two nodes.

Per whitepaper Section 3.2:
- ``weight`` is a learnable scalar (nn.Parameter) — receives gradient from
  backprop and Hebbian updates.
- ``projection`` is an optional ``nn.Linear`` used when source.output_dim
  differs from target.input_dim. Also learnable.
- Non-learnable stats (``coactivation_count``, ``strength``,
  ``last_active_step``) are updated in-place by the learning / growth code.

``transmit`` is the forward pass: project (if needed) then scale by weight.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from soma.core.utils import create_projection_if_needed, generate_uuid


class Edge(nn.Module):
    """Directed connection carrying a weighted activation from source -> target.

    Parameters
    ----------
    source_id, target_id:
        IDs of the endpoint nodes in the containing Graph.
    source_output_dim, target_input_dim:
        Feature dims of the endpoints. If they differ, an internal
        ``nn.Linear`` projection is created.
    creation_step:
        Global step at which the edge was created. Pruning respects a
        grace period relative to this value.
    initial_weight:
        Starting weight (default 0.01, per whitepaper synaptogenesis).
    edge_id:
        Optional explicit ID; defaults to a UUID.
    device:
        Optional device for all owned tensors.
    """

    def __init__(
        self,
        source_id: str,
        target_id: str,
        source_output_dim: int,
        target_input_dim: int,
        creation_step: int,
        *,
        initial_weight: float = 0.01,
        edge_id: str | None = None,
        device: torch.device | str | None = None,
        diversify: bool = False,  # reserved for future use
    ) -> None:
        super().__init__()

        if not source_id or not target_id:
            raise ValueError("Edge requires non-empty source_id and target_id")
        if source_id == target_id:
            raise ValueError(f"Edge cannot be a self-loop (both endpoints = {source_id[:8]})")
        if creation_step < 0:
            raise ValueError(f"creation_step must be non-negative, got {creation_step}")

        self.id: str = edge_id if edge_id is not None else generate_uuid()
        self.source_id: str = source_id
        self.target_id: str = target_id
        self.source_output_dim: int = source_output_dim
        self.target_input_dim: int = target_input_dim

        # Learnable scalar weight.
        self.weight = nn.Parameter(torch.tensor(float(initial_weight)))

        # Learnable projection, created iff dims differ.
        self.projection: nn.Linear | None = create_projection_if_needed(
            source_output_dim, target_input_dim, device=device
        )

        if device is not None:
            # Move the parameter (projection already handled above).
            self.weight.data = self.weight.data.to(device)

        # Non-learnable stats.
        self.coactivation_count: int = 0
        self.creation_step: int = creation_step
        self.last_active_step: int = creation_step
        self.strength: float = float(initial_weight)  # seed with |w|-ish magnitude

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, source_activation: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`transmit` so the edge can be called like a Module."""
        return self.transmit(source_activation)

    def transmit(self, source_activation: torch.Tensor) -> torch.Tensor:
        """Project (if needed) and scale the source activation by weight."""
        if source_activation.shape[-1] != self.source_output_dim:
            raise ValueError(
                f"Edge {self.id[:8]} expected source last-dim={self.source_output_dim}, "
                f"got shape {tuple(source_activation.shape)}"
            )
        signal: torch.Tensor
        if self.projection is not None:
            signal = self.projection(source_activation)
        else:
            signal = source_activation
        return signal * self.weight

    # ------------------------------------------------------------------
    # Stat updates (called by learning/pruning code)
    # ------------------------------------------------------------------
    def mark_active(self, step: int) -> None:
        """Record that this edge carried non-zero signal at ``step``."""
        self.last_active_step = step

    def increment_coactivation(self) -> None:
        self.coactivation_count += 1

    def update_strength(self, signal_magnitude: float, decay: float = 0.999) -> None:
        """EMA update of strength (utility tracking for pruning).

        ``strength = decay * strength + (1 - decay) * |weight * source_signal|``.
        The caller provides ``signal_magnitude`` (typically the source
        activation magnitude) so we avoid re-computing it per edge.
        """
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"strength decay must be in (0, 1], got {decay}")
        contribution = abs(float(self.weight.detach().item()) * float(signal_magnitude))
        self.strength = decay * self.strength + (1.0 - decay) * contribution

    def clamp_weight(self, max_abs: float) -> None:
        """In-place clamp of ``weight`` to ``[-max_abs, max_abs]``."""
        if max_abs <= 0.0:
            raise ValueError(f"max_abs must be positive, got {max_abs}")
        with torch.no_grad():
            self.weight.data.clamp_(-max_abs, max_abs)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize non-learnable scalar state."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_output_dim": self.source_output_dim,
            "target_input_dim": self.target_input_dim,
            "creation_step": self.creation_step,
            "last_active_step": self.last_active_step,
            "coactivation_count": self.coactivation_count,
            "strength": self.strength,
        }

    def load_scalar_state(self, state: dict[str, Any]) -> None:
        self.id = state["id"]
        self.source_id = state["source_id"]
        self.target_id = state["target_id"]
        self.creation_step = int(state["creation_step"])
        self.last_active_step = int(state["last_active_step"])
        self.coactivation_count = int(state["coactivation_count"])
        self.strength = float(state["strength"])

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        proj = "proj=yes" if self.projection is not None else "proj=no"
        return (
            f"id={self.id[:8]}..., {self.source_id[:8]}->{self.target_id[:8]}, "
            f"w={float(self.weight.detach().item()):.3f}, {proj}"
        )
