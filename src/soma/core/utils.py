"""Shared utility helpers for the core graph.

Keeps ID generation and projection-creation logic in one place so Node,
Edge, synaptogenesis, and neurogenesis can reuse them.
"""

from __future__ import annotations

import uuid

import torch
from torch import nn


def generate_uuid() -> str:
    """Return a new, globally-unique string ID.

    Used for ``Node.id`` and ``Edge.id``. We use ``uuid.uuid4().hex`` (32 hex
    chars) because:
    1. It's compact in serialization / logs.
    2. It's random, so there's no ordering bias.
    3. Collisions are astronomically unlikely at our graph sizes.
    """
    return uuid.uuid4().hex


def create_projection_if_needed(
    source_output_dim: int,
    target_input_dim: int,
    device: torch.device | str | None = None,
) -> nn.Linear | None:
    """Return ``nn.Linear(source_output_dim, target_input_dim)`` when needed.

    If ``source_output_dim == target_input_dim`` the projection is a no-op and
    we return ``None`` so callers can skip the extra matmul.

    The projection is used by ``Edge.transmit`` to bridge nodes with
    mismatched feature dimensions; it is learnable (trained by backprop).
    """
    if source_output_dim <= 0 or target_input_dim <= 0:
        raise ValueError(
            "Projection dims must be positive: "
            f"source={source_output_dim}, target={target_input_dim}"
        )
    if source_output_dim == target_input_dim:
        return None
    projection = nn.Linear(source_output_dim, target_input_dim, bias=False)
    # Small init so new projections don't disrupt established signal levels.
    nn.init.normal_(projection.weight, mean=0.0, std=0.1)
    if device is not None:
        projection = projection.to(device)
    return projection
