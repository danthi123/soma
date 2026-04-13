"""Synaptogenesis: create new edges between co-activated nodes.

Whitepaper Section 5.1. Called periodically (every
``config.synaptogenesis_interval`` steps). For each pair of nodes that
were both active this step, we compute a connection probability based on:

- Coactivation strength (product of activation magnitudes).
- Locality bonus (``exp(-distance / locality_scale)``) — nearby nodes
  in position space are more likely to wire together.
- Global ``synaptogenesis_rate`` scaling factor.

Edges are always directed; we consider both ``a -> b`` and ``b -> a`` as
independent candidates. An edge is only proposed if one doesn't already
exist in that direction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.utils import generate_uuid


def synaptogenesis(
    graph: Graph,
    activations: Mapping[str, torch.Tensor],
    step: int,
    config: SOMAConfig,
    *,
    rng: torch.Generator | None = None,
) -> list[Edge]:
    """Propose and add new edges between co-active nodes.

    Returns the list of newly-created ``Edge`` instances (may be empty).
    Mutation order within the call is deterministic given ``rng``.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    threshold = config.activation_threshold
    rate = config.synaptogenesis_rate
    locality = config.locality_scale

    active_ids: list[str] = []
    magnitudes: dict[str, float] = {}
    for nid, act in activations.items():
        # RMS magnitude (per-channel scale) rather than raw L2 norm, so
        # coactivation doesn't grow linearly with dim. A 64-dim tensor
        # at unit per-channel magnitude has norm=8 which would make
        # coact=64 and drive synaptogenesis prob to 1 for nearly every
        # pair regardless of rate. RMS keeps coact ~1 for unit signals.
        tensor = act.detach()
        numel = tensor.numel()
        mag = float(tensor.norm().item()) / math.sqrt(numel) if numel > 0 else 0.0
        if mag > threshold and nid in graph.nodes:
            active_ids.append(nid)
            magnitudes[nid] = mag

    if len(active_ids) < 2:
        return []

    # Pre-compute positions as CPU float32 tensors for distance math.
    positions: dict[str, torch.Tensor] = {
        nid: _to_cpu_float(cast(torch.Tensor, graph.nodes[nid].position)) for nid in active_ids
    }

    new_edges: list[Edge] = []
    # Iterate both directional pairs (a -> b and b -> a are sampled
    # independently since edges are directed).
    for source_id in active_ids:
        source_node = graph.nodes[source_id]
        for target_id in active_ids:
            if source_id == target_id:
                continue
            if graph.has_edge(source_id, target_id):
                continue
            target_node = graph.nodes[target_id]

            source_mag = magnitudes[source_id]
            target_mag = magnitudes[target_id]
            coact = source_mag * target_mag

            dist = float((positions[source_id] - positions[target_id]).norm().item())
            locality_bonus = float(torch.exp(torch.tensor(-dist / locality)).item())

            # Clamp probability to [0, 1]. Without this, when activations
            # are large (magnitude >> 1 during instability or spikes),
            # coact * rate easily exceeds 1 and every random draw
            # succeeds — creating edges between every co-active pair and
            # driving a positive-feedback loop: more edges -> bigger
            # activations -> more edges. Seen during Apr 13 run where
            # a single synaptogenesis event added 200+ edges mid-spike.
            prob = min(1.0, coact * locality_bonus * rate)
            if prob <= 0.0:
                continue

            draw = float(torch.rand(1, generator=rng).item())
            if draw >= prob:
                continue

            # Build and insert the new edge.
            init_weight = 0.01 * float(torch.randn(1, generator=rng).item())
            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                source_output_dim=source_node.output_dim,
                target_input_dim=target_node.input_dim,
                creation_step=step,
                initial_weight=init_weight,
                edge_id=generate_uuid(),
            )
            graph.add_edge(edge)
            edge.last_active_step = step
            new_edges.append(edge)

    return new_edges


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    """Detach -> CPU -> float32 -> flatten. Stable for distance math."""
    detached = tensor.detach()
    cpu = detached.cpu()
    flat = cpu.reshape(-1)
    return flat.to(torch.float32)
