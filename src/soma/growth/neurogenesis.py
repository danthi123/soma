"""Neurogenesis: create new ASSOCIATOR nodes when predictions persistently fail.

Whitepaper Section 5.2.

A new node is warranted when recent error substantially exceeds the
baseline — the graph lacks capacity to represent this signal. The new
node is born at a position biased toward the currently most-active
nodes (so it connects to live circuitry) and is wired bidirectionally
to its k nearest neighbors.
"""

from __future__ import annotations

from typing import cast

import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.core.utils import generate_uuid


def neurogenesis(
    graph: Graph,
    recent_errors: list[float],
    step: int,
    config: SOMAConfig,
    *,
    num_neighbors: int = 5,
    recent_window: int = 100,
    baseline_window: int = 1000,
    rng: torch.Generator | None = None,
) -> Node | None:
    """Potentially create a new ASSOCIATOR node; return it or None.

    Trigger condition: mean of the last ``recent_window`` errors divided
    by the mean of the last ``baseline_window`` errors exceeds
    ``config.neurogenesis_threshold``. Requires the graph to have room
    (``num_nodes < config.max_nodes``) and at least one existing node so
    we have somewhere to wire the newcomer.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if num_neighbors <= 0:
        raise ValueError(f"num_neighbors must be positive, got {num_neighbors}")
    if recent_window <= 0 or baseline_window < recent_window:
        raise ValueError(
            f"require 0 < recent_window ({recent_window}) <= baseline_window ({baseline_window})"
        )

    if graph.num_nodes == 0:
        return None  # nothing to wire to
    if graph.num_nodes >= config.max_nodes:
        return None
    if not recent_errors:
        return None

    recent_slice = recent_errors[-recent_window:]
    baseline_slice = recent_errors[-baseline_window:]
    if not recent_slice or not baseline_slice:
        return None

    recent_mean = sum(recent_slice) / len(recent_slice)
    baseline_mean = sum(baseline_slice) / len(baseline_slice)

    if baseline_mean <= 0.0:
        return None
    if recent_mean / baseline_mean <= config.neurogenesis_threshold:
        return None

    # Place the new node near active circuitry.
    new_position = _compute_new_position(graph, config, rng=rng)
    new_node = Node.make(
        NodeType.ASSOCIATOR,
        config,
        creation_step=step,
        position=new_position,
        maturity=0.0,
    )
    graph.add_node(new_node)

    # Connect to the k nearest existing nodes (bidirectional).
    neighbors = graph.get_nearest_nodes(new_position, k=num_neighbors, exclude=[new_node.id])
    for neighbor in neighbors:
        _add_wired_edge(
            graph,
            source=neighbor,
            target=new_node,
            step=step,
            init_weight_scale=config.neurogenesis_init_weight_scale,
            rng=rng,
        )
        _add_wired_edge(
            graph,
            source=new_node,
            target=neighbor,
            step=step,
            init_weight_scale=config.neurogenesis_init_weight_scale,
            rng=rng,
        )

    return new_node


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _compute_new_position(
    graph: Graph,
    config: SOMAConfig,
    *,
    rng: torch.Generator | None,
) -> torch.Tensor:
    """Position the newcomer near the ``k`` most-active nodes with jitter."""
    active = graph.most_active_nodes(k=10)
    if active:
        positions = torch.stack(
            [cast(torch.Tensor, n.position).detach().float().flatten().cpu() for n in active]
        )
        centroid = positions.mean(dim=0)
        jitter = torch.randn(config.position_dim, generator=rng) * config.position_jitter
        return centroid + jitter
    # No active nodes — drop a newcomer anywhere in position space.
    return torch.randn(config.position_dim, generator=rng) * 0.1


def _add_wired_edge(
    graph: Graph,
    source: Node,
    target: Node,
    *,
    step: int,
    init_weight_scale: float = 0.01,
    rng: torch.Generator | None,
) -> None:
    """Best-effort edge creation between ``source`` and ``target``.

    ``init_weight_scale`` gates how loud the new connection starts:
    smaller values reduce the immediate disturbance on the existing
    circuit and rely on Hebbian updates to amplify edges where
    co-activation actually supports them.
    """
    if graph.has_edge(source.id, target.id):
        return
    init_weight = init_weight_scale * float(torch.randn(1, generator=rng).item())
    edge = Edge(
        source_id=source.id,
        target_id=target.id,
        source_output_dim=source.output_dim,
        target_input_dim=target.input_dim,
        creation_step=step,
        initial_weight=init_weight,
        edge_id=generate_uuid(),
    )
    graph.add_edge(edge)
