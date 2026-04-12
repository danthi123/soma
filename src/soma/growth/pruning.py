"""Pruning: remove weak/stale edges and orphaned non-boundary nodes.

Whitepaper Section 5.3. Invariants enforced (per CLAUDE.md):
- SENSOR and OUTPUT nodes are never pruned.
- Edges younger than ``config.pruning_grace_period`` are never pruned
  (gives new connections a chance to prove themselves).
"""

from __future__ import annotations

from dataclasses import dataclass

from soma.core.config import SOMAConfig
from soma.core.graph import Graph
from soma.core.node import NodeType


@dataclass(frozen=True)
class PruningResult:
    """Summary of one pruning pass."""

    removed_edge_ids: list[str]
    removed_node_ids: list[str]

    @property
    def removed_edges(self) -> int:
        return len(self.removed_edge_ids)

    @property
    def removed_nodes(self) -> int:
        return len(self.removed_node_ids)


def pruning(graph: Graph, step: int, config: SOMAConfig) -> PruningResult:
    """Run one pruning pass over the graph.

    An edge is pruned if it is both:
    - Older than ``config.pruning_grace_period`` (age = step - creation_step).
    - Low-utility: ``strength < edge_strength_threshold`` **and**
      ``time_since_active > inactivity_threshold``.

    A non-boundary node is pruned if it is fully disconnected (no
    incoming and no outgoing edges) after the edge sweep.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    grace = config.pruning_grace_period
    strength_threshold = config.edge_strength_threshold
    inactivity = config.inactivity_threshold

    removed_edge_ids: list[str] = []
    for edge in graph.all_edges():
        age = step - edge.creation_step
        if age < grace:
            continue
        time_since_active = step - edge.last_active_step
        if edge.strength < strength_threshold and time_since_active > inactivity:
            removed_edge_ids.append(edge.id)

    for edge_id in removed_edge_ids:
        graph.remove_edge(edge_id)

    removed_node_ids: list[str] = []
    for node in graph.all_nodes():
        if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
            continue  # CLAUDE.md invariant — never prune boundaries.
        if not graph.get_incoming_edges(node.id) and not graph.get_outgoing_edges(node.id):
            removed_node_ids.append(node.id)

    for node_id in removed_node_ids:
        graph.remove_node(node_id)

    return PruningResult(removed_edge_ids=removed_edge_ids, removed_node_ids=removed_node_ids)
