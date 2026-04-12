"""Myelination: detect linear chains of mature edges and compress them.

Whitepaper Section 5.4.

A linear chain is a maximal path of ASSOCIATOR/INTEGRATOR nodes where
every internal node has in-degree 1 and out-degree 1, and every edge in
the chain is both old (``step - creation_step > myelination_age_threshold``)
and strong (``strength > myelination_strength_threshold``).

When a chain qualifies, it is replaced by a single new node of the same
type as its first node. Incoming edges to the chain's head are redirected
to the new node; outgoing edges from the chain's tail are also redirected.
The new node's parameters are freshly initialized — online learning plus
consolidation replay will re-fit the composed function.
"""

from __future__ import annotations

from dataclasses import dataclass

from soma.core.config import SOMAConfig
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


@dataclass(frozen=True)
class MyelinationResult:
    """Summary of one myelination pass."""

    chains_detected: int
    chains_compressed: int
    nodes_removed: int
    new_node_ids: list[str]


def myelination(
    graph: Graph,
    step: int,
    config: SOMAConfig,
    *,
    min_length: int = 2,
    max_length: int = 4,
) -> MyelinationResult:
    """Run one myelination pass and return a summary of changes."""
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if not 2 <= min_length <= max_length:
        raise ValueError(f"require 2 <= min_length ({min_length}) <= max_length ({max_length})")

    chains = detect_linear_chains(graph, min_length=min_length, max_length=max_length)
    compressed = 0
    removed = 0
    new_ids: list[str] = []
    for chain in chains:
        if not _chain_is_ripe(graph, chain, step, config):
            continue
        new_node = _compress_chain(graph, chain, step, config)
        compressed += 1
        removed += len(chain)
        new_ids.append(new_node.id)

    return MyelinationResult(
        chains_detected=len(chains),
        chains_compressed=compressed,
        nodes_removed=removed,
        new_node_ids=new_ids,
    )


def detect_linear_chains(
    graph: Graph,
    *,
    min_length: int = 2,
    max_length: int = 4,
) -> list[list[str]]:
    """Return non-overlapping lists of node IDs that form linear chains.

    A node qualifies for chain membership when it is ASSOCIATOR or
    INTEGRATOR, has exactly one incoming edge, and exactly one outgoing
    edge. A chain is a maximal sequence of such nodes where consecutive
    members are linked by their single in/out edges.
    """
    if min_length < 2 or max_length < min_length:
        raise ValueError(f"require 2 <= min_length ({min_length}) <= max_length ({max_length})")

    chainable_ids = {
        nid
        for nid, node in graph.nodes.items()
        if node.node_type in (NodeType.ASSOCIATOR, NodeType.INTEGRATOR)
        and len(graph.get_incoming_edges(nid)) == 1
        and len(graph.get_outgoing_edges(nid)) == 1
    }

    # Map each chainable node to its single downstream target id.
    successor: dict[str, str] = {}
    for nid in chainable_ids:
        out = graph.get_outgoing_edges(nid)
        successor[nid] = out[0].target_id

    # Identify chain starts: chainable nodes whose predecessor is NOT chainable.
    starts: list[str] = []
    for nid in chainable_ids:
        incoming = graph.get_incoming_edges(nid)
        pred = incoming[0].source_id
        if pred not in chainable_ids:
            starts.append(nid)

    # Walk each start forward, consuming chainable successors.
    chains: list[list[str]] = []
    visited: set[str] = set()
    for start in starts:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        cur = start
        while len(chain) < max_length:
            nxt = successor.get(cur)
            if nxt is None or nxt not in chainable_ids or nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        if len(chain) >= min_length:
            chains.append(chain)
    return chains


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _chain_is_ripe(
    graph: Graph,
    chain: list[str],
    step: int,
    config: SOMAConfig,
) -> bool:
    """True iff every internal edge in the chain is old and strong."""
    for a, b in zip(chain, chain[1:], strict=False):
        edge = graph.get_edge(a, b)
        if edge.strength <= config.myelination_strength_threshold:
            return False
        if (step - edge.creation_step) <= config.myelination_age_threshold:
            return False
    return True


def _compress_chain(
    graph: Graph,
    chain: list[str],
    step: int,
    config: SOMAConfig,
) -> Node:
    """Replace the chain with a single new node; reuse the first node's type."""
    head = graph.nodes[chain[0]]
    tail = graph.nodes[chain[-1]]

    merged = Node.make(
        head.node_type,
        config,
        creation_step=step,
        input_dim=head.input_dim,
        output_dim=tail.output_dim,
        maturity=min(head.maturity, tail.maturity),
    )
    graph.add_node(merged)

    # Redirect incoming edges (from outside the chain) to the merged node.
    incoming = [e for e in graph.get_incoming_edges(chain[0]) if e.source_id not in chain]
    outgoing = [e for e in graph.get_outgoing_edges(chain[-1]) if e.target_id not in chain]

    for old_edge in incoming:
        source_id = old_edge.source_id
        source_dim = old_edge.source_output_dim
        graph.remove_edge(old_edge.id)
        _add_redirected_edge(
            graph,
            source_id=source_id,
            target_id=merged.id,
            source_output_dim=source_dim,
            target_input_dim=merged.input_dim,
            step=step,
        )
    for old_edge in outgoing:
        target_id = old_edge.target_id
        target_dim = old_edge.target_input_dim
        graph.remove_edge(old_edge.id)
        _add_redirected_edge(
            graph,
            source_id=merged.id,
            target_id=target_id,
            source_output_dim=merged.output_dim,
            target_input_dim=target_dim,
            step=step,
        )

    # Remove chain nodes (cascade removes remaining internal edges).
    for nid in chain:
        if nid in graph.nodes:
            graph.remove_node(nid)
    return merged


def _add_redirected_edge(
    graph: Graph,
    *,
    source_id: str,
    target_id: str,
    source_output_dim: int,
    target_input_dim: int,
    step: int,
) -> None:
    """Insert an edge from ``source_id`` to ``target_id`` if one doesn't already exist."""
    if graph.has_edge(source_id, target_id):
        return
    from soma.core.edge import Edge

    edge = Edge(
        source_id=source_id,
        target_id=target_id,
        source_output_dim=source_output_dim,
        target_input_dim=target_input_dim,
        creation_step=step,
        initial_weight=0.1,
    )
    graph.add_edge(edge)
