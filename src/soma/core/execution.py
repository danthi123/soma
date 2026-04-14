"""Graph execution: topological ordering, cycle handling, forward pass.

Whitepaper Section 3.3 / 3.4. Key points:
- Execution is wave-by-wave in topological order.
- Cycles use previous-timestep activations for back-edges ("implicit
  recurrence"). ``topological_sort`` detects back-edges via iterative
  DFS (WHITE/GRAY/BLACK colouring).
- Dormant nodes (no active incoming signal) are skipped and produce no
  activation for this step.
- Edges that transmit a signal have their ``last_active_step`` bumped.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch

from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import NodeType


def topological_sort(graph: Graph) -> tuple[list[str], set[str]]:
    """Return a topological ordering of node IDs plus a set of back-edge IDs.

    Uses iterative DFS with three-colour marking so we can handle graphs
    with tens of thousands of nodes without hitting the Python recursion
    limit. Back-edges (source/target both GRAY when the edge is examined)
    are removed from the DAG view and reported separately so the
    executor can substitute previous-timestep activations.

    For a DAG the returned order satisfies: every non-back-edge
    ``u -> v`` places ``u`` before ``v``.
    """
    white, gray, black = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(graph.nodes, white)
    order: list[str] = []
    back_edges: set[str] = set()

    # Iterative DFS: the stack holds (node_id, iterator-over-outgoing-edges).
    # We visit a node when pushing it; we emit it when popping (reverse
    # post-order gives topological order).
    for start_id in list(graph.nodes):
        if color[start_id] != white:
            continue
        color[start_id] = gray
        stack: list[tuple[str, Iterator[Edge]]] = [
            (start_id, iter(graph.get_outgoing_edges(start_id)))
        ]
        while stack:
            current_id, edge_iter = stack[-1]
            advanced = False
            for edge in edge_iter:
                tgt = edge.target_id
                tgt_color = color[tgt]
                if tgt_color == white:
                    color[tgt] = gray
                    stack.append((tgt, iter(graph.get_outgoing_edges(tgt))))
                    advanced = True
                    break
                if tgt_color == gray:
                    # Back-edge: target is an ancestor still on the stack.
                    back_edges.add(edge.id)
                # BLACK targets are already fully processed; nothing to do.
            if not advanced:
                color[current_id] = black
                order.append(current_id)
                stack.pop()

    order.reverse()
    return order, back_edges


def execute_graph(
    graph: Graph,
    inputs: Mapping[str, torch.Tensor],
    current_step: int,
    *,
    previous_activations: Mapping[str, torch.Tensor] | None = None,
    record_edge_activity: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Execute a single forward pass through the graph.

    Parameters
    ----------
    graph:
        The graph to execute. Mutating the graph during execution (e.g.,
        via growth) is not supported — call structural updates between
        steps.
    inputs:
        Modality -> input tensor. Every key must name a registered
        SENSOR; extra keys raise ``KeyError``. Sensors not present in
        ``inputs`` are executed with whatever was previously set via
        ``Node.set_input`` (or zeros if never set).
    current_step:
        Global step number. Used for ``last_active_step`` bookkeeping.
    previous_activations:
        Optional map from node_id to the activation produced at the
        previous timestep. Supplies the signal carried by back-edges;
        any node_id not present means the back-edge contributes nothing
        this step.
    record_edge_activity:
        When True (default), edges that transmit signal this step have
        their ``last_active_step`` updated. Set False for pure
        evaluation paths (e.g., consolidation replay) that shouldn't
        influence pruning bookkeeping.

    Returns
    -------
    outputs:
        Modality -> activation for each registered OUTPUT that received
        signal this step (dormant outputs are omitted).
    activations:
        node_id -> activation for every node that produced one this
        step. Pass this back in as ``previous_activations`` on the next
        call to enable back-edge propagation.
    """
    _inject_sensor_inputs(graph, inputs)

    order, back_edges = topological_sort(graph)
    prev: Mapping[str, torch.Tensor] = previous_activations or {}
    activations: dict[str, torch.Tensor] = {}

    for node_id in order:
        node = graph.nodes[node_id]

        if node.node_type is NodeType.SENSOR:
            # Sensor forward reads its set_input buffer and returns it.
            activations[node_id] = node.forward({}, current_step)
            continue

        incoming_signals: dict[str, torch.Tensor] = {}
        for edge in graph.get_incoming_edges(node_id):
            source_act = _source_activation_for(edge, back_edges, activations, prev)
            if source_act is None:
                continue
            signal = edge.transmit(source_act)
            incoming_signals[edge.source_id] = signal
            if record_edge_activity:
                edge.mark_active(current_step)

        if not incoming_signals:
            # Node has no active input this step — stays dormant, no
            # activation recorded (whitepaper 3.3).
            continue

        activations[node_id] = node.forward(incoming_signals, current_step)

    outputs = _collect_outputs(graph, activations)
    return outputs, activations


def execute_graph_batched(
    graph: Graph,
    inputs: Mapping[str, torch.Tensor],
    current_step: int,
    *,
    previous_activations: Mapping[str, torch.Tensor] | None = None,
    record_edge_activity: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Batched variant of :func:`execute_graph`.

    Currently delegates to the sequential path. Subsequent tasks replace
    the body with wave-grouped batched matmuls.
    """
    return execute_graph(
        graph,
        inputs,
        current_step,
        previous_activations=previous_activations,
        record_edge_activity=record_edge_activity,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _inject_sensor_inputs(graph: Graph, inputs: Mapping[str, torch.Tensor]) -> None:
    """Push external tensors onto the corresponding SENSOR nodes.

    Raises ``KeyError`` if any modality is unknown. Shape mismatches
    surface from ``Node.set_input``.
    """
    for modality, data in inputs.items():
        sensor = graph.get_sensor(modality)  # raises KeyError if missing
        sensor.set_input(data)


def _source_activation_for(
    edge: Edge,
    back_edges: set[str],
    activations: Mapping[str, torch.Tensor],
    previous_activations: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    """Pick the right activation for an edge: current or previous step.

    Back-edges use the previous timestep's activation (whitepaper 3.3
    "implicit recurrence"). Forward edges use the current step.
    """
    if edge.id in back_edges:
        return previous_activations.get(edge.source_id)
    return activations.get(edge.source_id)


def _collect_outputs(
    graph: Graph,
    activations: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build modality -> output tensor from activations, skipping dormant outputs."""
    outputs: dict[str, torch.Tensor] = {}
    for modality, out_node in graph.output_nodes.items():
        act = activations.get(out_node.id)
        if act is not None:
            outputs[modality] = act
    return outputs
