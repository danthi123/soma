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
from torch.nn import functional as F  # noqa: N812

from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


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


def compute_wave_layers(graph: Graph) -> list[list[str]]:
    """Group node IDs into topological waves.

    A node's wave index is ``1 + max(wave of each predecessor via forward
    edges)``, or ``0`` if it has no forward-edge predecessors. Back-edges
    (detected the same way as in :func:`topological_sort`) are ignored
    for wave assignment because they carry previous-step signal, not
    this-step signal.

    Returns a list of waves, each a list of node IDs. The order of node
    IDs *within* a wave matches the order they appear in
    :func:`topological_sort` — the batched executor updates per-node
    state in this same order to keep sequential-vs-batched state-update
    ordering identical.
    """
    order, back_edges = topological_sort(graph)
    wave_of: dict[str, int] = {}
    for node_id in order:
        max_pred = -1
        for edge in graph.get_incoming_edges(node_id):
            if edge.id in back_edges:
                continue
            if edge.source_id in wave_of and wave_of[edge.source_id] > max_pred:
                max_pred = wave_of[edge.source_id]
        wave_of[node_id] = max_pred + 1

    if not wave_of:
        return []
    num_waves = max(wave_of.values()) + 1
    waves: list[list[str]] = [[] for _ in range(num_waves)]
    # Preserve topological order within each wave.
    for node_id in order:
        waves[wave_of[node_id]].append(node_id)
    return waves


def bucket_wave_by_shape(
    graph: Graph,
    wave: list[str],
) -> dict[tuple[int, int, int], list[str]]:
    """Bucket a wave's node IDs by their MLP shape.

    Key is ``(input_dim, hidden_dim, output_dim)``. SENSOR nodes are
    skipped because they don't run an MLP (their ``forward`` returns
    the injected input). Order within each bucket follows the order of
    ``wave`` — callers depend on this for deterministic state updates.
    """
    buckets: dict[tuple[int, int, int], list[str]] = {}
    for node_id in wave:
        node = graph.nodes[node_id]
        if node.node_type is NodeType.SENSOR:
            continue
        key = (node.input_dim, node.hidden_dim, node.output_dim)
        buckets.setdefault(key, []).append(node_id)
    return buckets


def execute_graph_batched(
    graph: Graph,
    inputs: Mapping[str, torch.Tensor],
    current_step: int,
    *,
    previous_activations: Mapping[str, torch.Tensor] | None = None,
    record_edge_activity: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Wave-batched variant of :func:`execute_graph`.

    Produces outputs and activations equal-up-to-float-reassociation
    (atol=1e-5) to the sequential executor. Batches the per-node MLP
    kernels by shape within each topological wave to collapse ~N kernel
    launches per wave into ~4 (W1 matmul + bias, GELU, W2 matmul + bias).

    Parameter documentation matches :func:`execute_graph` verbatim.
    """
    _inject_sensor_inputs(graph, inputs)

    _, back_edges = topological_sort(graph)
    waves = compute_wave_layers(graph)
    prev: Mapping[str, torch.Tensor] = previous_activations or {}
    activations: dict[str, torch.Tensor] = {}

    for wave in waves:
        # 1. Sensor nodes in this wave: forward each one sequentially
        # (their forward is just "return injected input"; no MLP).
        for node_id in wave:
            node = graph.nodes[node_id]
            if node.node_type is NodeType.SENSOR:
                activations[node_id] = node.forward({}, current_step)

        # 2. Non-sensor nodes: bucket by shape, batch per bucket.
        buckets = bucket_wave_by_shape(graph, wave)
        for bucket_ids in buckets.values():
            bucket_nodes = [graph.nodes[nid] for nid in bucket_ids]
            x, active_nodes = _aggregate_bucket_inputs(
                graph=graph,
                nodes=bucket_nodes,
                activations=activations,
                previous_activations=prev,
                back_edges=back_edges,
                current_step=current_step,
                record_edge_activity=record_edge_activity,
            )
            if not active_nodes:
                continue
            y = _batched_node_forward(active_nodes, x)
            # Unbind so each node's activation is its own tensor —
            # downstream loss.backward() can then route gradients freely.
            y_rows = torch.unbind(y, dim=0)
            for node, row in zip(active_nodes, y_rows, strict=True):
                activations[node.id] = row
            _record_batched_activations(active_nodes, y, current_step)

    outputs = _collect_outputs(graph, activations)
    return outputs, activations


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


def _batched_node_forward(nodes: list[Node], x: torch.Tensor) -> torch.Tensor:
    """Run the MLP step for a list of same-shape nodes in one batched pass.

    Parameters
    ----------
    nodes:
        Non-empty list. All must share ``(input_dim, hidden_dim, output_dim)``.
    x:
        Pre-aggregated input tensor of shape ``(len(nodes), input_dim)``.

    Returns
    -------
    Tensor of shape ``(len(nodes), output_dim)`` equal (up to float
    re-association) to ``torch.stack([n.forward({'_': x[i]}, ...)])``.

    Notes
    -----
    Gradients flow back to each node's individual ``linear1.weight``,
    ``linear1.bias``, ``linear2.weight``, ``linear2.bias`` because we
    build the batched weight tensor with ``torch.stack``, whose backward
    splits the accumulated gradient back to the input tensors. ``gain``
    is a Python float (not a ``nn.Parameter``), so it does not receive
    gradients — matches the sequential path.
    """
    assert nodes, "_batched_node_forward called with empty node list"
    input_dim = nodes[0].input_dim
    hidden_dim = nodes[0].hidden_dim
    output_dim = nodes[0].output_dim
    # Defensive: catch shape mismatches early.
    for n in nodes:
        assert (n.input_dim, n.hidden_dim, n.output_dim) == (
            input_dim,
            hidden_dim,
            output_dim,
        ), (
            f"Shape mismatch in bucket: {n.id[:8]} has "
            f"{(n.input_dim, n.hidden_dim, n.output_dim)}, expected "
            f"{(input_dim, hidden_dim, output_dim)}"
        )

    # Stack weight/bias into (N, hidden, input) / (N, hidden) tensors.
    # torch.stack preserves autograd edges, so grads flow back to each
    # node's own parameters.
    W1 = torch.stack([n.linear1.weight for n in nodes])  # (N, hidden, input)
    b1 = torch.stack([n.linear1.bias for n in nodes])  # (N, hidden)
    W2 = torch.stack([n.linear2.weight for n in nodes])  # (N, output, hidden)
    b2 = torch.stack([n.linear2.bias for n in nodes])  # (N, output)

    # Batched linear: h[n] = W1[n] @ x[n] + b1[n]. einsum keeps it
    # explicit; torch.bmm would also work.
    h = torch.einsum("nhi,ni->nh", W1, x) + b1
    h = F.gelu(h)
    h = torch.einsum("noh,nh->no", W2, h) + b2

    # Per-node gain: broadcast (N,) -> (N, 1).
    device = x.device
    dtype = x.dtype
    gain = torch.tensor([n.gain for n in nodes], device=device, dtype=dtype).unsqueeze(-1)
    h = h * gain

    if input_dim == output_dim:
        h = h + x

    return h


def _record_batched_activations(
    nodes: list[Node],
    outputs: torch.Tensor,
    current_step: int,
) -> None:
    """Per-node state update for a batched-forward output.

    Must match :meth:`Node._record_activation` exactly: append magnitude
    to ring buffer, EMA-update ``activation_ema`` with 0.99/0.01 blend,
    and bump ``last_active_step`` iff magnitude > threshold.

    Processes nodes in list order so sequential-vs-batched ordering of
    host-side state updates is identical.
    """
    # Compute per-row L2 norms in one shot, then pull to host for the
    # Python-float state that Node holds.
    mags = outputs.detach().norm(dim=-1).tolist()
    for node, mag in zip(nodes, mags, strict=True):
        node.activation_history.append(mag)
        node.activation_ema = 0.99 * node.activation_ema + 0.01 * mag
        if mag > node._activation_threshold:
            node.last_active_step = current_step


def _aggregate_bucket_inputs(
    graph: Graph,
    nodes: list[Node],
    activations: Mapping[str, torch.Tensor],
    previous_activations: Mapping[str, torch.Tensor],
    back_edges: set[str],
    current_step: int,
    record_edge_activity: bool,
) -> tuple[torch.Tensor, list[Node]]:
    """Aggregate incoming edges for each node, return stacked inputs.

    Nodes with zero active incoming edges are dropped (dormant). Returned
    ``active_nodes`` preserves the input order minus dropped nodes, and
    row ``i`` of the returned tensor is the aggregated input for
    ``active_nodes[i]``.

    Iterates edges in the same order as the sequential executor so
    edge ``mark_active`` bookkeeping is identical.
    """
    rows: list[torch.Tensor] = []
    active_nodes: list[Node] = []
    for node in nodes:
        agg: torch.Tensor | None = None
        for edge in graph.get_incoming_edges(node.id):
            source_act = _source_activation_for(edge, back_edges, activations, previous_activations)
            if source_act is None:
                continue
            signal = edge.transmit(source_act)
            agg = signal if agg is None else agg + signal
            if record_edge_activity:
                edge.mark_active(current_step)
        if agg is None:
            continue  # dormant
        rows.append(agg)
        active_nodes.append(node)
    if not rows:
        # Empty bucket — return a zero-row tensor so downstream shape
        # checks don't crash. Caller should test ``active_nodes``.
        return torch.empty(0), active_nodes
    return torch.stack(rows), active_nodes
