"""Combined gradient + Hebbian update step.

Whitepaper Section 3.4. Responsibilities, in order:

1. Run ``loss.backward()`` to populate gradients on the active subgraph.
2. Apply per-node SGD updates with a learning rate scaled by youthfulness
   (young nodes learn faster than mature ones).
3. Reinforce edge weights where source and target co-activated (Hebbian:
   fire-together-wire-together), clamp weights, EMA the strength metric
   used by pruning.
4. Advance maturity of active nodes by ``config.maturity_increment``.
5. Homeostatic gain adjustment for every node (not just active ones) so
   silent nodes get nudged back toward their target activation level.

``update_step`` is called once per interaction step by the SOMA main loop
after ``execute_graph`` and loss computation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph


def _rms_magnitude(tensor: torch.Tensor) -> float:
    """Per-channel RMS magnitude. Unlike L2 norm, this is dim-invariant:
    a 64-dim unit-per-channel activation reports ~1.0 instead of 8.0.
    Used in the Hebbian coactivation product to prevent the magnitude
    scale from tracking dim."""
    numel = tensor.numel()
    if numel == 0:
        return 0.0
    return float(tensor.detach().norm().item()) / math.sqrt(numel)


def update_step(
    graph: Graph,
    loss: torch.Tensor,
    activations: Mapping[str, torch.Tensor],
    config: SOMAConfig,
    *,
    lr_multiplier: float = 1.0,
    zero_grad: bool = True,
) -> None:
    """Apply one combined backprop + Hebbian + homeostatic update.

    Parameters
    ----------
    graph:
        The graph whose params we update in-place.
    loss:
        Scalar loss tensor with ``requires_grad=True``. Must be
        differentiable w.r.t. the graph's parameters. ``loss.backward()``
        is called here; the caller should not call it beforehand.
    activations:
        ``node_id -> activation`` map produced by ``execute_graph``.
        Used to derive active nodes/edges and compute Hebbian signals.
    config:
        Supplies BASE_LR, YOUTH_LR_MULTIPLIER, HEBBIAN_LR,
        MAX_EDGE_WEIGHT, MATURITY_INCREMENT, ACTIVATION_THRESHOLD,
        and gain bounds.
    lr_multiplier:
        External global LR scale (e.g., from ``HomeostaticRegulator``).
        Defaults to 1.0.
    zero_grad:
        When True (default), gradients are zeroed after they've been
        applied so the next ``backward()`` starts fresh. Set False if
        the caller wants to accumulate gradients across multiple steps.
    """
    # 1. Backprop through the active subgraph.
    loss.backward()  # type: ignore[no-untyped-call]

    # 1b. Global gradient clipping. Bounds the total L2 norm of all
    # node + edge gradients so a single explosive batch can't drive
    # weights to the clamp in one step. Params with grad=None are
    # filtered automatically by clip_grad_norm_.
    all_params = [param for node in graph.all_nodes() for param in node.parameters()]
    all_params.extend(param for edge in graph.all_edges() for param in edge.parameters())
    if all_params:
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=config.grad_clip_max_norm)

    # 2. Per-node SGD with maturity-scaled LR.
    active_node_ids = list(activations.keys())
    _apply_node_gradients(
        graph,
        active_node_ids,
        config=config,
        lr_multiplier=lr_multiplier,
        zero_grad=zero_grad,
    )

    # 3. Hebbian edge update + clamp + strength EMA.
    _apply_hebbian_edge_updates(graph, activations, config=config)

    # 4. Advance maturity of active nodes.
    for nid in active_node_ids:
        if nid in graph.nodes:
            graph.nodes[nid].advance_maturity(config.maturity_increment)

    # 5. Homeostatic gain for ALL nodes (whitepaper 3.4 step 4).
    _apply_homeostatic_gain(graph, config=config)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------
def _apply_node_gradients(
    graph: Graph,
    active_node_ids: list[str],
    *,
    config: SOMAConfig,
    lr_multiplier: float,
    zero_grad: bool,
) -> None:
    """SGD step on every active node's parameters."""
    with torch.no_grad():
        for nid in active_node_ids:
            if nid not in graph.nodes:
                # Node may have been pruned between execution and learning
                # (defensive — shouldn't happen under current ordering).
                continue
            node = graph.nodes[nid]
            node_lr = (
                config.base_lr
                * (1.0 + (1.0 - node.maturity) * config.youth_lr_multiplier)
                * lr_multiplier
            )
            for param in node.parameters():
                if param.grad is None:
                    continue
                param.data.add_(param.grad, alpha=-node_lr)
                if zero_grad:
                    param.grad.zero_()

        # Also update edge parameters (weights + projections) with the
        # current global LR. Hebbian will further nudge edge weights
        # afterwards.
        edge_lr = config.base_lr * lr_multiplier
        for edge in graph.all_edges():
            for param in edge.parameters():
                if param.grad is None:
                    continue
                param.data.add_(param.grad, alpha=-edge_lr)
                if zero_grad:
                    param.grad.zero_()


def _apply_hebbian_edge_updates(
    graph: Graph,
    activations: Mapping[str, torch.Tensor],
    *,
    config: SOMAConfig,
) -> None:
    """Fire-together-wire-together edge reinforcement, clamp, and strength EMA.

    Also applies passive multiplicative weight decay to every edge each
    step (active or not). The decay counterbalances the monotonically
    positive Hebbian add term so edge weights don't drift to the clamp
    boundary over time. With ``edge_weight_decay`` slightly under 1 and
    typical activations, the equilibrium is well below ``max_edge_weight``.
    """
    threshold = config.activation_threshold
    max_w = config.max_edge_weight
    hebbian_lr = config.hebbian_lr
    decay = config.edge_weight_decay

    with torch.no_grad():
        if decay < 1.0:
            for edge in graph.all_edges():
                edge.weight.data.mul_(decay)

        for edge in graph.all_edges():
            source_act = activations.get(edge.source_id)
            target_act = activations.get(edge.target_id)
            if source_act is None or target_act is None:
                continue  # edge did not participate this step

            source_mag = _rms_magnitude(source_act)
            target_mag = _rms_magnitude(target_act)

            if source_mag > threshold and target_mag > threshold:
                edge.increment_coactivation()
                edge.weight.data.add_(hebbian_lr * source_mag * target_mag)

            # Clamp after Hebbian bump so weights never exceed ±max_w.
            _clamp_edge(edge, max_w)

            # Strength EMA (pruning utility).
            edge.update_strength(signal_magnitude=source_mag, decay=0.999)


def _clamp_edge(edge: Edge, max_abs: float) -> None:
    edge.weight.data.clamp_(-max_abs, max_abs)


def _apply_homeostatic_gain(graph: Graph, *, config: SOMAConfig) -> None:
    """Bump gain up/down based on activation EMA vs target. Clamp to [gain_min, gain_max]."""
    hi = 1.2
    lo = 0.8
    up_factor = 1.001
    down_factor = 0.999
    for node in graph.all_nodes():
        if node.activation_ema > node.target_activation * hi:
            node.gain *= down_factor
        elif node.activation_ema < node.target_activation * lo:
            node.gain *= up_factor
        # Clamp after adjustment (uses the node's own gain bounds).
        node.clamp_gain()
