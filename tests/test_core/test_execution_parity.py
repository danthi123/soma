"""Strict parity between sequential and batched execute_graph.

If this test passes, the wave-batching optimization preserves all
observable behavior: outputs, activations, gradients on every param,
and per-node bookkeeping (gain/ema/last_active_step/history).

Uses ``copy.deepcopy(g1)`` to get ``g2`` — node IDs are UUIDs baked
into state_dict keys, so two freshly-built graphs can't share state
via ``load_state_dict``. Deepcopy preserves IDs, weights, and buffers.
"""

from __future__ import annotations

import copy

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph, execute_graph_batched
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


def _build_realistic_graph(config: SOMAConfig, seed: int = 2026) -> Graph:
    """10-ish-node mixed-type graph that exercises:

    - Multiple shape buckets in the same wave (assocs + integrators).
    - Forward edges that require a projection (dim mismatch between
      integrator output and output-node input).
    - Fan-in (multiple edges into the same output).
    """
    torch.manual_seed(seed)
    g = Graph()
    dim = config.sensor_output_dim
    # 1 sensor, 5 associators, 2 integrators, 1 projector assoc, 2 outputs
    s = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    g.add_node(s, modality="text")
    assocs = [Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config) for _ in range(5)]
    for a in assocs:
        g.add_node(a)
    # Different-shape integrators force a second bucket in their wave.
    integs = [Node(NodeType.INTEGRATOR, dim, dim * 4, dim * 2, 0, config) for _ in range(2)]
    for i in integs:
        g.add_node(i)
    # Projector assoc consumes integrator output (dim*2 -> dim) and then
    # feeds outputs. No projection needed on integrator -> proj because
    # dims match; projection DOES get exercised on integ -> output direct
    # edges added below.
    proj = Node(NodeType.ASSOCIATOR, dim * 2, dim * 2, dim, 0, config)
    g.add_node(proj)
    o_text = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    o_other = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    g.add_node(o_text, modality="text")
    g.add_node(o_other, modality="image")  # second output modality

    def _e(src: Node, tgt: Node, w: float = 0.3) -> None:
        g.add_edge(
            Edge(
                source_id=src.id,
                target_id=tgt.id,
                source_output_dim=src.output_dim,
                target_input_dim=tgt.input_dim,
                creation_step=0,
                initial_weight=w,
            )
        )

    # Fan-out from sensor.
    for a in assocs:
        _e(s, a)
    for i in integs:
        _e(s, i)
    # Integrators -> proj (same dim, no projection).
    for i in integs:
        _e(i, proj)
    # Also integ -> o_text direct (dim*2 -> dim: forces a projection).
    _e(integs[0], o_text)
    # Assocs + proj -> outputs.
    for a in assocs:
        _e(a, o_text)
        _e(a, o_other)
    _e(proj, o_text)
    _e(proj, o_other)
    return g


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig()


def test_full_parity_forward_outputs(config: SOMAConfig) -> None:
    g1 = _build_realistic_graph(config)
    g2 = copy.deepcopy(g1)
    data = torch.randn(config.sensor_output_dim)
    out_seq, act_seq = execute_graph(g1, inputs={"text": data}, current_step=5)
    out_bat, act_bat = execute_graph_batched(g2, inputs={"text": data}, current_step=5)
    assert set(out_seq) == set(out_bat)
    for k in out_seq:
        diff = (out_seq[k] - out_bat[k]).abs().max().item()
        assert torch.allclose(out_seq[k], out_bat[k], atol=1e-5, rtol=1e-5), (
            f"output[{k}] max diff = {diff}"
        )
    # Active-node ID sets must match.
    assert set(act_seq) == set(act_bat)
    for nid in act_seq:
        assert torch.allclose(act_seq[nid], act_bat[nid], atol=1e-5, rtol=1e-5), (
            f"activation[{nid[:8]}] max diff = "
            f"{(act_seq[nid] - act_bat[nid]).abs().max().item()}"
        )


def test_full_parity_gradients(config: SOMAConfig) -> None:
    g1 = _build_realistic_graph(config)
    g2 = copy.deepcopy(g1)
    data = torch.randn(config.sensor_output_dim)
    out_seq, _ = execute_graph(g1, inputs={"text": data}, current_step=5)
    out_bat, _ = execute_graph_batched(g2, inputs={"text": data}, current_step=5)
    # Same scalar loss in both.
    loss_seq = torch.stack(list(out_seq.values())).sum()
    loss_bat = torch.stack(list(out_bat.values())).sum()
    loss_seq.backward()
    loss_bat.backward()

    # Compare every gradient on every node and every edge.
    for nid in g1.nodes:
        n1, n2 = g1.nodes[nid], g2.nodes[nid]
        for pname in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
            p1 = dict(n1.named_parameters())[pname]
            p2 = dict(n2.named_parameters())[pname]
            if p1.grad is None and p2.grad is None:
                continue
            assert p1.grad is not None and p2.grad is not None, (
                f"grad-presence mismatch {nid[:8]}.{pname}"
            )
            diff = (p1.grad - p2.grad).abs().max().item()
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-5), (
                f"grad diverges on {nid[:8]}.{pname}: max diff {diff}"
            )
    for eid in g1.edges:
        e1, e2 = g1.edges[eid], g2.edges[eid]
        if e1.weight.grad is None and e2.weight.grad is None:
            continue
        assert e1.weight.grad is not None and e2.weight.grad is not None
        assert torch.allclose(e1.weight.grad, e2.weight.grad, atol=1e-5, rtol=1e-5)
        if e1.projection is not None:
            assert e2.projection is not None
            # Projection weight is always present; bias is optional
            # (create_projection_if_needed uses bias=False by default).
            if e1.projection.weight.grad is not None:
                assert e2.projection.weight.grad is not None
                assert torch.allclose(
                    e1.projection.weight.grad,
                    e2.projection.weight.grad,
                    atol=1e-5,
                    rtol=1e-5,
                )
            if (
                e1.projection.bias is not None
                and e1.projection.bias.grad is not None
            ):
                assert e2.projection.bias is not None
                assert e2.projection.bias.grad is not None
                assert torch.allclose(
                    e1.projection.bias.grad,
                    e2.projection.bias.grad,
                    atol=1e-5,
                    rtol=1e-5,
                )


def test_full_parity_per_node_state(config: SOMAConfig) -> None:
    g1 = _build_realistic_graph(config)
    g2 = copy.deepcopy(g1)
    data = torch.randn(config.sensor_output_dim)
    execute_graph(g1, inputs={"text": data}, current_step=42)
    execute_graph_batched(g2, inputs={"text": data}, current_step=42)
    for nid in g1.nodes:
        n1, n2 = g1.nodes[nid], g2.nodes[nid]
        assert n1.last_active_step == n2.last_active_step, (
            f"{nid[:8]}: last_active_step {n1.last_active_step} != {n2.last_active_step}"
        )
        assert n1.activation_ema == pytest.approx(
            n2.activation_ema, rel=1e-6, abs=1e-9
        ), f"{nid[:8]}: ema {n1.activation_ema} != {n2.activation_ema}"
        assert n1.activation_history.to_list() == pytest.approx(
            n2.activation_history.to_list(), rel=1e-6, abs=1e-9
        )
    # Edge last_active_step must match too.
    for eid in g1.edges:
        assert g1.edges[eid].last_active_step == g2.edges[eid].last_active_step


def test_full_parity_over_multiple_steps(config: SOMAConfig) -> None:
    """Chain three steps with previous_activations. Any drift in state
    should accumulate visibly over 3+ steps."""
    g1 = _build_realistic_graph(config)
    g2 = copy.deepcopy(g1)

    prev_seq: dict[str, torch.Tensor] = {}
    prev_bat: dict[str, torch.Tensor] = {}
    for step in range(3):
        data = torch.randn(config.sensor_output_dim)
        out_seq, prev_seq = execute_graph(
            g1,
            inputs={"text": data},
            current_step=step,
            previous_activations=prev_seq,
        )
        out_bat, prev_bat = execute_graph_batched(
            g2,
            inputs={"text": data},
            current_step=step,
            previous_activations=prev_bat,
        )
        for k in out_seq:
            assert torch.allclose(out_seq[k], out_bat[k], atol=1e-5, rtol=1e-5), (
                f"step {step}: output[{k}] divergence"
            )
