"""Tests for ``soma.core.learning.update_step``."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph
from soma.core.graph import Graph
from soma.core.learning import update_step
from soma.core.node import Node, NodeType


@pytest.fixture
def config() -> SOMAConfig:
    # Use larger learning rates so per-step changes are measurable.
    return SOMAConfig(
        base_lr=0.1,
        hebbian_lr=0.01,
        maturity_increment=0.01,
        activation_threshold=0.01,
    )


def _three_node_graph(config: SOMAConfig) -> tuple[Graph, Node, Node, Node]:
    """SENSOR(text) -> ASSOC -> OUTPUT(text), all 8-dim for fast tests."""
    graph = Graph()
    dim = 8
    sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    assoc = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
    out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    graph.add_node(sensor, modality="text")
    graph.add_node(assoc)
    graph.add_node(out, modality="text")
    graph.add_edge(
        Edge(
            source_id=sensor.id,
            target_id=assoc.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=0.5,
        )
    )
    graph.add_edge(
        Edge(
            source_id=assoc.id,
            target_id=out.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=0.5,
        )
    )
    return graph, sensor, assoc, out


class TestBackpropStep:
    def test_loss_decreases_over_steps(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)  # seed before graph init for deterministic weights
        graph, sensor, _, out = _three_node_graph(config)
        # Use a moderate LR for this test so updates don't overshoot on
        # the scale-1.0 random init.
        tame_config = SOMAConfig(
            base_lr=0.01,
            youth_lr_multiplier=1.0,
            hebbian_lr=0.0001,
            maturity_increment=0.01,
            activation_threshold=0.01,
        )
        input_data = torch.randn(sensor.output_dim)
        target = torch.zeros(out.output_dim)

        losses: list[float] = []
        for step in range(50):
            outputs, activations = execute_graph(
                graph, inputs={"text": input_data}, current_step=step
            )
            out_tensor = outputs["text"]
            loss = F.mse_loss(out_tensor, target)
            losses.append(float(loss.item()))
            update_step(graph, loss, activations, tame_config)

        # Some progress expected in 50 steps on a trivial regression.
        assert losses[-1] < losses[0] * 0.8

    def test_node_params_change(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _three_node_graph(config)
        before = assoc.linear1.weight.detach().clone()
        input_data = torch.randn(sensor.output_dim)
        outputs, activations = execute_graph(graph, inputs={"text": input_data}, current_step=1)
        loss = outputs["text"].sum()  # simple scalar
        update_step(graph, loss, activations, config)
        assert not torch.allclose(before, assoc.linear1.weight)

    def test_gradients_zeroed_by_default(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum()
        update_step(graph, loss, activations, config)
        # All grads should be zero (or None) after zero_grad=True.
        for p in assoc.parameters():
            assert p.grad is None or float(p.grad.abs().sum().item()) == 0.0


class TestHebbianUpdate:
    def test_coactivation_bumps_weight(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        edge = graph.get_edge(sensor.id, assoc.id)
        w_before = float(edge.weight.item())
        # Strong input so the activation threshold is surpassed.
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.ones(sensor.output_dim) * 5.0},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0  # no gradient contribution, isolate Hebbian
        update_step(graph, loss, activations, config)
        assert edge.coactivation_count == 1
        # Weight should increase (Hebbian adds a positive quantity for positive
        # activations magnitudes).
        assert float(edge.weight.item()) > w_before

    def test_edge_weight_clamp(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        edge = graph.get_edge(sensor.id, assoc.id)
        # Start with weight above MAX_EDGE_WEIGHT.
        with torch.no_grad():
            edge.weight.fill_(config.max_edge_weight + 10.0)
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.ones(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0
        update_step(graph, loss, activations, config)
        assert abs(float(edge.weight.item())) <= config.max_edge_weight + 1e-6

    def test_strength_updated(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        edge = graph.get_edge(sensor.id, assoc.id)
        initial_strength = edge.strength
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.ones(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0
        update_step(graph, loss, activations, config)
        # Strength EMA should have moved (either up or down).
        assert edge.strength != initial_strength


class TestMaturityAdvance:
    def test_maturity_increases(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0
        start = assoc.maturity
        update_step(graph, loss, activations, config)
        assert assoc.maturity == pytest.approx(start + config.maturity_increment)

    def test_maturity_capped_at_one(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        assoc.maturity = 0.995
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0
        update_step(graph, loss, activations, config)
        assert assoc.maturity <= 1.0


class TestHomeostaticGain:
    def test_overactive_node_reduces_gain(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        # Simulate highly-active node.
        assoc.activation_ema = assoc.target_activation * 5.0
        initial_gain = assoc.gain
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0
        update_step(graph, loss, activations, config)
        assert assoc.gain < initial_gain

    def test_underactive_node_raises_gain(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        assoc.activation_ema = assoc.target_activation * 0.1
        initial_gain = assoc.gain
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0
        update_step(graph, loss, activations, config)
        assert assoc.gain > initial_gain

    def test_gain_clamped_to_bounds(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        # Force out-of-range gain; update_step should clamp it.
        assoc.gain = 0.05
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum() * 0.0
        update_step(graph, loss, activations, config)
        assert assoc.gain >= config.gain_min

    def test_gain_applies_to_inactive_nodes(self, config: SOMAConfig) -> None:
        """Homeostasis should act on *all* nodes, not just active ones."""
        graph = Graph()
        dim = 8
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        dangling = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(dangling)
        graph.add_node(out, modality="text")
        graph.add_edge(
            Edge(
                source_id=sensor.id,
                target_id=out.id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
                initial_weight=1.0,
            )
        )
        dangling.activation_ema = dangling.target_activation * 0.1
        initial_gain = dangling.gain
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(dim)},
            current_step=1,
        )
        loss = outputs["text"].sum()
        update_step(graph, loss, activations, config)
        # Dangling is inactive but homeostasis still ran.
        assert dangling.gain > initial_gain


class TestLrMultiplier:
    def test_zero_multiplier_freezes_weights(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _three_node_graph(config)
        before = assoc.linear1.weight.detach().clone()
        outputs, activations = execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=1,
        )
        loss = outputs["text"].sum()
        update_step(graph, loss, activations, config, lr_multiplier=0.0)
        # With LR=0, the SGD step should leave linear1.weight unchanged.
        assert torch.allclose(before, assoc.linear1.weight)


class TestEdgeWeightDecay:
    """Verify Hebbian counterbalance: weights don't drift to clamp."""

    def _two_node_decay_setup(
        self, *, initial_weight: float, decay: float, hebbian_lr: float = 0.001
    ) -> tuple[Graph, Edge, SOMAConfig]:
        cfg = SOMAConfig(
            base_lr=0.001,
            hebbian_lr=hebbian_lr,
            edge_weight_decay=decay,
            activation_threshold=0.01,
        )
        graph = Graph()
        dim = 4
        sensor = Node(NodeType.SENSOR, dim, dim, dim, 0, cfg)
        out = Node(NodeType.OUTPUT, dim, dim, dim, 0, cfg)
        graph.add_node(sensor, modality="text")
        graph.add_node(out, modality="text")
        edge = Edge(
            source_id=sensor.id,
            target_id=out.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=initial_weight,
        )
        graph.add_edge(edge)
        return graph, edge, cfg

    def test_inactive_edge_decays_toward_zero(self) -> None:
        graph, edge, cfg = self._two_node_decay_setup(
            initial_weight=1.0, decay=0.99, hebbian_lr=0.0
        )
        loss = torch.zeros((), requires_grad=True)
        initial = float(edge.weight.detach().item())
        for _ in range(10):
            update_step(graph, loss, activations={}, config=cfg)
        final = float(edge.weight.detach().item())
        assert final < initial, f"weight should decay; initial={initial}, final={final}"
        assert abs(final - initial * (0.99**10)) < 1e-4

    def test_co_active_edge_reaches_equilibrium_below_clamp(self) -> None:
        graph, edge, cfg = self._two_node_decay_setup(
            initial_weight=0.01, decay=0.99, hebbian_lr=0.01
        )
        # Equilibrium w_eq = hebbian_lr * s_rms * t_rms / (1 - decay)
        # torch.ones(4) has RMS 1.0, so s*t = 1. w_eq = 0.01 * 1 / 0.01 = 1.0
        sensor_id, out_id = list(graph.nodes.keys())
        activations = {
            sensor_id: torch.ones(4),
            out_id: torch.ones(4),
        }
        loss = torch.zeros((), requires_grad=True)
        for _ in range(2000):
            update_step(graph, loss, activations=activations, config=cfg)
        final = float(edge.weight.detach().item())
        assert 0.5 < final < cfg.max_edge_weight, (
            f"equilibrium should approach 1.0 but stay under clamp, got {final}"
        )

    def test_decay_does_not_flip_sign(self) -> None:
        graph, edge, cfg = self._two_node_decay_setup(
            initial_weight=-1.0, decay=0.5, hebbian_lr=0.0
        )
        loss = torch.zeros((), requires_grad=True)
        for _ in range(20):
            update_step(graph, loss, activations={}, config=cfg)
        final = float(edge.weight.detach().item())
        assert final < 0.0, f"negative weight stayed negative? got {final}"
        assert final > -1e-3, f"weight should decay close to zero, got {final}"


class TestGradientClipping:
    """Verify gradient clipping bounds the update magnitude."""

    def test_huge_gradient_gets_clipped(self, config: SOMAConfig) -> None:
        graph, _sensor, assoc, _out = _three_node_graph(config)
        cfg = SOMAConfig(
            base_lr=config.base_lr,
            hebbian_lr=config.hebbian_lr,
            grad_clip_max_norm=0.5,
            activation_threshold=config.activation_threshold,
        )
        # Inject huge gradients on the assoc node's params.
        for param in assoc.parameters():
            param.grad = torch.full_like(param, 100.0)
        loss = torch.zeros((), requires_grad=True)
        before = {id(p): p.detach().clone() for p in assoc.parameters()}
        update_step(graph, loss, activations={assoc.id: torch.ones(8)}, config=cfg)
        for p in assoc.parameters():
            delta = float((p.detach() - before[id(p)]).abs().max().item())
            assert delta < 1.0, f"clipped update should be small, got {delta}"

    def test_clipping_handles_none_grads(self, config: SOMAConfig) -> None:
        graph, _, _, _ = _three_node_graph(config)
        loss = torch.zeros((), requires_grad=True)
        # No backward call; all grads are None. Should not raise.
        update_step(graph, loss, activations={}, config=config)
