"""Tests for ``soma.metacognition.homeostasis.HomeostaticRegulator``."""

from __future__ import annotations

import pytest

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.metacognition.homeostasis import HomeostaticRegulator


@pytest.fixture
def empty_graph() -> Graph:
    return Graph()


@pytest.fixture
def dense_graph() -> Graph:
    """Graph with avg edges-per-node exceeding the default limit."""
    config = SOMAConfig()
    graph = Graph()
    nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(4)]
    for n in nodes:
        graph.add_node(n)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            graph.add_edge(
                Edge(
                    source_id=a.id,
                    target_id=b.id,
                    source_output_dim=4,
                    target_input_dim=4,
                    creation_step=0,
                )
            )
            graph.add_edge(
                Edge(
                    source_id=b.id,
                    target_id=a.id,
                    source_output_dim=4,
                    target_input_dim=4,
                    creation_step=0,
                )
            )
    return graph


class TestConstruction:
    def test_defaults(self) -> None:
        reg = HomeostaticRegulator()
        assert reg.max_nodes == 50_000
        assert reg.max_edges_per_node == pytest.approx(20.0)
        assert reg.global_lr_multiplier == pytest.approx(1.0)

    def test_from_config(self) -> None:
        config = SOMAConfig(max_nodes=100, max_edges_per_node=5.0)
        reg = HomeostaticRegulator.from_config(config)
        assert reg.max_nodes == 100
        assert reg.max_edges_per_node == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"max_nodes": 0}, "max_nodes"),
            ({"max_edges_per_node": 0.0}, "max_edges_per_node"),
            ({"spike_sigma": 0.0}, "spike_sigma"),
            ({"spike_dampen": 0.0}, "spike_dampen"),
            ({"spike_dampen": 1.5}, "spike_dampen"),
            ({"recovery_factor": 0.5}, "recovery_factor"),
            ({"ema_decay": 0.0}, "ema_decay"),
            ({"ema_decay": 1.0}, "ema_decay"),
        ],
    )
    def test_validation(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            HomeostaticRegulator(**kwargs)  # type: ignore[arg-type]


class TestUpdateMechanics:
    def test_steady_loss_keeps_lr_multiplier_capped_at_one(self, empty_graph: Graph) -> None:
        reg = HomeostaticRegulator()
        for _ in range(50):
            lr = reg.update(empty_graph, current_loss=1.0)
        assert lr == pytest.approx(1.0)

    def test_spike_reduces_multiplier(self, empty_graph: Graph) -> None:
        reg = HomeostaticRegulator(spike_dampen=0.5)
        for _ in range(200):
            reg.update(empty_graph, current_loss=0.1)
        before = reg.global_lr_multiplier
        reg.update(empty_graph, current_loss=100.0)  # huge spike
        assert reg.global_lr_multiplier < before

    def test_recovery_restores_multiplier(self, empty_graph: Graph) -> None:
        reg = HomeostaticRegulator(spike_dampen=0.5, recovery_factor=1.05)
        for _ in range(200):
            reg.update(empty_graph, current_loss=0.1)
        reg.update(empty_graph, current_loss=100.0)
        low = reg.global_lr_multiplier
        # Many steady-loss steps should raise it back toward 1.0.
        for _ in range(500):
            reg.update(empty_graph, current_loss=0.1)
        assert reg.global_lr_multiplier > low
        assert reg.global_lr_multiplier <= 1.0

    def test_nan_loss_returns_none(self, empty_graph: Graph) -> None:
        """Non-finite loss returns None instead of raising; state unchanged."""
        reg = HomeostaticRegulator()
        initial_ema = reg.loss_ema
        initial_count = reg._update_count
        assert reg.update(empty_graph, current_loss=float("nan")) is None
        assert reg.update(empty_graph, current_loss=float("inf")) is None
        assert reg.update(empty_graph, current_loss=float("-inf")) is None
        assert reg.loss_ema == initial_ema, "EMA must not be corrupted by bad loss"
        assert reg._update_count == initial_count, "step count must not advance"

    def test_finite_loss_after_nan_works_normally(self, empty_graph: Graph) -> None:
        reg = HomeostaticRegulator()
        reg.update(empty_graph, current_loss=float("nan"))  # ignored
        mult = reg.update(empty_graph, current_loss=0.5)
        assert mult is not None
        assert 0.0 < mult <= 1.0


class TestGrowthGates:
    def test_empty_graph_allows_both(self, empty_graph: Graph) -> None:
        reg = HomeostaticRegulator(max_nodes=10, max_edges_per_node=5.0)
        reg.update(empty_graph, current_loss=1.0)
        assert reg.allow_neurogenesis
        assert reg.allow_synaptogenesis

    def test_dense_graph_blocks_synaptogenesis(self, dense_graph: Graph) -> None:
        # dense_graph has ~3 avg edges-per-node given 4 nodes and 12 edges.
        reg = HomeostaticRegulator(max_nodes=100, max_edges_per_node=1.0)
        reg.update(dense_graph, current_loss=1.0)
        assert reg.allow_synaptogenesis is False

    def test_node_cap_blocks_neurogenesis(self, dense_graph: Graph) -> None:
        reg = HomeostaticRegulator(max_nodes=dense_graph.num_nodes, max_edges_per_node=100.0)
        reg.update(dense_graph, current_loss=1.0)
        assert reg.allow_neurogenesis is False


class TestSerialization:
    def test_state_dict_round_trip(self, empty_graph: Graph) -> None:
        reg1 = HomeostaticRegulator()
        for _ in range(30):
            reg1.update(empty_graph, current_loss=0.5)
        state = reg1.state_dict()

        reg2 = HomeostaticRegulator()
        reg2.load_state_dict(state)
        assert reg2.loss_ema == pytest.approx(reg1.loss_ema)
        assert reg2.global_lr_multiplier == pytest.approx(reg1.global_lr_multiplier)
        assert reg2.allow_neurogenesis == reg1.allow_neurogenesis

    def test_reset(self, empty_graph: Graph) -> None:
        reg = HomeostaticRegulator()
        reg.update(empty_graph, current_loss=100.0)
        reg.reset()
        assert reg.loss_ema == pytest.approx(0.0)
        assert reg.global_lr_multiplier == pytest.approx(1.0)
