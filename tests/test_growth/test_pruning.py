"""Tests for ``soma.growth.pruning``."""

from __future__ import annotations

import pytest

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.growth.pruning import PruningResult, pruning


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig(
        pruning_grace_period=100,
        edge_strength_threshold=0.01,
        inactivity_threshold=500,
    )


def _make_edge(source: Node, target: Node, *, step: int = 0) -> Edge:
    return Edge(
        source_id=source.id,
        target_id=target.id,
        source_output_dim=source.output_dim,
        target_input_dim=target.input_dim,
        creation_step=step,
    )


def _linear_graph(config: SOMAConfig) -> tuple[Graph, Node, Node, Node]:
    """SENSOR -> ASSOC -> OUTPUT, all matched dims."""
    graph = Graph()
    dim = 8
    s = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
    o = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    graph.add_node(s, modality="text")
    graph.add_node(a)
    graph.add_node(o, modality="text")
    graph.add_edge(_make_edge(s, a))
    graph.add_edge(_make_edge(a, o))
    return graph, s, a, o


class TestPruning:
    def test_grace_period_protects_young_edges(self, config: SOMAConfig) -> None:
        graph, _, _, _ = _linear_graph(config)
        # Step is just barely past creation, well within grace.
        result = pruning(graph, step=10, config=config)
        assert result.removed_edges == 0
        assert graph.num_edges == 2

    def test_weak_and_stale_edge_removed(self, config: SOMAConfig) -> None:
        graph, s, a, _ = _linear_graph(config)
        edge = graph.get_edge(s.id, a.id)
        edge.strength = config.edge_strength_threshold * 0.1  # weak
        edge.last_active_step = 0  # stale
        # Make sure we're past grace + inactivity.
        step = config.pruning_grace_period + config.inactivity_threshold + 10
        result = pruning(graph, step=step, config=config)
        assert edge.id in result.removed_edge_ids
        assert not graph.has_edge(s.id, a.id)

    def test_strong_edge_preserved(self, config: SOMAConfig) -> None:
        graph, s, a, _ = _linear_graph(config)
        edge = graph.get_edge(s.id, a.id)
        edge.strength = 1.0  # strong
        edge.last_active_step = 0  # stale, but strong enough to survive
        step = config.pruning_grace_period + config.inactivity_threshold + 10
        result = pruning(graph, step=step, config=config)
        assert edge.id not in result.removed_edge_ids
        assert graph.has_edge(s.id, a.id)

    def test_active_edge_preserved(self, config: SOMAConfig) -> None:
        graph, s, a, _ = _linear_graph(config)
        edge = graph.get_edge(s.id, a.id)
        edge.strength = 0.0  # weak
        step = config.pruning_grace_period + config.inactivity_threshold + 10
        edge.last_active_step = step - 1  # recently active
        result = pruning(graph, step=step, config=config)
        assert edge.id not in result.removed_edge_ids

    def test_orphaned_associator_removed(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = 8
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        orphan = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(orphan)  # no edges
        result = pruning(graph, step=100, config=config)
        assert orphan.id in result.removed_node_ids
        assert orphan.id not in graph.nodes
        assert sensor.id in graph.nodes  # boundary preserved

    def test_boundary_nodes_never_pruned(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = 8
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(out, modality="text")
        # No edges — would normally orphan the sensor/output.
        result = pruning(graph, step=10_000, config=config)
        assert result.removed_node_ids == []
        assert sensor.id in graph.nodes
        assert out.id in graph.nodes

    def test_cascade_removal(self, config: SOMAConfig) -> None:
        """Pruning a bridge edge then orphaning a node in one pass."""
        graph = Graph()
        dim = 8
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        mid = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(mid)
        graph.add_node(out, modality="text")
        e1 = _make_edge(sensor, mid)
        e2 = _make_edge(mid, out)
        graph.add_edge(e1)
        graph.add_edge(e2)
        # Make both edges weak + stale.
        for e in (e1, e2):
            e.strength = 0.0
            e.last_active_step = 0
        step = config.pruning_grace_period + config.inactivity_threshold + 100
        result = pruning(graph, step=step, config=config)
        assert set(result.removed_edge_ids) == {e1.id, e2.id}
        assert mid.id in result.removed_node_ids

    def test_returns_pruning_result(self, config: SOMAConfig) -> None:
        graph, _, _, _ = _linear_graph(config)
        result = pruning(graph, step=5, config=config)
        assert isinstance(result, PruningResult)
        assert result.removed_edges == 0
        assert result.removed_nodes == 0

    def test_rejects_negative_step(self, config: SOMAConfig) -> None:
        graph, _, _, _ = _linear_graph(config)
        with pytest.raises(ValueError, match="step"):
            pruning(graph, step=-1, config=config)
