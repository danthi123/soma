"""Tests for ``soma.core.graph.Graph``."""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig()


def _make_node(
    config: SOMAConfig,
    node_type: NodeType = NodeType.ASSOCIATOR,
    *,
    step: int = 0,
    input_dim: int | None = None,
    output_dim: int | None = None,
    position: torch.Tensor | None = None,
) -> Node:
    return Node.make(
        node_type,
        config,
        creation_step=step,
        input_dim=input_dim,
        output_dim=output_dim,
        position=position,
    )


def _make_edge(source: Node, target: Node, *, step: int = 0, weight: float = 0.1) -> Edge:
    return Edge(
        source_id=source.id,
        target_id=target.id,
        source_output_dim=source.output_dim,
        target_input_dim=target.input_dim,
        creation_step=step,
        initial_weight=weight,
    )


class TestAddRemoveNodes:
    def test_add_and_size(self, config: SOMAConfig) -> None:
        graph = Graph()
        assert len(graph) == 0
        assert graph.num_nodes == 0
        n = _make_node(config)
        graph.add_node(n)
        assert graph.num_nodes == 1
        assert n.id in graph.nodes
        assert graph.nodes[n.id] is n

    def test_rejects_duplicate_add(self, config: SOMAConfig) -> None:
        graph = Graph()
        n = _make_node(config)
        graph.add_node(n)
        with pytest.raises(ValueError, match="already in the graph"):
            graph.add_node(n)

    def test_sensor_requires_modality(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = _make_node(config, NodeType.SENSOR)
        with pytest.raises(ValueError, match="modality"):
            graph.add_node(sensor)

    def test_sensor_registered(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = _make_node(config, NodeType.SENSOR)
        graph.add_node(sensor, modality="text")
        assert graph.get_sensor("text") is sensor
        assert graph.sensor_nodes == {"text": sensor}

    def test_duplicate_sensor_modality_rejected(self, config: SOMAConfig) -> None:
        graph = Graph()
        s1 = _make_node(config, NodeType.SENSOR)
        s2 = _make_node(config, NodeType.SENSOR)
        graph.add_node(s1, modality="text")
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(s2, modality="text")

    def test_output_registered(self, config: SOMAConfig) -> None:
        graph = Graph()
        out = _make_node(config, NodeType.OUTPUT)
        graph.add_node(out, modality="text")
        assert graph.get_output("text") is out
        assert graph.output_nodes == {"text": out}

    def test_modality_rejected_for_non_boundary(self, config: SOMAConfig) -> None:
        graph = Graph()
        n = _make_node(config, NodeType.ASSOCIATOR)
        with pytest.raises(ValueError, match="only valid"):
            graph.add_node(n, modality="text")

    def test_remove_node_cascades_edges(self, config: SOMAConfig) -> None:
        graph = Graph()
        n1 = _make_node(config)
        n2 = _make_node(config)
        n3 = _make_node(config)
        for n in (n1, n2, n3):
            graph.add_node(n)
        e12 = _make_edge(n1, n2)
        e13 = _make_edge(n1, n3)
        e23 = _make_edge(n2, n3)
        graph.add_edge(e12)
        graph.add_edge(e13)
        graph.add_edge(e23)
        assert graph.num_edges == 3

        graph.remove_node(n2.id)
        assert graph.num_edges == 1  # only e13 remains
        assert graph.has_edge(n1.id, n3.id)
        assert not graph.has_edge(n1.id, n2.id)
        assert not graph.has_edge(n2.id, n3.id)
        assert n2.id not in graph.nodes

    def test_remove_sensor_blocked(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = _make_node(config, NodeType.SENSOR)
        graph.add_node(sensor, modality="text")
        with pytest.raises(ValueError, match="invariant"):
            graph.remove_node(sensor.id)

    def test_remove_sensor_with_override(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = _make_node(config, NodeType.SENSOR)
        graph.add_node(sensor, modality="text")
        graph.remove_node(sensor.id, allow_boundary_removal=True)
        assert sensor.id not in graph.nodes
        assert "text" not in graph.sensor_nodes

    def test_remove_missing_node_raises(self, config: SOMAConfig) -> None:
        graph = Graph()
        with pytest.raises(KeyError):
            graph.remove_node("missing")


class TestAddRemoveEdges:
    def test_add_and_lookup(self, config: SOMAConfig) -> None:
        graph = Graph()
        n1 = _make_node(config)
        n2 = _make_node(config)
        graph.add_node(n1)
        graph.add_node(n2)
        e = _make_edge(n1, n2)
        graph.add_edge(e)
        assert graph.num_edges == 1
        assert graph.has_edge(n1.id, n2.id)
        assert not graph.has_edge(n2.id, n1.id)
        assert graph.get_edge(n1.id, n2.id) is e
        assert graph.edges[e.id] is e

    def test_reject_duplicate_edge_pair(self, config: SOMAConfig) -> None:
        graph = Graph()
        n1 = _make_node(config)
        n2 = _make_node(config)
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(_make_edge(n1, n2))
        with pytest.raises(ValueError, match="already exists"):
            graph.add_edge(_make_edge(n1, n2))

    def test_reject_edge_with_missing_endpoint(self, config: SOMAConfig) -> None:
        graph = Graph()
        n1 = _make_node(config)
        n2 = _make_node(config)
        graph.add_node(n1)
        # n2 not in graph.
        with pytest.raises(ValueError, match="target"):
            graph.add_edge(_make_edge(n1, n2))

    def test_reject_dim_mismatch(self, config: SOMAConfig) -> None:
        graph = Graph()
        n1 = _make_node(config, output_dim=32)
        n2 = _make_node(config, input_dim=32)
        graph.add_node(n1)
        graph.add_node(n2)
        # Bogus edge with wrong source dim declared.
        bogus = Edge(
            source_id=n1.id,
            target_id=n2.id,
            source_output_dim=999,
            target_input_dim=32,
            creation_step=0,
        )
        with pytest.raises(ValueError, match="source dim"):
            graph.add_edge(bogus)

    def test_incoming_outgoing(self, config: SOMAConfig) -> None:
        graph = Graph()
        a = _make_node(config)
        b = _make_node(config)
        c = _make_node(config)
        for n in (a, b, c):
            graph.add_node(n)
        e_ab = _make_edge(a, b)
        e_cb = _make_edge(c, b)
        e_bc = _make_edge(b, c)
        for e in (e_ab, e_cb, e_bc):
            graph.add_edge(e)
        incoming_b = {e.id for e in graph.get_incoming_edges(b.id)}
        assert incoming_b == {e_ab.id, e_cb.id}
        outgoing_b = {e.id for e in graph.get_outgoing_edges(b.id)}
        assert outgoing_b == {e_bc.id}

    def test_remove_edge_clears_indexes(self, config: SOMAConfig) -> None:
        graph = Graph()
        a = _make_node(config)
        b = _make_node(config)
        graph.add_node(a)
        graph.add_node(b)
        e = _make_edge(a, b)
        graph.add_edge(e)
        graph.remove_edge(e.id)
        assert graph.num_edges == 0
        assert not graph.has_edge(a.id, b.id)
        assert graph.get_incoming_edges(b.id) == []
        assert graph.get_outgoing_edges(a.id) == []


class TestQueries:
    def test_nodes_by_type(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = _make_node(config, NodeType.SENSOR)
        assoc = _make_node(config, NodeType.ASSOCIATOR)
        out = _make_node(config, NodeType.OUTPUT)
        graph.add_node(sensor, modality="text")
        graph.add_node(assoc)
        graph.add_node(out, modality="text")
        assert graph.nodes_by_type(NodeType.SENSOR) == [sensor]
        assert graph.nodes_by_type(NodeType.ASSOCIATOR) == [assoc]
        assert graph.nodes_by_type(NodeType.INTEGRATOR) == []

    def test_active_nodes(self, config: SOMAConfig) -> None:
        graph = Graph()
        n_recent = _make_node(config)
        n_old = _make_node(config)
        n_recent.last_active_step = 10
        n_old.last_active_step = 0
        graph.add_node(n_recent)
        graph.add_node(n_old)
        assert graph.active_nodes(current_step=10, lookback=1) == [n_recent]
        assert set(graph.active_nodes(current_step=10, lookback=100)) == {n_recent, n_old}

    def test_active_edges(self, config: SOMAConfig) -> None:
        graph = Graph()
        a = _make_node(config)
        b = _make_node(config)
        c = _make_node(config)
        for n in (a, b, c):
            graph.add_node(n)
        e1 = _make_edge(a, b)
        e2 = _make_edge(b, c)
        graph.add_edge(e1)
        graph.add_edge(e2)
        e1.mark_active(50)
        e2.mark_active(10)
        active = graph.active_edges(current_step=50, lookback=1)
        assert active == [e1]

    def test_most_active_nodes(self, config: SOMAConfig) -> None:
        graph = Graph()
        n1 = _make_node(config)
        n2 = _make_node(config)
        n3 = _make_node(config)
        n1.activation_ema = 0.1
        n2.activation_ema = 0.9
        n3.activation_ema = 0.5
        for n in (n1, n2, n3):
            graph.add_node(n)
        top2 = graph.most_active_nodes(k=2)
        assert [n.activation_ema for n in top2] == [pytest.approx(0.9), pytest.approx(0.5)]

    def test_get_nearest_nodes(self, config: SOMAConfig) -> None:
        graph = Graph()
        pos_dim = config.position_dim
        near = _make_node(config, position=torch.zeros(pos_dim))
        far = _make_node(
            config,
            position=torch.ones(pos_dim) * 10.0,
        )
        mid = _make_node(config, position=torch.ones(pos_dim) * 2.0)
        for n in (near, far, mid):
            graph.add_node(n)
        query = torch.zeros(pos_dim)
        result = graph.get_nearest_nodes(query, k=2)
        assert [n.id for n in result] == [near.id, mid.id]
        # Exclude removes near; far + mid remain, nearest is mid.
        result2 = graph.get_nearest_nodes(query, k=1, exclude=[near.id])
        assert [n.id for n in result2] == [mid.id]


class TestSerialization:
    def test_round_trip(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = _make_node(config, NodeType.SENSOR, step=0)
        assoc = _make_node(config, NodeType.ASSOCIATOR, step=1)
        out = _make_node(config, NodeType.OUTPUT, step=2)
        graph.add_node(sensor, modality="text")
        graph.add_node(assoc)
        graph.add_node(out, modality="text")

        # Use matched dims to avoid projections mid-test:
        # sensor.output=64, assoc.input=64, assoc.output=64, out.input=64.
        e_sa = _make_edge(sensor, assoc, step=2, weight=0.5)
        e_ao = _make_edge(assoc, out, step=3, weight=0.2)
        graph.add_edge(e_sa)
        graph.add_edge(e_ao)

        # Mutate some state to check it round-trips.
        assoc.maturity = 0.8
        assoc.activation_ema = 0.42
        e_sa.coactivation_count = 99
        e_sa.strength = 1.5
        e_sa.mark_active(7)

        snapshot = graph.serialize()
        rebuilt = Graph.deserialize(snapshot, config)

        assert rebuilt.num_nodes == 3
        assert rebuilt.num_edges == 2
        assert rebuilt.get_sensor("text").id == sensor.id
        assert rebuilt.get_output("text").id == out.id
        rebuilt_assoc = rebuilt.nodes[assoc.id]
        assert rebuilt_assoc.maturity == pytest.approx(0.8)
        assert rebuilt_assoc.activation_ema == pytest.approx(0.42)
        rebuilt_edge = rebuilt.get_edge(sensor.id, assoc.id)
        assert rebuilt_edge.coactivation_count == 99
        assert rebuilt_edge.strength == pytest.approx(1.5)
        assert rebuilt_edge.last_active_step == 7
        # Learnable params match.
        assert torch.equal(
            rebuilt.nodes[assoc.id].linear1.weight,
            assoc.linear1.weight,
        )
        assert torch.equal(rebuilt_edge.weight, e_sa.weight)


class TestParameterDiscovery:
    def test_pytorch_discovers_node_and_edge_params(self, config: SOMAConfig) -> None:
        graph = Graph()
        n1 = _make_node(config)
        n2 = _make_node(config)
        graph.add_node(n1)
        graph.add_node(n2)
        e = _make_edge(n1, n2)
        graph.add_edge(e)
        params = list(graph.parameters())
        # n1: linear1 + linear2 = 4 params, n2 same, edge: weight = 1.
        # (no projection because same dims)
        assert len(params) == 4 + 4 + 1
        assert any(p is e.weight for p in params)
