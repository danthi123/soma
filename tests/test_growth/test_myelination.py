"""Tests for ``soma.growth.myelination``."""

from __future__ import annotations

import pytest

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.growth.myelination import (
    MyelinationResult,
    detect_linear_chains,
    myelination,
)


@pytest.fixture
def config() -> SOMAConfig:
    # Small thresholds so chains are ripe for our tests.
    return SOMAConfig(
        myelination_strength_threshold=0.1,
        myelination_age_threshold=50,
    )


def _make_chain_graph(
    config: SOMAConfig,
    chain_length: int,
    *,
    with_endpoints: bool = True,
    age: int = 100,
) -> tuple[Graph, list[str]]:
    """Build SENSOR -> [N chained ASSOCs] -> OUTPUT, all 4-dim.

    Returns (graph, chain_node_ids) — only the ASSOC chain.
    """
    graph = Graph()
    dim = 4
    sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    chain_nodes: list[Node] = [
        Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config) for _ in range(chain_length)
    ]
    if with_endpoints:
        graph.add_node(sensor, modality="text")
    for n in chain_nodes:
        graph.add_node(n)
    if with_endpoints:
        graph.add_node(out, modality="text")

    def _edge(src: Node, tgt: Node, *, strength: float = 1.0) -> Edge:
        e = Edge(
            source_id=src.id,
            target_id=tgt.id,
            source_output_dim=src.output_dim,
            target_input_dim=tgt.input_dim,
            creation_step=0,
            initial_weight=1.0,
        )
        e.strength = strength
        return e

    if with_endpoints:
        graph.add_edge(_edge(sensor, chain_nodes[0]))
    for a, b in zip(chain_nodes, chain_nodes[1:], strict=False):
        graph.add_edge(_edge(a, b))
    if with_endpoints:
        graph.add_edge(_edge(chain_nodes[-1], out))
    return graph, [n.id for n in chain_nodes]


class TestDetect:
    def test_detects_chain(self, config: SOMAConfig) -> None:
        graph, chain_ids = _make_chain_graph(config, chain_length=3)
        chains = detect_linear_chains(graph, min_length=2, max_length=4)
        assert len(chains) == 1
        assert chains[0] == chain_ids

    def test_respects_min_length(self, config: SOMAConfig) -> None:
        graph, _ = _make_chain_graph(config, chain_length=2)
        assert detect_linear_chains(graph, min_length=3, max_length=4) == []
        assert len(detect_linear_chains(graph, min_length=2, max_length=4)) == 1

    def test_respects_max_length(self, config: SOMAConfig) -> None:
        graph, chain_ids = _make_chain_graph(config, chain_length=4)
        chains = detect_linear_chains(graph, min_length=2, max_length=3)
        # Max length truncates the chain to 3 — the first 3 nodes.
        assert len(chains) == 1
        assert chains[0] == chain_ids[:3]

    def test_skips_branching_nodes(self, config: SOMAConfig) -> None:
        """Add a second outgoing edge from chain[0] so it's no longer in a linear chain."""
        graph, chain_ids = _make_chain_graph(config, chain_length=3)
        extra = Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config)
        graph.add_node(extra)
        # chain_ids[0] now has 2 outgoing edges -> disqualifies it from
        # being "internal chainable".
        graph.add_edge(
            Edge(
                source_id=chain_ids[0],
                target_id=extra.id,
                source_output_dim=4,
                target_input_dim=4,
                creation_step=0,
            )
        )
        chains = detect_linear_chains(graph)
        # The chain shrinks — chain_ids[1] and chain_ids[2] still form a chain.
        assert all(chain_ids[0] not in c for c in chains)

    def test_validation(self, config: SOMAConfig) -> None:
        graph, _ = _make_chain_graph(config, chain_length=3)
        with pytest.raises(ValueError):
            detect_linear_chains(graph, min_length=1, max_length=4)
        with pytest.raises(ValueError):
            detect_linear_chains(graph, min_length=3, max_length=2)


class TestMyelination:
    def test_ripe_chain_is_compressed(self, config: SOMAConfig) -> None:
        graph, chain_ids = _make_chain_graph(config, chain_length=3)
        result = myelination(graph, step=100, config=config)
        assert isinstance(result, MyelinationResult)
        assert result.chains_compressed == 1
        assert result.nodes_removed == 3
        # Chain nodes should be gone, one new node should remain.
        for nid in chain_ids:
            assert nid not in graph.nodes
        assert len(result.new_node_ids) == 1
        new_id = result.new_node_ids[0]
        assert new_id in graph.nodes

    def test_unripe_chain_is_skipped(self, config: SOMAConfig) -> None:
        graph, chain_ids = _make_chain_graph(config, chain_length=3)
        # Make edges weak.
        for a, b in zip(chain_ids, chain_ids[1:], strict=False):
            edge = graph.get_edge(a, b)
            edge.strength = 0.0  # below threshold
        result = myelination(graph, step=100, config=config)
        assert result.chains_compressed == 0
        assert result.nodes_removed == 0

    def test_boundary_nodes_never_in_chain(self, config: SOMAConfig) -> None:
        """SENSOR and OUTPUT must stay intact even when technically in a chain."""
        graph, chain_ids = _make_chain_graph(config, chain_length=2)
        # After myelination, sensors/outputs must still be present.
        myelination(graph, step=100, config=config)
        assert "text" in graph.sensor_nodes
        assert "text" in graph.output_nodes

    def test_edges_redirected(self, config: SOMAConfig) -> None:
        """Sensor -> chain[0] edge should become sensor -> merged; merged -> output."""
        graph, chain_ids = _make_chain_graph(config, chain_length=2)
        sensor = graph.get_sensor("text")
        out = graph.get_output("text")
        result = myelination(graph, step=100, config=config)
        assert len(result.new_node_ids) == 1
        new_id = result.new_node_ids[0]
        assert graph.has_edge(sensor.id, new_id)
        assert graph.has_edge(new_id, out.id)

    def test_rejects_negative_step(self, config: SOMAConfig) -> None:
        graph, _ = _make_chain_graph(config, chain_length=2)
        with pytest.raises(ValueError, match="step"):
            myelination(graph, step=-1, config=config)

    def test_rejects_bad_length_bounds(self, config: SOMAConfig) -> None:
        graph, _ = _make_chain_graph(config, chain_length=2)
        with pytest.raises(ValueError, match="min_length"):
            myelination(graph, step=100, config=config, min_length=1, max_length=4)
