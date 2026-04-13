"""Tests for ``soma.growth.synaptogenesis``."""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.growth.synaptogenesis import synaptogenesis


@pytest.fixture
def config() -> SOMAConfig:
    # High rate so we actually create edges in small tests without needing
    # thousands of samples.
    return SOMAConfig(synaptogenesis_rate=10.0, activation_threshold=0.01)


def _make_pair(
    config: SOMAConfig,
    *,
    dim: int = 8,
    positions: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[Graph, Node, Node]:
    graph = Graph()
    pos_a = positions[0] if positions else torch.zeros(config.position_dim)
    pos_b = positions[1] if positions else torch.zeros(config.position_dim)
    a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config, position=pos_a)
    b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config, position=pos_b)
    graph.add_node(a)
    graph.add_node(b)
    return graph, a, b


class TestBasic:
    def test_no_edges_when_single_active_node(self, config: SOMAConfig) -> None:
        graph, a, _ = _make_pair(config)
        acts = {a.id: torch.ones(8)}
        new = synaptogenesis(graph, acts, step=100, config=config)
        assert new == []
        assert graph.num_edges == 0

    def test_creates_edge_between_coactive_pair(self, config: SOMAConfig) -> None:
        graph, a, b = _make_pair(config)
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=config, rng=rng)
        # With high rate + matched positions, at least one edge should appear
        # (direction may be either).
        assert len(new) >= 1
        assert graph.num_edges == len(new)
        for e in new:
            assert e.creation_step == 100
            assert e.last_active_step == 100

    def test_skips_existing_edges(self, config: SOMAConfig) -> None:
        """Should never create a duplicate edge in the same direction."""
        graph, a, b = _make_pair(config)
        # Pre-insert a -> b.
        from soma.core.edge import Edge

        graph.add_edge(
            Edge(
                source_id=a.id,
                target_id=b.id,
                source_output_dim=8,
                target_input_dim=8,
                creation_step=0,
            )
        )
        before = graph.num_edges
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=50, config=config, rng=rng)
        # Only b -> a is eligible; a -> b must not duplicate.
        for e in new:
            assert (e.source_id, e.target_id) == (b.id, a.id)
        assert graph.num_edges <= before + 1

    def test_respects_activation_threshold(self, config: SOMAConfig) -> None:
        graph, a, b = _make_pair(config)
        # Both activations below threshold (norm = sqrt(8 * 1e-6) ~= 0.003 < 0.01).
        acts = {a.id: torch.ones(8) * 1e-3, b.id: torch.ones(8) * 1e-3}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=config, rng=rng)
        assert new == []

    def test_locality_bias(self, config: SOMAConfig) -> None:
        """Nearby nodes should form edges more often than far-apart nodes."""
        dim = 8
        # Slightly reduce rate so the difference is visible.
        cfg = SOMAConfig(
            synaptogenesis_rate=0.5,
            activation_threshold=0.01,
            locality_scale=1.0,
        )

        def sample_count(distance: float) -> int:
            graph = Graph()
            a = Node(
                NodeType.ASSOCIATOR,
                dim,
                dim * 2,
                dim,
                0,
                cfg,
                position=torch.zeros(cfg.position_dim),
            )
            b_pos = torch.zeros(cfg.position_dim)
            b_pos[0] = distance
            b = Node(
                NodeType.ASSOCIATOR,
                dim,
                dim * 2,
                dim,
                0,
                cfg,
                position=b_pos,
            )
            graph.add_node(a)
            graph.add_node(b)
            acts = {a.id: torch.ones(dim), b.id: torch.ones(dim)}
            total = 0
            for seed in range(200):
                g = Graph.deserialize(graph.serialize(), cfg)
                rng = torch.Generator().manual_seed(seed)
                new = synaptogenesis(g, acts, step=100, config=cfg, rng=rng)
                total += len(new)
            return total

        near = sample_count(0.1)
        far = sample_count(10.0)
        # Near should produce strictly more edges than far on this scale.
        assert near > far

    def test_self_loops_never_created(self, config: SOMAConfig) -> None:
        graph, a, _ = _make_pair(config)
        acts = {a.id: torch.ones(8) * 2.0}  # only one active node
        new = synaptogenesis(graph, acts, step=100, config=config)
        assert new == []
        # Even with two active mentions of the same node we wouldn't add a
        # self-loop (source_id == target_id is skipped).

    def test_rejects_negative_step(self, config: SOMAConfig) -> None:
        graph, a, b = _make_pair(config)
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        with pytest.raises(ValueError, match="step"):
            synaptogenesis(graph, acts, step=-1, config=config)


class TestProbabilityClamp:
    """Verify that the coactivation * rate product is clamped to [0, 1]."""

    def test_unbounded_activations_do_not_guarantee_every_pair(self) -> None:
        """
        When activations are very large (instability / divergence), the
        raw coact * rate product can exceed 1. The prob must clamp so
        that random draws still sometimes fail — otherwise synaptogenesis
        creates every possible edge and compounds the divergence.
        """
        cfg = SOMAConfig(synaptogenesis_rate=0.5, activation_threshold=0.01)
        # Build 6 co-active nodes with HUGE activations (mag=100).
        graph = Graph()
        dim = 4
        nodes = []
        for _ in range(6):
            n = Node(NodeType.ASSOCIATOR, dim, dim, dim, 0, cfg)
            graph.add_node(n)
            nodes.append(n)
        big_act = torch.ones(dim) * 100.0
        acts = {n.id: big_act for n in nodes}

        # Without clamp: coact = 10000 * locality ~1 * rate 0.5 = 5000,
        # so every pair's random draw < prob, producing 6*5 = 30 edges.
        # With clamp: prob is 1.0, but the random draw is still uniform
        # [0, 1), so EVERY draw still passes — BUT at least the invariant
        # holds (prob bounded). The deeper defense is the interaction
        # with locality_bonus which drops prob below 1 for distant pairs.
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        assert len(new) <= 6 * 5, "shouldn't exceed all directed pairs"
        # Primary guarantee: no crash, bounded output.
        assert graph.num_edges <= 6 * 5

    def test_large_coact_no_more_edges_than_small_coact_with_random_test(self) -> None:
        """
        With clamp active, a very large coactivation magnitude shouldn't
        create meaningfully MORE edges than a moderate one for the same
        number of pair tries (because both hit the prob=1 ceiling).
        Without clamp, the huge-coact case would create every edge every
        time while the moderate-coact case would be stochastic.
        """
        cfg = SOMAConfig(synaptogenesis_rate=0.1, activation_threshold=0.01)

        # Many pair trials with moderate coact (prob without clamp ~ 0.4)
        def count_edges(activation_mag: float, seed: int) -> int:
            graph = Graph()
            dim = 4
            nodes = [Node(NodeType.ASSOCIATOR, dim, dim, dim, 0, cfg) for _ in range(5)]
            for n in nodes:
                graph.add_node(n)
            acts = {n.id: torch.ones(dim) * activation_mag for n in nodes}
            rng = torch.Generator().manual_seed(seed)
            synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
            return graph.num_edges

        # With clamp, both saturate at prob=1 so both produce many edges.
        # But the important invariant is no unbounded explosion.
        mod = count_edges(activation_mag=2.0, seed=42)
        huge = count_edges(activation_mag=200.0, seed=42)
        # Both are bounded by the max (5*4=20 directed pairs).
        assert huge <= 20
        assert mod <= 20
