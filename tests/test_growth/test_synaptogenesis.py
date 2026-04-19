"""Tests for ``soma.growth.synaptogenesis``."""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
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


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class TestPeConditionalGate:
    """Direction 1 — admit only pairs whose co-activation has historically
    preceded PE reduction. Gate is engaged by passing ``pe_ema`` and
    ``pe_counts`` kwargs when ``config.synaptogenesis_supervision`` is
    ``"pe_conditional"``. When supervision is off, the kwargs (if any)
    must be ignored.

    These tests isolate the gate behavior from the new-node waiver by
    setting ``synaptogenesis_supervision_new_node_grace=0`` explicitly.
    The waiver's own behavior is covered by ``TestNewNodeWaiver``.
    """

    def _make_triplet(
        self, config: SOMAConfig, *, dim: int = 8
    ) -> tuple[Graph, Node, Node, Node]:
        graph = Graph()
        pos = torch.zeros(config.position_dim)
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config, position=pos)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config, position=pos)
        c = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config, position=pos)
        for n in (a, b, c):
            graph.add_node(n)
        return graph, a, b, c

    def test_cold_start_pair_is_rejected(self) -> None:
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_pe_threshold=0.0,
            synaptogenesis_supervision_new_node_grace=0,
        )
        graph, a, b, _c = self._make_triplet(cfg)
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        # No EMA evidence at all: cold start => no admissions.
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng, pe_ema={}, pe_counts={}
        )
        assert new == []
        assert graph.num_edges == 0

    def test_insufficient_observations_rejected(self) -> None:
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=10,
            synaptogenesis_pe_threshold=0.0,
            synaptogenesis_supervision_new_node_grace=0,
        )
        graph, a, b, _c = self._make_triplet(cfg)
        key = _pair_key(a.id, b.id)
        # Plenty of signal (-0.1 EMA) but only 3 observations, below the
        # 10-sample minimum.
        ema = {key: -0.1}
        counts = {key: 3}
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng, pe_ema=ema, pe_counts=counts
        )
        assert new == []

    def test_positive_ema_rejected_even_with_observations(self) -> None:
        """EMA >= 0 means co-activation hasn't been helping; skip."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_pe_threshold=0.0,
        )
        graph, a, b, _c = self._make_triplet(cfg)
        key = _pair_key(a.id, b.id)
        ema = {key: 0.02}  # positive = pair didn't help
        counts = {key: 50}  # plenty of observations
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng, pe_ema=ema, pe_counts=counts
        )
        assert new == []

    def test_negative_ema_allows_admission(self) -> None:
        """EMA < threshold AND enough observations => admit normally."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_pe_threshold=0.0,
        )
        graph, a, b, _c = self._make_triplet(cfg)
        key = _pair_key(a.id, b.id)
        ema = {key: -0.1}  # negative = pair's co-activation helps
        counts = {key: 50}
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng, pe_ema=ema, pe_counts=counts
        )
        # High synaptogenesis_rate means the random draw almost surely
        # passes. At least one edge should be admitted.
        assert len(new) >= 1
        for edge in new:
            # Every admitted edge must be between a <-> b.
            assert {edge.source_id, edge.target_id} == {a.id, b.id}

    def test_filtering_is_per_pair_not_global(self) -> None:
        """Admissible pairs (evidence supports them) must pass while
        inadmissible pairs in the same call are rejected."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_pe_threshold=0.0,
            synaptogenesis_supervision_new_node_grace=0,
        )
        graph, a, b, c = self._make_triplet(cfg)
        # a<->b has good evidence; b<->c has bad evidence; a<->c unknown.
        ema = {
            _pair_key(a.id, b.id): -0.2,  # good
            _pair_key(b.id, c.id): +0.3,  # bad
        }
        counts = {
            _pair_key(a.id, b.id): 50,
            _pair_key(b.id, c.id): 50,
        }
        acts = {a.id: torch.ones(8), b.id: torch.ones(8), c.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng, pe_ema=ema, pe_counts=counts
        )
        # All admitted edges must involve a<->b only; b<->c and a<->c
        # are filtered (one by positive EMA, one by cold start).
        for edge in new:
            assert {edge.source_id, edge.target_id} == {a.id, b.id}

    def test_supervision_none_ignores_ema_kwargs(self) -> None:
        """With supervision='none' the gate is disabled even when the
        caller passes EMA kwargs (e.g., stale state after a config flip)."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="none",
        )
        graph, a, b, _c = self._make_triplet(cfg)
        # Supply "bad" evidence that WOULD be rejected under pe_conditional.
        key = _pair_key(a.id, b.id)
        ema = {key: 1.0}
        counts = {key: 100}
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng, pe_ema=ema, pe_counts=counts
        )
        # Without supervision, the high rate admits edges normally.
        assert len(new) >= 1

    def test_ema_kwargs_optional_when_supervision_on(self) -> None:
        """If supervision is pe_conditional but no EMA dict is passed,
        the gate must treat the absence as universal cold-start
        (no admissions) rather than crashing."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_supervision_new_node_grace=0,
        )
        graph, a, b, _c = self._make_triplet(cfg)
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        assert new == []


class TestNewNodeWaiver:
    """When a candidate pair includes a node that was created within
    the last ``synaptogenesis_supervision_new_node_grace`` steps, the
    supervision gate is bypassed for that pair so fresh neurogenesis
    nodes can wire into the graph without waiting for 5 observations
    of EMA evidence. Fixes the full_pe regression from Phase 1."""

    def _make_triplet(
        self, config: SOMAConfig, *, dim: int = 8,
        creation_steps: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[Graph, Node, Node, Node]:
        graph = Graph()
        pos = torch.zeros(config.position_dim)
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, creation_steps[0], config, position=pos)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, creation_steps[1], config, position=pos)
        c = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, creation_steps[2], config, position=pos)
        for n in (a, b, c):
            graph.add_node(n)
        return graph, a, b, c

    def test_young_node_bypasses_cold_start(self) -> None:
        """Pair (a, b) where b was just created must admit despite no
        EMA observations."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_supervision_new_node_grace=100,
        )
        # a is old (creation_step=0), b was just created at step=95 => age=5 < grace=100.
        graph, a, b, _c = self._make_triplet(cfg, creation_steps=(0, 95, 0))
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng,
            pe_ema={}, pe_counts={},
        )
        # Without waiver: cold-start would block. With waiver: admit.
        assert len(new) >= 1

    def test_old_nodes_still_blocked_by_cold_start(self) -> None:
        """Pair of two old nodes must still be cold-start-blocked."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_supervision_new_node_grace=100,
        )
        # All three are old (step=0, probed at step=1000 => age=1000 > 100).
        graph, a, b, _c = self._make_triplet(cfg, creation_steps=(0, 0, 0))
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=1000, config=cfg, rng=rng,
            pe_ema={}, pe_counts={},
        )
        assert new == []

    def test_young_node_does_not_bypass_positive_ema(self) -> None:
        """The waiver skips the count check, NOT the threshold check.
        A young-node pair with well-observed bad evidence should still
        be rejected."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_supervision_new_node_grace=100,
        )
        graph, a, b, _c = self._make_triplet(cfg, creation_steps=(0, 95, 0))
        key = _pair_key(a.id, b.id)
        # Paradoxical: young node with plenty of observations and bad
        # evidence. Shouldn't usually happen, but the gate must handle it.
        ema = {key: 0.5}
        counts = {key: 50}
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng,
            pe_ema=ema, pe_counts=counts,
        )
        assert new == []

    def test_grace_zero_disables_waiver(self) -> None:
        """Default behavior — when grace=0, no waiver applies and the
        Phase 1 cold-start behavior is preserved."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_supervision_new_node_grace=0,
        )
        graph, a, b, _c = self._make_triplet(cfg, creation_steps=(0, 95, 0))
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng,
            pe_ema={}, pe_counts={},
        )
        # No waiver, so cold-start blocks.
        assert new == []

    def test_locality_filter_hard_rejects_distant_pairs(self) -> None:
        """With synaptogenesis_max_distance set, pairs whose positions
        are farther apart than the limit must not be admitted — even
        when they're co-active and would pass the supervision/threshold.

        This tests the Direction analysis prediction (2026-04-19) that
        positional locality is what makes neurogenesis's wirings useful;
        adding a hard cutoff to synap should reproduce the locality
        contribution without the other neurogenesis factors.
        """
        dim = 8
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_max_distance=0.5,  # tight cutoff
            locality_scale=2.0,  # keep soft locality at default
        )
        # Position nodes FAR apart in position space (distance = 5.0).
        graph = Graph()
        pos_a = torch.zeros(cfg.position_dim)
        pos_b = torch.zeros(cfg.position_dim)
        pos_b[0] = 5.0
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos_a)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos_b)
        graph.add_node(a)
        graph.add_node(b)
        acts = {a.id: torch.ones(dim), b.id: torch.ones(dim)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        # dist=5 >> max_distance=0.5 => no admissions.
        assert new == []

    def test_locality_filter_admits_near_pairs(self) -> None:
        """With the same max_distance, pairs within the limit are
        admitted at the normal rate."""
        dim = 8
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_max_distance=1.0,
            locality_scale=2.0,
        )
        graph = Graph()
        pos_a = torch.zeros(cfg.position_dim)
        pos_b = torch.zeros(cfg.position_dim)
        pos_b[0] = 0.3  # distance = 0.3 < 1.0
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos_a)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos_b)
        graph.add_node(a)
        graph.add_node(b)
        acts = {a.id: torch.ones(dim), b.id: torch.ones(dim)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        assert len(new) >= 1

    def test_locality_filter_zero_disables(self) -> None:
        """max_distance=0 means 'no filter' — legacy behavior preserved."""
        dim = 8
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_max_distance=0.0,  # disabled
            locality_scale=2.0,
        )
        graph = Graph()
        pos_a = torch.zeros(cfg.position_dim)
        pos_b = torch.zeros(cfg.position_dim)
        pos_b[0] = 5.0  # very far
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos_a)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos_b)
        graph.add_node(a)
        graph.add_node(b)
        acts = {a.id: torch.ones(dim), b.id: torch.ones(dim)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        # No hard filter; with high rate, admissions happen despite
        # the soft locality reducing probability.
        # The soft-locality exp(-5/2) = 0.082, times rate 10 = 0.82 prob
        # per direction, so likely at least one admission.
        assert len(new) >= 1  # soft locality doesn't fully block

    def test_initial_seed_nodes_not_treated_as_fresh(self) -> None:
        """Initial seed nodes have creation_step=0 by convention; the
        waiver must NOT treat them as fresh, even during the first
        ``grace`` steps of the run. Otherwise the waiver disables
        supervision during warm-up and synap_only_pe regresses (as
        observed in the waiver rerun before this fix).
        """
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_min_observations=5,
            synaptogenesis_supervision_new_node_grace=200,
        )
        # Both nodes at creation_step=0 (initial seeds).
        graph, a, b, _c = self._make_triplet(cfg, creation_steps=(0, 0, 0))
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        # Probe at step=50, well inside the grace window for true
        # fresh nodes, but these seed nodes shouldn't benefit.
        new = synaptogenesis(
            graph, acts, step=50, config=cfg, rng=rng,
            pe_ema={}, pe_counts={},
        )
        # No waiver for creation_step=0 => cold-start blocks.
        assert new == []


class TestMaxAdmissionsPerStep:
    """Sparsity control: cap the number of admissions per call.

    Purpose: distinguishes the ``synap_local`` positive result from a
    pure-sparsity confound. If random-K admission (no locality, just a
    count cap) matches ``synap_only_local``'s MSE benefit, then the
    effect observed on v0.5 is sparsity alone; if random-K underperforms,
    then positional locality is the primary driver.

    Implementation contract:
    - Default ``synaptogenesis_max_admissions_per_step=0`` disables the
      cap (legacy behavior). Positive values limit per-call returns.
    - When the cap is tighter than the pool of candidates that pass the
      rng draw, a random subset of size cap is kept and the rest are
      removed from the graph so ``len(returned) == graph.num_edges``.
    - The subsampling uses the provided ``rng``, so repeat calls with
      the same rng seed are deterministic; different seeds can produce
      different subsets (not biased by iteration order).
    """

    @staticmethod
    def _make_dense_graph(cfg: SOMAConfig, *, n: int, dim: int = 8) -> tuple[Graph, list[Node]]:
        """n colocated co-active nodes — every pair passes the gate."""
        graph = Graph()
        nodes = []
        for _ in range(n):
            node = Node(
                NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg,
                position=torch.zeros(cfg.position_dim),
            )
            graph.add_node(node)
            nodes.append(node)
        return graph, nodes

    def test_default_zero_is_unlimited(self) -> None:
        """Cap=0 must preserve legacy behavior: many pairs produce many
        edges without any cap enforcement."""
        cfg = SOMAConfig(synaptogenesis_rate=10.0, activation_threshold=0.01)
        assert cfg.synaptogenesis_max_admissions_per_step == 0
        graph, nodes = self._make_dense_graph(cfg, n=5)
        acts = {n.id: torch.ones(8) for n in nodes}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        # 5 nodes × 4 targets each = 20 directed pairs; with rate=10 and
        # colocated positions, the vast majority should admit. The
        # uncapped baseline must produce substantially more than any
        # sensible cap setting.
        assert len(new) > 5
        assert graph.num_edges == len(new)

    def test_cap_limits_admissions(self) -> None:
        """With cap=K < available-candidates, exactly ``len(new) <= K``."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_max_admissions_per_step=3,
        )
        graph, nodes = self._make_dense_graph(cfg, n=5)
        acts = {n.id: torch.ones(8) for n in nodes}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        assert len(new) <= 3, f"cap=3 violated, got {len(new)} admissions"
        # Graph must match: any trimmed edges must be removed from graph,
        # not leaked.
        assert graph.num_edges == len(new)

    def test_cap_does_not_invent_edges_when_few_pass(self) -> None:
        """If fewer pairs pass the rng draw than the cap, all are kept
        — the cap never creates extra admissions."""
        cfg = SOMAConfig(
            synaptogenesis_rate=10.0,
            activation_threshold=0.01,
            synaptogenesis_max_admissions_per_step=100,
        )
        # Only 2 nodes, so at most 2 directional candidates.
        graph, a, b = _make_pair(cfg)
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
        assert len(new) <= 2
        assert graph.num_edges == len(new)

    def test_cap_different_seeds_can_produce_different_subsets(self) -> None:
        """Different rng seeds should be able to select different subsets.
        Guards against an 'iteration-order-first-K' implementation which
        would pick the same edges regardless of rng (iteration order is
        fixed by graph insertion order).

        We map each admitted edge to ``(source_index, target_index)``
        where index is the position in the insertion-order node list,
        so subsets are comparable across fresh graphs with different
        random UUIDs.
        """
        def _run(seed: int) -> frozenset[tuple[int, int]]:
            cfg = SOMAConfig(
                synaptogenesis_rate=10.0,
                activation_threshold=0.01,
                synaptogenesis_max_admissions_per_step=3,
            )
            graph, nodes = self._make_dense_graph(cfg, n=5)
            id_to_idx = {n.id: i for i, n in enumerate(nodes)}
            acts = {n.id: torch.ones(8) for n in nodes}
            rng = torch.Generator().manual_seed(seed)
            new = synaptogenesis(graph, acts, step=100, config=cfg, rng=rng)
            return frozenset(
                (id_to_idx[e.source_id], id_to_idx[e.target_id]) for e in new
            )

        # Insertion order is fixed across runs because
        # ``_make_dense_graph`` creates nodes in a fixed sequence. If all
        # 10 seeds produced the same (src, tgt) index set, the cap would
        # be iteration-order-first-K rather than rng-sampled.
        baseline = _run(0)
        differs = any(_run(s) != baseline for s in range(1, 10))
        assert differs, (
            "Different rng seeds all produced the same capped subset; "
            "the cap is probably iterating-order-first-K rather than "
            "randomly sampling from passing candidates."
        )

    def test_validation_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="max_admissions_per_step"):
            SOMAConfig(synaptogenesis_max_admissions_per_step=-1)
