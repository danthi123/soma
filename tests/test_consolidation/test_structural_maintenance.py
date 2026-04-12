"""Tests for structural maintenance inside ``consolidation_cycle`` (Unit 23)."""

from __future__ import annotations

import pytest
import torch

from soma.consolidation.cycle import ConsolidationResult, consolidation_cycle
from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.memory.episodic_memory import EpisodicMemory


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig(
        base_lr=0.01,
        consolidation_lr_ratio=0.5,
        consolidation_replay_steps=6,
        pruning_grace_period=5,
        edge_strength_threshold=0.1,
        inactivity_threshold=5,
        myelination_strength_threshold=100.0,  # never ripe -> no myelination
        consolidation_error_threshold=0.5,
    )


def _build_graph(config: SOMAConfig) -> tuple[Graph, Node, Node]:
    graph = Graph()
    dim = 4
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
            initial_weight=1.0,
        )
    )
    graph.add_edge(
        Edge(
            source_id=assoc.id,
            target_id=out.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=1.0,
        )
    )
    return graph, sensor, out


def _unpacker(
    experience: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    half = experience.shape[-1] // 2
    return {"text": experience[:half]}, {"text": experience[half:]}


def _seed_episodic(em: EpisodicMemory, n: int = 6, *, seed: int = 0) -> None:
    torch.manual_seed(seed)
    for t in range(n):
        exp = torch.randn(em.value_dim)
        em.encode(experience=exp, prediction_error=1.0, current_step=t)


class TestStructuralMaintenance:
    def test_maintenance_can_be_disabled(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, _, _ = _build_graph(config)
        em = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        _seed_episodic(em)
        result = consolidation_cycle(
            graph,
            em,
            current_step=100,
            config=config,
            experience_unpacker=_unpacker,
            run_structural_maintenance=False,
        )
        assert result.edges_pruned == 0
        assert result.nodes_pruned == 0
        assert result.chains_compressed == 0

    def test_runs_pruning(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, sensor, _ = _build_graph(config)
        # Pre-stage a weak, stale edge that pruning should remove during
        # the consolidation cycle. Sensor -> new orphan node -> dead.
        dim = 4
        dead = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        graph.add_node(dead)
        dead_edge = Edge(
            source_id=sensor.id,
            target_id=dead.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=0.01,
        )
        dead_edge.strength = 0.0
        dead_edge.last_active_step = 0
        graph.add_edge(dead_edge)

        em = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        _seed_episodic(em)
        # Step is far enough past grace + inactivity thresholds.
        step = config.pruning_grace_period + config.inactivity_threshold + 50
        result = consolidation_cycle(
            graph,
            em,
            current_step=step,
            config=config,
            experience_unpacker=_unpacker,
        )
        assert result.edges_pruned >= 1
        # Dead associator should also be removed (became orphan after edge removal).
        assert dead.id not in graph.nodes

    def test_returns_result_with_fields(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, _, _ = _build_graph(config)
        em = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        _seed_episodic(em)
        result = consolidation_cycle(
            graph,
            em,
            current_step=100,
            config=config,
            experience_unpacker=_unpacker,
        )
        assert isinstance(result, ConsolidationResult)
        # New fields exist.
        assert hasattr(result, "edges_pruned")
        assert hasattr(result, "nodes_pruned")
        assert hasattr(result, "chains_compressed")
        assert hasattr(result, "neurogenesis_nodes")

    def test_calls_decay_old_entries(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, _, _ = _build_graph(config)
        em = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        # Three "old and over-replayed" entries that should get decayed.
        for _ in range(3):
            em.encode(torch.randn(em.value_dim), prediction_error=1.0, current_step=0)
        with torch.no_grad():
            em._replay_count()[:3] = 10
        # Three "sampleable" entries with moderate age so sample_for_replay
        # returns a non-empty batch (and the cycle proceeds to decay).
        current_step = 200_000
        for offset in (1000, 950, 1050):
            em.encode(
                torch.randn(em.value_dim),
                prediction_error=5.0,
                current_step=current_step - offset,
            )
        before = em.num_valid
        _ = consolidation_cycle(
            graph,
            em,
            current_step=current_step,
            config=config,
            experience_unpacker=_unpacker,
        )
        # The three over-replayed ancient entries should now be invalid.
        assert em.num_valid < before
