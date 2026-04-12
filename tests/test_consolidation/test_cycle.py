"""Tests for ``soma.consolidation.cycle.consolidation_cycle``."""

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
    # Consolidation uses base_lr * consolidation_lr_ratio = 0.05 * 0.5 = 0.025.
    return SOMAConfig(
        base_lr=0.05,
        consolidation_lr_ratio=0.5,
        consolidation_replay_steps=10,
    )


def _build_graph(config: SOMAConfig) -> tuple[Graph, Node, Node, Node]:
    """SENSOR(text) -> ASSOC -> OUTPUT(text), 4-dim."""
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
    return graph, sensor, assoc, out


def _halves_unpacker(
    experience: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """First half -> sensor input; second half -> target. Shared 'text' modality."""
    half = experience.shape[-1] // 2
    return {"text": experience[:half]}, {"text": experience[half:]}


def _seed_episodic(episodic: EpisodicMemory, n: int = 8, *, seed: int = 0) -> None:
    torch.manual_seed(seed)
    for t in range(n):
        exp = torch.randn(episodic.value_dim)
        episodic.encode(experience=exp, prediction_error=1.0, current_step=t * 10)


class TestBehavior:
    def test_returns_empty_when_no_memories(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, *_ = _build_graph(config)
        episodic = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        result = consolidation_cycle(
            graph,
            episodic,
            current_step=100,
            config=config,
            experience_unpacker=_halves_unpacker,
        )
        assert isinstance(result, ConsolidationResult)
        assert result.num_replayed == 0
        assert result.mean_loss == 0.0

    def test_replays_all_available(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, *_ = _build_graph(config)
        episodic = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        _seed_episodic(episodic, n=5)
        result = consolidation_cycle(
            graph,
            episodic,
            current_step=1000,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=5,
        )
        assert result.num_replayed == 5
        assert len(result.replay_losses) == 5

    def test_updates_graph_parameters(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, _, assoc, _ = _build_graph(config)
        episodic = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        _seed_episodic(episodic, n=6)
        before = assoc.linear1.weight.detach().clone()
        consolidation_cycle(
            graph,
            episodic,
            current_step=1000,
            config=config,
            experience_unpacker=_halves_unpacker,
        )
        assert not torch.allclose(before, assoc.linear1.weight)

    def test_replay_does_not_bump_edge_last_active(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, sensor, assoc, _ = _build_graph(config)
        edge = graph.get_edge(sensor.id, assoc.id)
        edge.mark_active(step=5)  # set to known value
        episodic = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        _seed_episodic(episodic, n=3)
        consolidation_cycle(
            graph,
            episodic,
            current_step=1000,
            config=config,
            experience_unpacker=_halves_unpacker,
        )
        # record_edge_activity=False inside the cycle, so last_active_step
        # should NOT be bumped to 1000.
        assert edge.last_active_step == 5

    def test_mature_node_lr_higher_than_young(self, config: SOMAConfig) -> None:
        """Consolidation should adjust mature nodes more than young ones."""
        torch.manual_seed(0)
        graph_mature, _, mature_assoc, _ = _build_graph(config)
        graph_young, _, young_assoc, _ = _build_graph(config)
        # Match weights so the only difference is maturity.
        young_assoc.load_state_dict(mature_assoc.state_dict())
        mature_assoc.maturity = 1.0
        young_assoc.maturity = 0.0

        # Build two independent episodic memories with identical contents.
        torch.manual_seed(1)
        episodic_mature = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        experiences = [torch.randn(8) for _ in range(5)]
        for t, exp in enumerate(experiences):
            episodic_mature.encode(exp, 1.0, t * 10)
        episodic_young = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        for t, exp in enumerate(experiences):
            episodic_young.encode(exp, 1.0, t * 10)

        before_mature = mature_assoc.linear1.weight.detach().clone()
        before_young = young_assoc.linear1.weight.detach().clone()

        rng1 = torch.Generator().manual_seed(42)
        rng2 = torch.Generator().manual_seed(42)
        consolidation_cycle(
            graph_mature,
            episodic_mature,
            current_step=1000,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=5,
            rng=rng1,
        )
        consolidation_cycle(
            graph_young,
            episodic_young,
            current_step=1000,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=5,
            rng=rng2,
        )

        mature_delta = float((mature_assoc.linear1.weight - before_mature).abs().sum().item())
        young_delta = float((young_assoc.linear1.weight - before_young).abs().sum().item())
        # Mature gets LR = lr * (0.5 + 0.5 * 1.0) = lr; young gets 0.5 * lr.
        assert mature_delta > young_delta

    def test_rejects_negative_step(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, *_ = _build_graph(config)
        episodic = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        with pytest.raises(ValueError, match="current_step"):
            consolidation_cycle(
                graph,
                episodic,
                current_step=-1,
                config=config,
                experience_unpacker=_halves_unpacker,
            )

    def test_rejects_non_positive_num_replay_steps(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        graph, *_ = _build_graph(config)
        episodic = EpisodicMemory(capacity=8, key_dim=4, value_dim=8)
        with pytest.raises(ValueError, match="num_replay_steps"):
            consolidation_cycle(
                graph,
                episodic,
                current_step=0,
                config=config,
                experience_unpacker=_halves_unpacker,
                num_replay_steps=0,
            )
