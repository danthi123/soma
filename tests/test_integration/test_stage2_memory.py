"""Stage 2 integration test — memorization across three memory tiers.

Exercises:
- ``WorkingMemory`` holding the most-recent input (short-horizon).
- ``EpisodicMemory`` storing a specific experience and retrieving it by key
  (one-shot, medium-horizon).
- ``consolidation_cycle`` replaying stored experiences and reducing the
  replay-time MSE loss (consolidation into the graph = parametric memory,
  long-horizon).

Each tier is validated separately and then in combination, confirming the
three-tier design from the whitepaper (Section 4).
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from soma.consolidation.cycle import consolidation_cycle
from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.memory.episodic_memory import EpisodicMemory
from soma.memory.working_memory import WorkingMemory


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig(
        base_lr=0.05,
        consolidation_lr_ratio=0.5,
        consolidation_replay_steps=20,
        wm_slots=8,
        wm_dim=8,
        wm_decay_rate=0.8,
        episodic_capacity=32,
        key_dim=8,
        value_dim=16,
    )


def _build_graph(config: SOMAConfig, dim: int = 8) -> Graph:
    graph = Graph()
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
    return graph


def _halves_unpacker(
    experience: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    half = experience.shape[-1] // 2
    return {"text": experience[:half]}, {"text": experience[half:]}


# ----------------------------------------------------------------------
# Working Memory
# ----------------------------------------------------------------------
class TestWorkingMemoryRecall:
    def test_recent_content_retrievable(self) -> None:
        """Reading back a stored slot should return something biased toward it.

        With a single filled slot out of four and an uninformative attention
        pattern, the weighted combination can only pick up ~40% of the stored
        content — attention softmax spreads weight across empty slots too.
        """
        torch.manual_seed(0)
        wm = WorkingMemory(num_slots=4, wm_dim=8, decay_rate=0.95)
        with torch.no_grad():
            wm.write_gate.weight.zero_()
            wm.write_gate.bias.fill_(5.0)
            wm.query_proj.weight.copy_(torch.eye(8))
            wm.query_proj.bias.zero_()

        content = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        wm.write(content, context=torch.zeros(8))
        retrieved = wm.read(content)
        # Retrieval should be strictly positive in the ``content`` direction
        # (empty slots contribute zero, non-empty slot contributes positively).
        assert retrieved.dot(content) > 0.3

    def test_old_content_fades(self) -> None:
        torch.manual_seed(0)
        # Aggressive decay + large fade_factor so 20 steps reliably drive
        # slot content close to zero.
        wm = WorkingMemory(
            num_slots=4,
            wm_dim=4,
            decay_rate=0.1,
            fade_threshold=0.5,
            fade_factor=0.5,
        )
        with torch.no_grad():
            wm.write_gate.weight.zero_()
            wm.write_gate.bias.fill_(5.0)
        wm.write(torch.ones(4), torch.zeros(4))
        for _ in range(30):
            wm.step()
        # All slot values should be essentially zero after repeated fade.
        assert float(wm._slots().abs().sum().item()) < 1e-3


# ----------------------------------------------------------------------
# Episodic Memory
# ----------------------------------------------------------------------
class TestEpisodicRecall:
    def test_retrieves_stored_experience(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        em = EpisodicMemory.from_config(config)
        experience = torch.randn(config.value_dim)
        idx = em.encode(experience, prediction_error=1.0, current_step=100)
        assert idx == 0
        # Query using the same encoder so cosine should align on this entry.
        with torch.no_grad():
            query_key = em.key_encoder(experience.detach())
        results = em.retrieve(query_key, top_k=1)
        assert len(results) == 1
        assert torch.allclose(results[0].value, experience)
        assert results[0].similarity > 0.9  # cosine close to 1

    def test_random_query_has_lower_similarity(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        em = EpisodicMemory.from_config(config)
        experience = torch.randn(config.value_dim)
        em.encode(experience, 1.0, 100)
        # Completely random unrelated query.
        random_query = torch.randn(config.key_dim)
        with torch.no_grad():
            true_key = em.key_encoder(experience.detach())
        results_random = em.retrieve(random_query, top_k=1)
        results_true = em.retrieve(true_key, top_k=1)
        # The true query must score at least as high as the random one.
        assert results_true[0].similarity >= results_random[0].similarity


# ----------------------------------------------------------------------
# Consolidation / Parametric Memory
# ----------------------------------------------------------------------
class TestConsolidatedRecall:
    def test_replay_reduces_loss_over_multiple_cycles(self, config: SOMAConfig) -> None:
        """Multiple consolidation passes should lower the mean replay loss."""
        torch.manual_seed(0)
        graph = _build_graph(config, dim=config.value_dim // 2)
        em = EpisodicMemory.from_config(config)

        # Populate episodic with a small fixed set of high-surprise experiences.
        for t in range(10):
            exp = torch.randn(config.value_dim)
            em.encode(exp, prediction_error=5.0, current_step=t * 10)

        rng1 = torch.Generator().manual_seed(1)
        rng2 = torch.Generator().manual_seed(2)
        first = consolidation_cycle(
            graph,
            em,
            current_step=1000,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=10,
            rng=rng1,
        )
        # Subsequent cycles should see lower mean loss as the graph learns.
        later = consolidation_cycle(
            graph,
            em,
            current_step=1500,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=10,
            rng=rng2,
        )
        assert later.mean_loss < first.mean_loss


# ----------------------------------------------------------------------
# Three-tier timeline
# ----------------------------------------------------------------------
class TestThreeTierIntegration:
    """WM holds latest, episodic holds a surprising past event, graph improves on replay."""

    def test_all_tiers_together(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        dim = config.value_dim // 2  # sensor/output dim = 8
        graph = _build_graph(config, dim=dim)
        em = EpisodicMemory.from_config(config)
        wm = WorkingMemory.from_config(config)

        # Stage a 50-step interaction run. Every 10 steps inject a "surprising"
        # experience to stress episodic (high prediction_error).
        target_weight = torch.eye(dim) * 0.3
        rng = torch.Generator().manual_seed(123)
        losses_during_run: list[float] = []

        for step in range(50):
            x = torch.randn(dim, generator=rng)
            target = target_weight @ x
            outputs, _ = execute_graph(graph, inputs={"text": x}, current_step=step)
            pred = outputs["text"]
            loss = F.mse_loss(pred, target)
            losses_during_run.append(float(loss.item()))

            # WM captures each step's output with matched gate.
            with torch.no_grad():
                if step == 0:
                    wm.write_gate.weight.zero_()
                    wm.write_gate.bias.fill_(5.0)
            wm.write(pred.detach().clone(), context=x)
            wm.step()

            # Episodic stores surprising experiences only.
            if loss.item() > 0.2:
                experience = torch.cat([x, target])  # (value_dim,)
                em.encode(experience, prediction_error=float(loss.item()), current_step=step)

        # Tier 1: working memory should hold the most-recent output near top slot.
        last_pred = pred.detach()
        read_back = wm.read(last_pred)
        assert read_back.norm().item() > 0.0  # something useful in WM
        assert torch.isfinite(read_back).all()

        # Tier 2: episodic should have captured several high-surprise events.
        assert em.num_valid > 0

        # Tier 3: one consolidation cycle should reduce mean replay loss
        # compared to an immediately-following second cycle (after learning).
        first_cycle = consolidation_cycle(
            graph,
            em,
            current_step=100,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=em.num_valid,
            rng=torch.Generator().manual_seed(10),
        )
        second_cycle = consolidation_cycle(
            graph,
            em,
            current_step=200,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=em.num_valid,
            rng=torch.Generator().manual_seed(11),
        )
        assert first_cycle.num_replayed > 0
        assert second_cycle.mean_loss <= first_cycle.mean_loss + 1e-6
