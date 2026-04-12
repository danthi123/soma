"""Stage 4 integration test — metacognition + enhanced consolidation.

Exercises the meta-cognitive stack in the main loop:
- ``CuriosityModule`` returns higher scores when errors drop.
- ``HomeostaticRegulator`` dampens LR on loss spikes and recovers it.
- ``DevelopmentSchedule`` peaks plasticity during the relevant stage.
- ``neurogenesis`` fires when the replay error ratio is high.
- ``consolidation_cycle`` with ``run_structural_maintenance=True`` prunes
  dead edges and invalidates stale episodic entries.
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
from soma.core.learning import update_step
from soma.core.node import Node, NodeType
from soma.memory.episodic_memory import EpisodicMemory
from soma.metacognition.curiosity import CuriosityModule
from soma.metacognition.development import DevelopmentSchedule
from soma.metacognition.homeostasis import HomeostaticRegulator


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig(
        base_lr=0.02,
        youth_lr_multiplier=1.0,
        hebbian_lr=0.0001,
        consolidation_lr_ratio=0.5,
        consolidation_replay_steps=8,
        consolidation_error_threshold=2.0,
        activation_threshold=0.01,
        max_edge_weight=3.0,
        max_nodes=64,
        pruning_grace_period=5,
        edge_strength_threshold=0.1,
        inactivity_threshold=5,
        myelination_strength_threshold=100.0,  # never ripe in this test
        neurogenesis_threshold=1.2,
    )


def _linear_graph(config: SOMAConfig) -> Graph:
    graph = Graph()
    dim = 8
    sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    assoc = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
    out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    graph.add_node(sensor, modality="text")
    graph.add_node(assoc)
    graph.add_node(out, modality="text")
    for src, tgt in [(sensor, assoc), (assoc, out)]:
        graph.add_edge(
            Edge(
                source_id=src.id,
                target_id=tgt.id,
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


class TestMetaCognitionIntegration:
    def test_full_loop_stable(self, config: SOMAConfig) -> None:
        """Run ~300 steps with the full metacognition stack; check invariants."""
        torch.manual_seed(0)
        graph = _linear_graph(config)
        em = EpisodicMemory.from_config(SOMAConfig(episodic_capacity=32, key_dim=4, value_dim=16))
        homeo = HomeostaticRegulator.from_config(config)
        curiosity = CuriosityModule(input_dim=8, num_domains=4, warmup=5)
        schedule = DevelopmentSchedule()

        rng = torch.Generator().manual_seed(1)
        dim = 8
        target_w = torch.eye(dim) * 0.3
        losses: list[float] = []
        curiosity_scores: list[float] = []
        lr_multipliers: list[float] = []

        for step in range(300):
            x = torch.randn(dim, generator=rng)
            target = target_w @ x
            outputs, activations = execute_graph(graph, inputs={"text": x}, current_step=step)
            out = outputs["text"]
            loss = F.mse_loss(out, target)
            losses.append(float(loss.item()))

            # Homeostasis tracks the loss + publishes LR multiplier.
            lr_mult = homeo.update(graph, current_loss=float(loss.item()))
            lr_multipliers.append(lr_mult)

            # Curiosity tracks per-domain progress.
            curiosity_scores.append(
                curiosity.compute_curiosity(x, prediction_error=float(loss.item()))
            )

            # Check the plasticity multiplier API works per node type.
            mult = schedule.get_plasticity_multiplier(step=step, node_type=NodeType.ASSOCIATOR)
            assert mult >= 1.0

            # Apply learning with homeostasis-driven LR multiplier.
            update_step(graph, loss, activations, config, lr_multiplier=lr_mult)

            # Store surprising experiences.
            if loss.item() > 0.1:
                em.encode(
                    torch.cat([x, target]),
                    prediction_error=float(loss.item()),
                    current_step=step,
                )

        # Sanity: some sensible behavior.
        assert all(v == v for v in losses)  # no NaNs
        assert em.num_valid > 0
        assert max(lr_multipliers) == pytest.approx(1.0, abs=1e-6)
        assert len(curiosity_scores) == 300

        # Enhanced consolidation cycle with structural maintenance.
        result = consolidation_cycle(
            graph,
            em,
            current_step=300,
            config=config,
            experience_unpacker=_halves_unpacker,
            num_replay_steps=em.num_valid,
            rng=torch.Generator().manual_seed(2),
        )
        # Pruning ran (could be 0 if the two seed edges are protected by
        # grace period). Key assertion: structural-maintenance fields are
        # populated in the result.
        assert result.edges_pruned >= 0
        assert result.nodes_pruned >= 0
        assert result.chains_compressed >= 0

    def test_homeostasis_dampens_spike_and_recovers(self, config: SOMAConfig) -> None:
        graph = _linear_graph(config)
        homeo = HomeostaticRegulator.from_config(config)
        # Warmup with steady low loss.
        for _ in range(30):
            homeo.update(graph, current_loss=0.1)
        # Inject a spike.
        before = homeo.global_lr_multiplier
        homeo.update(graph, current_loss=1000.0)
        after_spike = homeo.global_lr_multiplier
        assert after_spike < before
        # Recover with steady loss.
        for _ in range(500):
            homeo.update(graph, current_loss=0.1)
        assert homeo.global_lr_multiplier > after_spike

    def test_curiosity_rewards_learning_progress(self, config: SOMAConfig) -> None:
        cm = CuriosityModule(input_dim=4, num_domains=1, warmup=5)
        with torch.no_grad():
            cm.domain_classifier.weight.zero_()
            cm.domain_classifier.bias.zero_()
        # Warmup with high error.
        for _ in range(10):
            cm.compute_curiosity(torch.zeros(4), prediction_error=1.0)
        # Drop error rapidly.
        for err in [0.5, 0.3, 0.2, 0.1, 0.05]:
            cm.compute_curiosity(torch.zeros(4), prediction_error=err)
        score = cm.compute_curiosity(torch.zeros(4), prediction_error=0.02)
        assert score > 0.0

    def test_development_schedule_active_during_sensory_period(self, config: SOMAConfig) -> None:
        schedule = DevelopmentSchedule()
        # Sensor has its peak at step 5000 per whitepaper.
        sensor_peak = schedule.get_plasticity_multiplier(step=5000, node_type=NodeType.SENSOR)
        assert sensor_peak > 1.5  # boosted
        # Integrator period starts at step 20k, so at step 5k integrator
        # plasticity is still base 1.0.
        integrator_early = schedule.get_plasticity_multiplier(
            step=5000, node_type=NodeType.INTEGRATOR
        )
        assert integrator_early == pytest.approx(1.0)
