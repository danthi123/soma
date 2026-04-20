"""Tests for Direction 4b spatial distillation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def _spatial_config(**kw):
    """Minimal config with spatial distillation enabled."""
    return SOMAConfig.developmental(
        initial_associator_count=4,
        initial_integrator_count=2,
        max_nodes=16,
        projection_mode="learnable",
        position_mode="learnable",
        projection_distillation_target="llm_spatial",
        **kw,
    )


class TestPositionProjector:
    def test_position_projector_registered_as_buffer(self) -> None:
        """position_projector is a fixed random matrix used to map
        flat W_i to position_dim. Should be a buffer (not parameter),
        shape (sensor_dim**2, position_dim), registered on the module
        so it follows .to(device) and state_dict() boundaries."""
        cfg = _spatial_config()
        pred = PredictiveSOMA(config=cfg)
        assert hasattr(pred, "_position_projector")
        proj = pred._position_projector
        assert isinstance(proj, torch.Tensor)
        assert not isinstance(proj, torch.nn.Parameter)
        expected_shape = (cfg.sensor_output_dim**2, cfg.position_dim)
        assert proj.shape == expected_shape
        # Must be a registered buffer (follows the Module across
        # .to(device) / state_dict() calls)
        buffer_names = [name for name, _ in pred.named_buffers()]
        assert "_position_projector" in buffer_names

    def test_position_projector_absent_when_not_spatial(self) -> None:
        """Frozen position_mode — no need for projector."""
        cfg = SOMAConfig.developmental(
            initial_associator_count=4,
            max_nodes=16,
        )
        pred = PredictiveSOMA(config=cfg)
        # Attribute may or may not exist but must be None / absent when inactive
        proj = getattr(pred, "_position_projector", None)
        assert proj is None


class TestLearnablePositions:
    def test_learnable_positions_are_parameters(self) -> None:
        """When position_mode=learnable, node.position is nn.Parameter."""
        cfg = _spatial_config()
        pred = PredictiveSOMA(config=cfg)
        from soma.core.node import NodeType

        any_associator = False
        for node in pred.soma.graph.all_nodes():
            if node.node_type == NodeType.ASSOCIATOR:
                any_associator = True
                assert isinstance(node.position, torch.nn.Parameter), (
                    f"Node {node.id} position is {type(node.position)}"
                )
        assert any_associator, "need at least one associator"

    def test_frozen_positions_stay_tensors(self) -> None:
        """Default mode — positions are plain Tensors."""
        cfg = SOMAConfig.developmental(initial_associator_count=4, max_nodes=16)
        pred = PredictiveSOMA(config=cfg)
        from soma.core.node import NodeType

        for node in pred.soma.graph.all_nodes():
            if node.node_type == NodeType.ASSOCIATOR:
                assert not isinstance(node.position, torch.nn.Parameter)

    def test_initial_position_norms_recorded(self) -> None:
        """For norm-preservation after optimizer step, we need to know
        each position's initial L2 norm."""
        cfg = _spatial_config()
        pred = PredictiveSOMA(config=cfg)
        from soma.core.node import NodeType

        for node in pred.soma.graph.all_nodes():
            if node.node_type == NodeType.ASSOCIATOR:
                recorded = pred._initial_position_norms.get(node.id)
                assert recorded is not None
                actual = node.position.norm().item()
                assert abs(recorded - actual) < 1e-6

    def test_learnable_positions_in_optimizer(self) -> None:
        """Positions must be included in the prediction optimizer when
        learnable; otherwise backward() gradient isn't applied."""
        cfg = _spatial_config()
        pred = PredictiveSOMA(config=cfg)
        opt_params = set()
        for g in pred._pred_optimizer.param_groups:
            for p in g["params"]:
                opt_params.add(id(p))
        from soma.core.node import NodeType

        for node in pred.soma.graph.all_nodes():
            if node.node_type == NodeType.ASSOCIATOR:
                assert id(node.position) in opt_params


class TestEnsurePosition:
    def test_ensure_position_no_op_when_not_learnable(self) -> None:
        """Frozen mode: _ensure_position is a no-op — adding a fresh node
        and calling _ensure_position must NOT wrap its position as a
        Parameter, must NOT record an initial norm, and must NOT touch
        the optimizer. Also must not crash on unknown node ids."""
        cfg = SOMAConfig.developmental(initial_associator_count=4, max_nodes=16)
        # Default developmental preset uses position_mode='frozen_random'
        assert cfg.position_mode == "frozen_random"
        pred = PredictiveSOMA(config=cfg)

        # Unknown node id -> silent no-op
        pred._ensure_position("nonexistent_node_id")

        # Real fresh node -> still no-op in frozen mode
        from soma.core.node import Node, NodeType

        new_node = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=cfg.associator_input_dim,
            hidden_dim=cfg.associator_hidden_dim,
            output_dim=cfg.associator_output_dim,
            creation_step=0,
            config=cfg,
            position=torch.randn(cfg.position_dim) * 0.1,
            device=pred.device,
        )
        pred.soma.graph.add_node(new_node)
        opt_params_before = {id(p) for g in pred._pred_optimizer.param_groups for p in g["params"]}
        norms_before = dict(pred._initial_position_norms)

        pred._ensure_position(new_node.id)

        assert not isinstance(new_node.position, torch.nn.Parameter)
        assert new_node.id not in pred._initial_position_norms
        assert pred._initial_position_norms == norms_before
        opt_params_after = {id(p) for g in pred._pred_optimizer.param_groups for p in g["params"]}
        assert opt_params_after == opt_params_before

    def test_ensure_position_creates_parameter_on_new_node(self) -> None:
        """When neurogenesis creates a new node mid-run, _ensure_position
        wraps its position as nn.Parameter, records initial norm, adds
        to optimizer."""
        cfg = _spatial_config()
        pred = PredictiveSOMA(config=cfg)
        from soma.core.node import Node, NodeType

        new_node = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=cfg.associator_input_dim,
            hidden_dim=cfg.associator_hidden_dim,
            output_dim=cfg.associator_output_dim,
            creation_step=0,
            config=cfg,
            position=torch.randn(cfg.position_dim) * 0.1,
            device=pred.device,
        )
        pred.soma.graph.add_node(new_node)
        pred._ensure_position(new_node.id)

        assert isinstance(new_node.position, torch.nn.Parameter)
        assert new_node.id in pred._initial_position_norms
        opt_params = set()
        for g in pred._pred_optimizer.param_groups:
            for p in g["params"]:
                opt_params.add(id(p))
        assert id(new_node.position) in opt_params


class TestCompetitiveDistillation:
    def test_only_top_k_winners_receive_distill_gradient(self) -> None:
        """With K=2 winners and 4+ associators, `_last_distill_winners`
        must have length <= K (exactly K when K <= available) and must
        be a subset of the active associator ids. Verifies the gating
        actually gates, not just that the path fires."""
        # Helper already sets initial_associator_count=4.
        cfg = _spatial_config(projection_distillation_winners=2)
        pred = PredictiveSOMA(config=cfg)

        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        from soma.core.node import NodeType

        associators = [n for n in pred.soma.graph.all_nodes() if n.node_type == NodeType.ASSOCIATOR]
        assert len(associators) >= 4, "need at least 4 associators to distinguish top-2 gating"
        assoc_ids = {n.id for n in associators}

        x = torch.randn(cfg.sensor_output_dim)
        # Prime _last_summary
        pred.process_input(x, source_text="step 1")
        # Second step: distill path fires
        pred.process_input(x, source_text="step 2")

        # Teacher consulted on second step (after summary primed).
        assert teacher.embed.call_count >= 1, (
            "teacher.embed should be called when target=llm_spatial"
        )

        # Gating: exactly 2 winners recorded, and they must be real
        # associator ids. If gating were broken and distill hit all
        # nodes, the list would have 4 entries.
        winners = pred._last_distill_winners
        assert len(winners) == 2, (
            f"expected exactly K=2 winners, got {len(winners)}: {winners}"
        )
        assert set(winners) <= assoc_ids
        # No duplicates in winners.
        assert len(set(winners)) == len(winners)

        # Projections remain finite (no NaN/Inf from optimizer step).
        for proj in pred._input_projections.values():
            assert torch.isfinite(proj).all()

    def test_k_zero_means_all_nodes_receive_distill(self) -> None:
        """Backward-compat: K=0 means no competitive gate (fall back to
        Direction 4a mean-target behavior). Must fire under
        target=llm_spatial AND leave `_last_distill_winners` untouched
        (empty), proving the K>0 selection branch was NOT entered."""
        cfg = _spatial_config(projection_distillation_winners=0)
        pred = PredictiveSOMA(config=cfg)

        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        pred.process_input(x, source_text="t1")
        pred.process_input(x, source_text="t2")

        # K=0 path must still engage the teacher (mean-target fallback).
        assert teacher.embed.call_count >= 1, (
            "K=0 should fall back to mean-target and still consult teacher"
        )
        # Gating: K=0 must NOT enter the winner-selection branch, so
        # the debug hook stays empty.
        assert pred._last_distill_winners == [], (
            f"K=0 should skip winner selection, got {pred._last_distill_winners}"
        )
        for proj in pred._input_projections.values():
            assert torch.isfinite(proj).all()


class TestPositionCouplingLoss:
    def test_position_loss_fires_when_spatial_active(self) -> None:
        """After a step, positions should have moved toward PCA(W_i)."""
        cfg = _spatial_config(position_coupling_weight=10.0)  # strong coupling
        pred = PredictiveSOMA(config=cfg)

        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        from soma.core.node import NodeType
        initial_positions = {
            n.id: n.position.detach().clone()
            for n in pred.soma.graph.all_nodes()
            if n.node_type == NodeType.ASSOCIATOR
        }

        x = torch.randn(cfg.sensor_output_dim)
        pred.process_input(x, source_text="t1")
        pred.process_input(x, source_text="t2")  # triggers backward

        # At least one position should have moved
        moved_any = False
        for nid, p0 in initial_positions.items():
            if nid not in pred.soma.graph.nodes:
                continue
            p1 = pred.soma.graph.nodes[nid].position
            if (p1 - p0).abs().max().item() > 1e-6:
                moved_any = True
                break
        assert moved_any, "positions should have moved under spatial distill"

    def test_position_loss_zero_when_weight_zero(self) -> None:
        """β=0 disables the position loss contribution entirely. No
        other loss touches node.position, so positions must not move
        across steps (beyond machine epsilon) — proves the position
        path is actually gated by β, not always firing."""
        cfg = _spatial_config(position_coupling_weight=0.0)
        pred = PredictiveSOMA(config=cfg)

        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        from soma.core.node import NodeType

        initial_positions = {
            n.id: n.position.detach().clone()
            for n in pred.soma.graph.all_nodes()
            if n.node_type == NodeType.ASSOCIATOR
        }

        x = torch.randn(cfg.sensor_output_dim)
        pred.process_input(x, source_text="t1")
        pred.process_input(x, source_text="t2")

        # Every initial associator must still be at its starting position.
        for nid, p0 in initial_positions.items():
            if nid not in pred.soma.graph.nodes:
                continue
            p1 = pred.soma.graph.nodes[nid].position
            assert torch.isfinite(p1).all()
            assert torch.allclose(p1, p0, atol=1e-6), (
                f"position for {nid} moved with β=0: "
                f"max delta {(p1 - p0).abs().max().item()}"
            )


class TestNormPreservation:
    def test_position_norms_preserved_across_steps(self) -> None:
        """After each optimizer step, each position should be rescaled
        to its initial L2 norm (keeps 0.5 locality cutoff calibrated)."""
        cfg = _spatial_config(position_coupling_weight=10.0)
        pred = PredictiveSOMA(config=cfg)

        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        for step in range(5):
            pred.process_input(x, source_text=f"t{step}")

        from soma.core.node import NodeType
        for node in pred.soma.graph.all_nodes():
            if node.node_type != NodeType.ASSOCIATOR:
                continue
            expected = pred._initial_position_norms[node.id]
            actual = node.position.norm().item()
            # Ratio-rescale residual should be near float32 round-off
            # (<<1e-5). Loose tolerance would hide a silent no-op
            # (e.g. rescale branch guarded wrong and never firing).
            assert abs(actual - expected) < 1e-5, (
                f"Node {node.id} position norm drifted: "
                f"init={expected}, now={actual}"
            )
