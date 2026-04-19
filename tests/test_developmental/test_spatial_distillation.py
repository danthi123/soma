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
        expected_shape = (cfg.sensor_output_dim ** 2, cfg.position_dim)
        assert proj.shape == expected_shape
        # Must be a registered buffer (follows the Module across
        # .to(device) / state_dict() calls)
        buffer_names = [name for name, _ in pred.named_buffers()]
        assert "_position_projector" in buffer_names

    def test_position_projector_absent_when_not_spatial(self) -> None:
        """Frozen position_mode — no need for projector."""
        cfg = SOMAConfig.developmental(
            initial_associator_count=4, max_nodes=16,
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
