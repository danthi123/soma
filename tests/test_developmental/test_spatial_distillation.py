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
        shape (sensor_dim**2, position_dim)."""
        cfg = _spatial_config()
        pred = PredictiveSOMA(config=cfg)
        assert hasattr(pred, "_position_projector")
        proj = pred._position_projector
        assert isinstance(proj, torch.Tensor)
        # Not a Parameter — frozen
        assert not isinstance(proj, torch.nn.Parameter)
        # Shape check
        expected_shape = (cfg.sensor_output_dim ** 2, cfg.position_dim)
        assert proj.shape == expected_shape

    def test_position_projector_absent_when_not_spatial(self) -> None:
        """Frozen position_mode — no need for projector."""
        cfg = SOMAConfig.developmental(
            initial_associator_count=4, max_nodes=16,
        )
        pred = PredictiveSOMA(config=cfg)
        # Attribute may or may not exist but must be None / absent when inactive
        proj = getattr(pred, "_position_projector", None)
        assert proj is None
