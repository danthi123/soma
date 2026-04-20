"""Tests for Direction 4a distillation loss in PredictiveSOMA."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def _tiny_config(**kw):
    return SOMAConfig.developmental(
        initial_associator_count=4,
        initial_integrator_count=2,
        max_nodes=16,
        projection_mode="learnable",
        **kw,
    )


class TestDistillationIntegration:
    def test_distillation_off_by_default(self) -> None:
        """No teacher attachment, no distillation behavior."""
        cfg = _tiny_config()
        pred = PredictiveSOMA(config=cfg)
        # No teacher attached → attribute should be None or not present
        teacher = getattr(pred, "_teacher", None)
        assert teacher is None

    def test_attach_teacher_sets_field(self) -> None:
        cfg = _tiny_config(projection_distillation_target="llm_embedding")
        pred = PredictiveSOMA(config=cfg)

        fake_teacher = MagicMock()
        fake_teacher.name = "fake"
        fake_teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(fake_teacher)
        assert pred._teacher is fake_teacher

    def test_process_input_calls_teacher_when_distillation_on(self) -> None:
        cfg = _tiny_config(projection_distillation_target="llm_embedding")
        pred = PredictiveSOMA(config=cfg)
        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        # Two calls: first primes _last_summary, second triggers prediction + distill
        pred.process_input(x, source_text="hello")
        pred.process_input(x, source_text="world")

        # Teacher should be called for at least the second step (with source_text)
        assert teacher.embed.called

    def test_process_input_skips_teacher_when_distillation_off(self) -> None:
        cfg = _tiny_config()  # default = "none"
        pred = PredictiveSOMA(config=cfg)
        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        pred.process_input(x, source_text="hello")
        pred.process_input(x, source_text="world")
        assert teacher.embed.call_count == 0

    def test_distillation_loss_runs_without_error_on_trivial_input(self) -> None:
        cfg = _tiny_config(
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=1.0,
        )
        pred = PredictiveSOMA(config=cfg)
        teacher = MagicMock()
        teacher.name = "fake"
        # Return same embedding every time → distill loss should be
        # consistent across calls even if non-zero
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        for step in range(5):
            pred.process_input(x, source_text=f"step {step}")

        # Projections should still be finite
        for p in pred._input_projections.values():
            assert torch.isfinite(p).all()

    def test_zero_weight_disables_distill_gradient_contribution(self) -> None:
        """alpha=0 means: distill loss is present in the graph but its
        gradient contribution is zero. prediction_head weights should
        update identically to the no-distill baseline. (Teacher is still
        called, which is fine — it's useful for ablation/logging.)

        Node IDs are UUIDs so projection dicts aren't comparable by key
        across instances, but prediction_head lives at a fixed attr path
        and its weights are deterministically initialized via
        torch.manual_seed, making it a reliable comparison point.
        """
        cfg_on = _tiny_config(
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=0.0,
            seed=7,
        )
        cfg_off = _tiny_config(seed=7)

        torch.manual_seed(0)
        pred_on = PredictiveSOMA(config=cfg_on)
        torch.manual_seed(0)
        pred_off = PredictiveSOMA(config=cfg_off)

        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred_on.attach_teacher(teacher)

        torch.manual_seed(0)
        x = torch.randn(cfg_on.sensor_output_dim)
        for step in range(3):
            pred_on.process_input(x, source_text=f"t{step}")
            pred_off.process_input(x, source_text=f"t{step}")

        # prediction_head.weight is a stable named parameter and its
        # gradient updates depend only on pred_loss (plus alpha*distill
        # which is zero here). Difference must be tiny.
        diff = (
            pred_on.prediction_head.weight - pred_off.prediction_head.weight
        ).abs().max().item()
        assert diff < 1e-4, f"prediction_head weights diverged: {diff}"

        # Teacher is called on steps 2+ (step 1 has _last_summary=None
        # so the whole prediction/distill block is skipped). 3 steps →
        # 2 teacher calls. alpha=0 still exercises the teacher path for
        # ablation/logging purposes.
        assert teacher.embed.call_count == 2
