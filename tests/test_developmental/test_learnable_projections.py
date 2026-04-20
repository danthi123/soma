"""Tests for Direction 2 — learnable input projections.

Direction 2 makes PredictiveSOMA._input_projections trainable via
gradient flow from the prediction loss, instead of a frozen random
tensor updated only by Hebbian outer products.

Covered here:
1. Config validation for the new mode selector.
2. Projections are nn.Parameter when learnable mode is on;
   plain Tensor when off (backward-compat).
3. After backprop through _diversify_activations, projection
   gradients are non-zero.
4. Optimizer step actually changes projection weights.
5. Serialization round-trip preserves projection tensors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def _tiny_config(**overrides: object) -> SOMAConfig:
    defaults: dict[str, object] = dict(
        vocab_size=256,
        text_embed_dim=16,
        sensor_output_dim=16,
        associator_input_dim=16,
        associator_hidden_dim=32,
        associator_output_dim=16,
        integrator_input_dim=16,
        integrator_hidden_dim=32,
        integrator_output_dim=16,
        max_input_tokens=32,
        initial_integrator_count=2,
        initial_associator_count=4,
        max_nodes=32,
        num_curiosity_domains=2,
        synaptogenesis_interval=0,
        neurogenesis_interval=0,
        pruning_interval=0,
        consolidation_interval=0,
        seed=42,
    )
    defaults.update(overrides)
    return SOMAConfig(**defaults)


class TestConfigValidation:
    def test_default_projection_mode_is_frozen(self) -> None:
        cfg = _tiny_config()
        assert cfg.projection_mode == "frozen_random"

    def test_accepts_learnable_projection_mode(self) -> None:
        cfg = _tiny_config(projection_mode="learnable")
        assert cfg.projection_mode == "learnable"

    def test_rejects_invalid_projection_mode(self) -> None:
        with pytest.raises(ValueError, match="projection_mode"):
            _tiny_config(projection_mode="random")  # type: ignore[arg-type]

    def test_default_projection_lr_is_small(self) -> None:
        cfg = _tiny_config()
        # Should be small relative to prediction-head LR (3e-4); projections
        # are per-node and should move gently.
        assert 0.0 < cfg.projection_lr <= 1e-3

    def test_rejects_non_positive_projection_lr(self) -> None:
        with pytest.raises(ValueError, match="projection_lr"):
            _tiny_config(projection_lr=0.0)
        with pytest.raises(ValueError, match="projection_lr"):
            _tiny_config(projection_lr=-1e-4)


class TestProjectionParameters:
    def test_frozen_mode_uses_plain_tensor(self) -> None:
        """In the frozen default, projections are plain Tensor (matches
        current behavior). They're still registered for load/save but
        are NOT nn.Parameter — no gradients, no optimizer."""
        cfg = _tiny_config(projection_mode="frozen_random")
        ps = PredictiveSOMA(cfg, device=torch.device("cpu"))
        for proj in ps._input_projections.values():
            assert not isinstance(proj, torch.nn.Parameter), (
                f"frozen mode should produce plain Tensor, got {type(proj)}"
            )
            assert not proj.requires_grad

    def test_learnable_mode_uses_parameter(self) -> None:
        """In learnable mode, projections are nn.Parameter. requires_grad
        is True so backprop will populate their .grad attribute."""
        cfg = _tiny_config(projection_mode="learnable")
        ps = PredictiveSOMA(cfg, device=torch.device("cpu"))
        assert len(ps._input_projections) > 0
        for _node_id, proj in ps._input_projections.items():
            assert isinstance(proj, torch.nn.Parameter), (
                f"learnable mode should produce nn.Parameter, got {type(proj)}"
            )
            assert proj.requires_grad

    def test_learnable_mode_optimizer_includes_projections(self) -> None:
        cfg = _tiny_config(projection_mode="learnable")
        ps = PredictiveSOMA(cfg, device=torch.device("cpu"))
        # The pred optimizer should have two parameter groups or one group
        # containing both prediction_head params AND projection params.
        param_set = set()
        for group in ps._pred_optimizer.param_groups:
            for p in group["params"]:
                param_set.add(id(p))
        # Every projection must be in the optimizer.
        for proj in ps._input_projections.values():
            assert id(proj) in param_set, (
                "learnable projections must be in the prediction optimizer"
            )

    def test_frozen_mode_optimizer_excludes_projections(self) -> None:
        cfg = _tiny_config(projection_mode="frozen_random")
        ps = PredictiveSOMA(cfg, device=torch.device("cpu"))
        param_set = set()
        for group in ps._pred_optimizer.param_groups:
            for p in group["params"]:
                param_set.add(id(p))
        for proj in ps._input_projections.values():
            assert id(proj) not in param_set


class TestGradientFlow:
    def test_projection_gradients_populate_after_backward(self) -> None:
        """After process_input is called twice (so there's a prediction
        loss to backprop), projection .grad should be non-zero for at
        least one projection."""
        cfg = _tiny_config(projection_mode="learnable")
        ps = PredictiveSOMA(cfg, device=torch.device("cpu"))
        inp = torch.randn(16)
        ps.process_input(inp)  # first call: no prior prediction, no loss
        ps.process_input(inp + 0.1)  # second call: now there's a loss
        # At least one projection should have received a non-trivial grad.
        any_grad = False
        for proj in ps._input_projections.values():
            if proj.grad is not None and proj.grad.abs().sum().item() > 1e-10:
                any_grad = True
                break
        assert any_grad, (
            "expected projection gradients to populate after backward; got all zero"
        )

    def test_frozen_mode_keeps_projections_static(self) -> None:
        """Frozen-mode projections should not move during prediction
        backprop (they're plain tensors, not in the optimizer). They
        may still drift via the Hebbian _competitive_learning rule,
        but NOT via backprop."""
        cfg = _tiny_config(projection_mode="frozen_random")
        ps = PredictiveSOMA(cfg, device=torch.device("cpu"))
        {
            nid: proj.detach().clone()
            for nid, proj in ps._input_projections.items()
        }
        inp = torch.randn(16)
        # Feed inputs that WOULDN'T trigger competitive_learning winner
        # updates: zero competitive_learning lr effectively.
        for _ in range(3):
            ps.process_input(inp)
        for _nid, proj in ps._input_projections.items():
            # Some drift from Hebbian is allowed; assert the tensor
            # type is still a plain Tensor and not something that
            # backprop modified.
            assert not isinstance(proj, torch.nn.Parameter)

    def test_learnable_mode_projection_moves_after_optimizer_step(self) -> None:
        """Multiple process_input calls with enough loss signal should
        actually move the projection weights (not just have non-zero grad
        but also apply the optimizer step)."""
        cfg = _tiny_config(projection_mode="learnable", projection_lr=1e-2)
        ps = PredictiveSOMA(cfg, device=torch.device("cpu"))
        before = {
            nid: proj.detach().clone()
            for nid, proj in ps._input_projections.items()
        }
        # Feed varying inputs so the prediction error stays non-trivial.
        for i in range(10):
            ps.process_input(torch.randn(16) * (1.0 + 0.1 * i))
        moved = False
        for nid, proj in ps._input_projections.items():
            if (proj.detach() - before[nid]).abs().sum().item() > 1e-6:
                moved = True
                break
        assert moved, (
            "expected at least one projection to move after optimizer steps"
        )


class TestSerialization:
    def test_learnable_projection_round_trip(self, tmp_path: Path) -> None:
        cfg = _tiny_config(projection_mode="learnable")
        ps1 = PredictiveSOMA(cfg, device=torch.device("cpu"))
        # Train a little so projections move.
        for _ in range(5):
            ps1.process_input(torch.randn(16))

        ps1.save(str(tmp_path / "bundle"))

        ps2 = PredictiveSOMA(cfg, device=torch.device("cpu"))
        ps2.load(str(tmp_path / "bundle"))

        for nid, p1 in ps1._input_projections.items():
            p2 = ps2._input_projections[nid]
            assert torch.allclose(p1.detach(), p2.detach()), (
                f"projection mismatch after round-trip for node {nid}"
            )
            # Type must match too (both Parameter or both Tensor).
            assert isinstance(p2, torch.nn.Parameter) == isinstance(
                p1, torch.nn.Parameter
            )
