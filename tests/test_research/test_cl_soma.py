"""Unit tests for the SOMA CL adapter (research/cl/soma_cl.py)."""

from __future__ import annotations

import torch
import torch.nn as nn

from research.cl.soma_cl import SomaClassifier, _build_soma, _mnist_soma_config


def _make_classifier(
    *,
    frozen: bool = False,
    device: str = "cpu",
) -> SomaClassifier:
    """Build a small SomaClassifier for testing."""
    config = _mnist_soma_config(seed=123)
    soma = _build_soma(config, device=device)
    return SomaClassifier(soma, num_classes=10, input_dim=784, frozen=frozen)


class TestSomaClassifierForwardShape:
    """Verify that forward produces correct output shape."""

    def test_single_sample(self) -> None:
        model = _make_classifier()
        x = torch.randn(1, 784)
        logits = model(x)
        assert logits.shape == (1, 10)

    def test_batch(self) -> None:
        model = _make_classifier()
        x = torch.randn(4, 784)
        logits = model(x)
        assert logits.shape == (4, 10)

    def test_output_requires_grad(self) -> None:
        """Logits should require grad (head is trainable)."""
        model = _make_classifier()
        x = torch.randn(2, 784)
        logits = model(x)
        assert logits.requires_grad


class TestHebrianFires:
    """Verify that SOMA's Hebbian learning fires during forward."""

    def test_global_step_advances(self) -> None:
        """Each sample in the batch should advance global_step."""
        model = _make_classifier()
        step_before = model.soma.global_step
        x = torch.randn(3, 784)
        _ = model(x)
        step_after = model.soma.global_step
        # Each sample triggers one soma.step() call
        assert step_after == step_before + 3

    def test_frozen_skips_updates(self) -> None:
        """Frozen mode should still advance steps but in eval_mode."""
        model = _make_classifier(frozen=True)
        step_before = model.soma.global_step
        x = torch.randn(2, 784)
        _ = model(x)
        step_after = model.soma.global_step
        # Steps still advance (eval_mode=True still increments)
        assert step_after == step_before + 2


class TestHeadTrains:
    """Verify that the linear head updates via backprop."""

    def test_head_weight_changes(self) -> None:
        model = _make_classifier()
        optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        w_before = model.head.weight.data.clone()

        x = torch.randn(4, 784)
        y = torch.randint(0, 10, (4,))

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        w_after = model.head.weight.data
        assert not torch.allclose(
            w_before, w_after
        ), "Head weights should change after training step"


class TestIntegratorAssertion:
    """Verify the seed-graph integrator count assertion."""

    def test_integrators_present(self) -> None:
        config = _mnist_soma_config(seed=42)
        soma = _build_soma(config, device="cpu")
        from soma.core.node import NodeType

        n_int = len(soma.graph.nodes_by_type(NodeType.INTEGRATOR))
        assert n_int == config.initial_integrator_count


class TestConsolidation:
    """Verify that consolidation runs without error."""

    def test_consolidation_runs(self) -> None:
        model = _make_classifier()
        # Feed a few samples to populate episodic memory
        for _ in range(5):
            x = torch.randn(1, 784)
            _ = model(x)
        # Consolidation should not raise
        model.consolidate()

    def test_consolidation_disabled(self) -> None:
        model = _make_classifier()
        model.enable_consolidation = False
        # Should be a no-op
        model.consolidate()
