"""Tests for the CL harness: smoke tests + 3-tuple feature-cache support."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from research.cl import harness
from research.cl.baselines.naive import mnist_factory
from research.cl.datasets import permuted_mnist
from research.cl.harness import _evaluate, _train_one_epoch, _unpack_batch

# ------------------------------------------------------------------
# Original smoke tests
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def harness_result():
    """Run a minimal 2-task, 1-epoch harness pass."""
    tasks = permuted_mnist(n_tasks=2, batch_size=256, seed=0)
    return harness.run(
        model_factory=mnist_factory,
        tasks=tasks,
        n_epochs=1,
        device=torch.device("cpu"),
        verbose=False,
    )


def test_accuracy_matrix_shape(harness_result) -> None:
    A = harness_result["accuracy_matrix"]
    assert A.shape == (2, 2)


def test_accuracy_matrix_range(harness_result) -> None:
    A = harness_result["accuracy_matrix"]
    assert np.all(A >= 0.0) and np.all(A <= 1.0)


def test_metrics_present(harness_result) -> None:
    for key in ("acc", "bwt", "fwt", "wall_clock", "task_times"):
        assert key in harness_result


def test_diagonal_reasonable(harness_result) -> None:
    """After 1 epoch of training, accuracy on the current task should be above chance (10%)."""
    A = harness_result["accuracy_matrix"]
    for i in range(2):
        assert A[i, i] > 0.10, f"Task {i} diagonal {A[i, i]:.3f} below chance"


# ------------------------------------------------------------------
# _unpack_batch tests
# ------------------------------------------------------------------

CPU = torch.device("cpu")


class TestUnpackBatch:
    """Tests for _unpack_batch with 2-tuple and 3-tuple inputs."""

    def test_2tuple_returns_none_cache(self) -> None:
        """2-tuple (x, y) should return (x, y, None)."""
        x = torch.randn(4, 8)
        y = torch.tensor([0, 1, 2, 1])
        x_out, y_out, cache_out = _unpack_batch((x, y), CPU)
        assert torch.equal(x_out, x)
        assert torch.equal(y_out, y)
        assert cache_out is None

    def test_3tuple_returns_cache(self) -> None:
        """3-tuple (x, soma_cache, y) should return (x, y, soma_cache)."""
        x = torch.randn(4, 8)
        soma_cache = torch.randn(4, 16)
        y = torch.tensor([0, 1, 2, 1])
        x_out, y_out, cache_out = _unpack_batch((x, soma_cache, y), CPU)
        assert torch.equal(x_out, x)
        assert torch.equal(y_out, y)
        assert cache_out is not None
        assert torch.equal(cache_out, soma_cache)

    def test_3tuple_reorders_correctly(self) -> None:
        """The 3-tuple order is (x, soma_cache, y) in, (x, y, soma_cache) out."""
        x = torch.ones(2, 3)
        soma_cache = torch.ones(2, 5) * 2.0
        y = torch.tensor([0, 1])
        x_out, y_out, cache_out = _unpack_batch((x, soma_cache, y), CPU)
        # x should be all 1s, cache all 2s
        assert x_out.mean().item() == pytest.approx(1.0)
        assert cache_out is not None
        assert cache_out.mean().item() == pytest.approx(2.0)

    def test_tensors_moved_to_device(self) -> None:
        """All returned tensors should be on the target device."""
        x = torch.randn(2, 4)
        soma_cache = torch.randn(2, 6)
        y = torch.tensor([0, 1])
        x_out, y_out, cache_out = _unpack_batch((x, soma_cache, y), CPU)
        assert x_out.device == CPU
        assert y_out.device == CPU
        assert cache_out is not None
        assert cache_out.device == CPU


# ------------------------------------------------------------------
# Helpers for _evaluate / _train_one_epoch tests
# ------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """Minimal model that accepts optional soma_cache kwarg."""

    def __init__(self, input_dim: int, num_classes: int, cache_dim: int = 0) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, num_classes)
        self._cache_dim = cache_dim
        if cache_dim > 0:
            self.cache_proj = nn.Linear(input_dim + cache_dim, num_classes)

    def forward(self, x: torch.Tensor, soma_cache: torch.Tensor | None = None) -> torch.Tensor:
        if soma_cache is not None:
            combined = torch.cat([x, soma_cache], dim=1)
            return self.cache_proj(combined)
        return self.proj(x)


def _make_2tuple_loader(
    n_samples: int = 32,
    input_dim: int = 8,
    num_classes: int = 3,
    batch_size: int = 16,
) -> DataLoader:
    """Create a DataLoader that yields (x, y) 2-tuples."""
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _make_3tuple_loader(
    n_samples: int = 32,
    input_dim: int = 8,
    cache_dim: int = 16,
    num_classes: int = 3,
    batch_size: int = 16,
) -> DataLoader:
    """Create a DataLoader that yields (x, soma_cache, y) 3-tuples."""
    x = torch.randn(n_samples, input_dim)
    soma_cache = torch.randn(n_samples, cache_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return DataLoader(TensorDataset(x, soma_cache, y), batch_size=batch_size)


# ------------------------------------------------------------------
# _evaluate tests
# ------------------------------------------------------------------


class TestEvaluate:
    """Tests for _evaluate with 2-tuple and 3-tuple DataLoaders."""

    def test_evaluate_2tuple(self) -> None:
        """_evaluate should work with a standard (x, y) DataLoader."""
        model = _SimpleModel(input_dim=8, num_classes=3)
        loader = _make_2tuple_loader()
        acc = _evaluate(model, loader, CPU)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_3tuple(self) -> None:
        """_evaluate should work with a (x, soma_cache, y) DataLoader."""
        model = _SimpleModel(input_dim=8, num_classes=3, cache_dim=16)
        loader = _make_3tuple_loader()
        acc = _evaluate(model, loader, CPU)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_restores_train_mode(self) -> None:
        """_evaluate should leave the model in train mode."""
        model = _SimpleModel(input_dim=8, num_classes=3)
        model.train(True)
        loader = _make_2tuple_loader()
        _evaluate(model, loader, CPU)
        assert model.training

    def test_evaluate_empty_loader(self) -> None:
        """_evaluate on an empty loader should return 0.0."""
        model = _SimpleModel(input_dim=8, num_classes=3)
        empty_ds = TensorDataset(torch.empty(0, 8), torch.empty(0, dtype=torch.long))
        loader = DataLoader(empty_ds, batch_size=16)
        acc = _evaluate(model, loader, CPU)
        assert acc == 0.0


# ------------------------------------------------------------------
# _train_one_epoch tests
# ------------------------------------------------------------------


class TestTrainOneEpoch:
    """Tests for _train_one_epoch with 2-tuple and 3-tuple DataLoaders."""

    def test_train_one_epoch_2tuple(self) -> None:
        """_train_one_epoch should run without errors on (x, y) data."""
        model = _SimpleModel(input_dim=8, num_classes=3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loader = _make_2tuple_loader()
        loss = _train_one_epoch(model, optimizer, loader, CPU)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_train_one_epoch_3tuple(self) -> None:
        """_train_one_epoch should run without errors on (x, soma_cache, y) data."""
        model = _SimpleModel(input_dim=8, num_classes=3, cache_dim=16)
        optimizer = SGD(model.parameters(), lr=0.01)
        loader = _make_3tuple_loader()
        loss = _train_one_epoch(model, optimizer, loader, CPU)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_train_one_epoch_updates_params(self) -> None:
        """Training should actually update model parameters."""
        model = _SimpleModel(input_dim=8, num_classes=3, cache_dim=16)
        optimizer = SGD(model.parameters(), lr=0.1)
        loader = _make_3tuple_loader()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        _train_one_epoch(model, optimizer, loader, CPU)
        any_changed = any(not torch.equal(params_before[n], p) for n, p in model.named_parameters())
        assert any_changed, "No parameters changed after training"

    def test_train_one_epoch_loss_decreases(self) -> None:
        """Loss should decrease over multiple epochs."""
        torch.manual_seed(42)
        model = _SimpleModel(input_dim=8, num_classes=3)
        optimizer = SGD(model.parameters(), lr=0.05)
        loader = _make_2tuple_loader(n_samples=64)
        loss1 = _train_one_epoch(model, optimizer, loader, CPU)
        for _ in range(4):
            _train_one_epoch(model, optimizer, loader, CPU)
        loss5 = _train_one_epoch(model, optimizer, loader, CPU)
        assert loss5 < loss1, f"Loss did not decrease: {loss1:.4f} -> {loss5:.4f}"


# ------------------------------------------------------------------
# SomaClassifier.forward cache path tests (lightweight, no real SOMA)
# ------------------------------------------------------------------


class _FakeSomaClassifier(nn.Module):
    """Mimics SomaClassifier forward logic for testing cache vs full path.

    Uses simple linear layers instead of a real SOMA graph. Replicates
    the branching logic in SomaClassifier.forward: when soma_cache is
    provided, concatenate input_proj(x) with soma_cache; otherwise,
    concatenate input_proj(x) with a "graph output" placeholder.
    """

    def __init__(
        self,
        input_dim: int = 8,
        proj_dim: int = 4,
        soma_out_dim: int = 6,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim, bias=False)
        # Simulates the SOMA graph output path
        self.fake_graph = nn.Linear(proj_dim, soma_out_dim, bias=False)
        head_dim = proj_dim + soma_out_dim
        self.layer_norm: nn.LayerNorm | None = None
        self.head = nn.Linear(head_dim, num_classes)
        self._use_cached = False

    def forward(self, x: torch.Tensor, soma_cache: torch.Tensor | None = None) -> torch.Tensor:
        """Replicate SomaClassifier.forward branching."""
        if soma_cache is not None:
            projected = self.input_proj(x)
            combined = torch.cat([projected, soma_cache], dim=1)
        elif self._use_cached:
            combined = x
        else:
            projected = self.input_proj(x)
            soma_t = self.fake_graph(projected).detach()
            combined = torch.cat([projected, soma_t], dim=1)

        if self.layer_norm is not None:
            combined = self.layer_norm(combined)
        logits = self.head(combined)
        return logits


class TestSomaClassifierForward:
    """Tests for the SomaClassifier forward branching (cache vs full path)."""

    def test_forward_with_soma_cache_uses_input_proj(self) -> None:
        """When soma_cache is provided, input_proj should still process x."""
        model = _FakeSomaClassifier(input_dim=8, proj_dim=4, soma_out_dim=6)
        x = torch.randn(2, 8)
        soma_cache = torch.randn(2, 6)
        logits = model(x, soma_cache=soma_cache)
        assert logits.shape == (2, 3)

    def test_forward_without_cache_uses_graph_path(self) -> None:
        """Without soma_cache, forward should use the full graph path."""
        model = _FakeSomaClassifier(input_dim=8, proj_dim=4, soma_out_dim=6)
        x = torch.randn(2, 8)
        logits = model(x)
        assert logits.shape == (2, 3)

    def test_cache_path_input_proj_gets_gradients(self) -> None:
        """input_proj should receive gradients when using the cache path."""
        model = _FakeSomaClassifier(input_dim=8, proj_dim=4, soma_out_dim=6)
        x = torch.randn(4, 8)
        soma_cache = torch.randn(4, 6)
        logits = model(x, soma_cache=soma_cache)
        loss = logits.sum()
        loss.backward()
        assert model.input_proj.weight.grad is not None
        assert model.input_proj.weight.grad.abs().sum() > 0

    def test_cache_and_graph_produce_same_shape(self) -> None:
        """Both paths should produce identically shaped output."""
        model = _FakeSomaClassifier(input_dim=8, proj_dim=4, soma_out_dim=6)
        x = torch.randn(3, 8)
        soma_cache = torch.randn(3, 6)
        logits_cached = model(x, soma_cache=soma_cache)
        logits_graph = model(x)
        assert logits_cached.shape == logits_graph.shape

    def test_evaluate_with_3tuple_and_cache_model(self) -> None:
        """End-to-end: _evaluate works with a cache-aware model + 3-tuple loader."""
        model = _FakeSomaClassifier(input_dim=8, proj_dim=4, soma_out_dim=16, num_classes=3)
        loader = _make_3tuple_loader(n_samples=32, input_dim=8, cache_dim=16, num_classes=3)
        acc = _evaluate(model, loader, CPU)
        assert 0.0 <= acc <= 1.0

    def test_train_with_3tuple_and_cache_model(self) -> None:
        """End-to-end: _train_one_epoch works with cache-aware model + 3-tuple loader."""
        model = _FakeSomaClassifier(input_dim=8, proj_dim=4, soma_out_dim=16, num_classes=3)
        optimizer = SGD(model.parameters(), lr=0.01)
        loader = _make_3tuple_loader(n_samples=32, input_dim=8, cache_dim=16, num_classes=3)
        loss = _train_one_epoch(model, optimizer, loader, CPU)
        assert isinstance(loss, float)
        assert loss > 0.0
