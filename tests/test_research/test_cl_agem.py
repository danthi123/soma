"""Unit tests for A-GEM baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from research.cl.baselines.agem import AGEM, make_agem_components

_CPU = torch.device("cpu")


def _tiny_model_and_data(
    device: torch.device = _CPU,
) -> tuple[nn.Module, SGD, DataLoader]:
    """Create a small MLP and a trivial 2-class dataset."""
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)).to(device)
    opt = SGD(model.parameters(), lr=0.01)
    x = torch.randn(64, 8)
    y = torch.randint(0, 2, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    return model, opt, loader


class TestAGEMProjection:
    def test_no_projection_when_aligned(self) -> None:
        g = torch.tensor([1.0, 2.0, 3.0])
        g_ref = torch.tensor([1.0, 1.0, 1.0])
        result = AGEM._project_gradient(g, g_ref)
        assert torch.allclose(result, g)

    def test_projection_when_conflicting(self) -> None:
        g = torch.tensor([-1.0, -1.0, 0.0])
        g_ref = torch.tensor([1.0, 1.0, 0.0])
        result = AGEM._project_gradient(g, g_ref)
        # Projected gradient should have non-negative dot with g_ref
        assert torch.dot(result, g_ref).item() >= -1e-6

    def test_projection_removes_conflicting_component(self) -> None:
        g = torch.tensor([-3.0, 0.0])
        g_ref = torch.tensor([1.0, 0.0])
        result = AGEM._project_gradient(g, g_ref)
        # Dot product should be zero (tangent to constraint)
        assert abs(torch.dot(result, g_ref).item()) < 1e-6


class TestAGEMBuffer:
    def test_buffer_empty_initially(self) -> None:
        agem = AGEM(buffer_per_task=32)
        assert len(agem._buffer_x) == 0

    def test_buffer_populated_after_task(self) -> None:
        agem = AGEM(buffer_per_task=32)
        model, opt, loader = _tiny_model_and_data()
        agem.on_task_end(model, 0, loader, torch.device("cpu"))
        assert len(agem._buffer_x) == 1
        assert agem._buffer_x[0].size(0) == 32

    def test_buffer_size_capped(self) -> None:
        agem = AGEM(buffer_per_task=16)
        model, opt, loader = _tiny_model_and_data()
        # Dataset has 64 samples; buffer should store only 16
        agem.on_task_end(model, 0, loader, torch.device("cpu"))
        assert agem._buffer_x[0].size(0) == 16


class TestAGEMTraining:
    def test_train_one_epoch_no_buffer(self) -> None:
        agem = AGEM(buffer_per_task=32)
        model, opt, loader = _tiny_model_and_data()
        loss = agem.train_one_epoch(model, opt, loader, torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_train_one_epoch_with_buffer(self) -> None:
        agem = AGEM(buffer_per_task=32)
        model, opt, loader = _tiny_model_and_data()
        # Populate buffer from "task 0"
        agem.on_task_end(model, 0, loader, torch.device("cpu"))
        # Train on "task 1" with projection
        loss = agem.train_one_epoch(model, opt, loader, torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_make_agem_components(self) -> None:
        train_fn, hook = make_agem_components(buffer_per_task=64)
        assert callable(train_fn)
        assert callable(hook)


class TestAGEMWithHarness:
    """Smoke test: run A-GEM through the harness on a tiny synthetic task."""

    def test_harness_2_tasks(self) -> None:
        from research.cl import harness

        def factory(device: torch.device) -> tuple[nn.Module, SGD]:
            model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
            return model, SGD(model.parameters(), lr=0.01)

        x0 = torch.randn(100, 8)
        y0 = torch.randint(0, 2, (100,))
        x1 = torch.randn(100, 8)
        y1 = torch.randint(0, 2, (100,))

        tr0 = DataLoader(TensorDataset(x0, y0), batch_size=32)
        tr1 = DataLoader(TensorDataset(x1, y1), batch_size=32)
        te0 = DataLoader(TensorDataset(x0, y0), batch_size=32)
        te1 = DataLoader(TensorDataset(x1, y1), batch_size=32)
        tasks = [(0, tr0, te0), (1, tr1, te1)]

        train_fn, hook = make_agem_components(buffer_per_task=32)
        result = harness.run(
            model_factory=factory,
            tasks=tasks,
            n_epochs=2,
            device=torch.device("cpu"),
            verbose=False,
            train_one_epoch=train_fn,
            on_task_end=hook,
        )

        assert result["accuracy_matrix"].shape == (2, 2)
        assert 0.0 <= result["acc"] <= 1.0
