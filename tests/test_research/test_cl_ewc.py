"""Unit tests for EWC baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from research.cl.baselines.ewc import EWC, make_ewc_components

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


class TestEWCPenalty:
    def test_penalty_zero_before_any_task(self) -> None:
        ewc = EWC(lam=400.0)
        model, _, _ = _tiny_model_and_data()
        p = ewc.penalty(model)
        assert p.item() == 0.0

    def test_penalty_nonzero_after_task(self) -> None:
        ewc = EWC(lam=400.0)
        model, opt, loader = _tiny_model_and_data()

        # Train one epoch, then consolidate
        ewc.train_one_epoch(model, opt, loader, torch.device("cpu"))
        ewc.on_task_end(model, 0, loader, torch.device("cpu"))

        # Perturb parameters
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        p = ewc.penalty(model)
        assert p.item() > 0.0

    def test_fisher_accumulates_across_tasks(self) -> None:
        ewc = EWC(lam=100.0)
        model, opt, loader = _tiny_model_and_data()

        ewc.train_one_epoch(model, opt, loader, torch.device("cpu"))
        ewc.on_task_end(model, 0, loader, torch.device("cpu"))
        fisher_after_1 = {k: v.clone() for k, v in ewc._fisher.items()}

        ewc.train_one_epoch(model, opt, loader, torch.device("cpu"))
        ewc.on_task_end(model, 1, loader, torch.device("cpu"))

        # Fisher should be >= after two tasks (accumulation)
        for name in fisher_after_1:
            assert torch.all(ewc._fisher[name] >= fisher_after_1[name] - 1e-8)


class TestEWCTraining:
    def test_train_one_epoch_returns_float(self) -> None:
        ewc = EWC(lam=400.0)
        model, opt, loader = _tiny_model_and_data()
        loss = ewc.train_one_epoch(model, opt, loader, torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_make_ewc_components(self) -> None:
        train_fn, hook = make_ewc_components(lam=100.0)
        assert callable(train_fn)
        assert callable(hook)


class TestEWCWithHarness:
    """Smoke test: run EWC through the harness on a tiny synthetic task."""

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

        train_fn, hook = make_ewc_components(lam=100.0)
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
