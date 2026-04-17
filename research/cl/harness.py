"""Generic continual-learning train/eval harness.

The harness is model-agnostic: it accepts a ``model_factory`` callable that
returns ``(model, optimizer)`` and a list of ``(task_id, train_loader,
test_loader)`` tuples.  It trains sequentially on each task, evaluating on
*all* tasks after each one, and returns the accuracy matrix plus computed
metrics.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from research.cl import metrics


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return classification accuracy on *loader*."""
    model.train(False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    model.train(True)
    return correct / total if total > 0 else 0.0


def _train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train(True)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


ModelFactory = Callable[[torch.device], tuple[nn.Module, Optimizer]]


def run(
    model_factory: ModelFactory,
    tasks: list[tuple[int, DataLoader, DataLoader]],
    n_epochs: int = 5,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full CL benchmark.

    Parameters
    ----------
    model_factory:
        Callable that takes ``device`` and returns ``(model, optimizer)``.
        Called once at the start.
    tasks:
        List of ``(task_id, train_loader, test_loader)``.
    n_epochs:
        Epochs to train on each task.
    device:
        Torch device. Auto-detects CUDA if available.
    verbose:
        Print progress to stdout.

    Returns
    -------
    dict with keys:
        ``accuracy_matrix`` — np array shape ``(T, T)``
        ``acc``             — average accuracy (float)
        ``bwt``             — backward transfer (float)
        ``fwt``             — forward transfer (float)
        ``wall_clock``      — total seconds (float)
        ``task_times``      — per-task training seconds (list)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = len(tasks)
    A = np.zeros((T, T), dtype=np.float64)

    model, optimizer = model_factory(device)
    model.to(device)

    t0 = time.perf_counter()
    task_times: list[float] = []

    for idx, (task_id, train_loader, _test_loader) in enumerate(tasks):
        task_t0 = time.perf_counter()
        if verbose:
            print(f"--- Task {task_id} ({idx + 1}/{T}) ---")
        for epoch in range(n_epochs):
            avg_loss = _train_one_epoch(model, optimizer, train_loader, device)
            if verbose:
                print(f"  epoch {epoch + 1}/{n_epochs}  loss={avg_loss:.4f}")

        task_times.append(time.perf_counter() - task_t0)

        # Evaluate on ALL tasks
        for j, (_tid_j, _tr_j, te_j) in enumerate(tasks):
            A[idx, j] = _evaluate(model, te_j, device)
        if verbose:
            accs_str = " ".join(f"{A[idx, j]:.3f}" for j in range(T))
            print(f"  eval: [{accs_str}]")

    wall_clock = time.perf_counter() - t0

    return {
        "accuracy_matrix": A,
        "acc": metrics.acc(A),
        "bwt": metrics.bwt(A),
        "fwt": metrics.fwt(A),
        "wall_clock": wall_clock,
        "task_times": task_times,
    }
