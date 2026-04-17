"""Smoke test for the CL harness: 2-task MNIST, 1 epoch, no crash."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.cl import harness
from research.cl.baselines.naive import mnist_factory
from research.cl.datasets import permuted_mnist


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
