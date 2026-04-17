"""Smoke tests for CL dataset loaders (shape checks, task counts)."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def mnist_tasks():
    """Load 2-task Permuted-MNIST (minimal for shape checks)."""
    from research.cl.datasets import permuted_mnist

    return permuted_mnist(n_tasks=2, batch_size=64, seed=42)


class TestPermutedMNIST:
    def test_task_count(self, mnist_tasks) -> None:
        assert len(mnist_tasks) == 2

    def test_tuple_structure(self, mnist_tasks) -> None:
        task_id, train_loader, test_loader = mnist_tasks[0]
        assert isinstance(task_id, int)
        assert task_id == 0

    def test_batch_shape(self, mnist_tasks) -> None:
        _tid, train_loader, _te = mnist_tasks[0]
        x, y = next(iter(train_loader))
        assert x.shape[1] == 784
        assert y.dtype == torch.long

    def test_permutation_differs(self, mnist_tasks) -> None:
        """Task 0 and task 1 should have different pixel orderings."""
        _t0, tr0, _te0 = mnist_tasks[0]
        _t1, tr1, _te1 = mnist_tasks[1]
        x0, _ = next(iter(tr0))
        x1, _ = next(iter(tr1))
        # The same underlying images but permuted differently --
        # pixel means across the batch should differ.
        assert not torch.allclose(x0.mean(dim=0), x1.mean(dim=0), atol=0.01)


class TestSplitCIFAR10:
    @pytest.fixture(scope="class")
    def cifar_tasks(self):
        from research.cl.datasets import split_cifar10

        return split_cifar10(batch_size=64)

    def test_task_count(self, cifar_tasks) -> None:
        assert len(cifar_tasks) == 5

    def test_batch_shape(self, cifar_tasks) -> None:
        _tid, train_loader, _te = cifar_tasks[0]
        x, y = next(iter(train_loader))
        assert x.shape[1] == 3072  # 3*32*32 flattened
        assert y.dtype == torch.long

    def test_classes_per_task(self, cifar_tasks) -> None:
        """Each task should contain exactly 2 unique class labels."""
        for _tid, _tr, te in cifar_tasks:
            all_labels = torch.cat([y for _, y in te])
            unique = all_labels.unique()
            assert len(unique) == 2, f"Expected 2 classes, got {unique.tolist()}"
