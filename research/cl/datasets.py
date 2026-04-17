"""CL benchmark dataset loaders.

Provides:
  - ``permuted_mnist(n_tasks, batch_size, seed)`` -- Permuted-MNIST (10 tasks default)
  - ``split_cifar10(batch_size)``                 -- Split-CIFAR-10  (5 tasks, 2 classes each)

Each returns an iterator of ``(task_id, train_loader, test_loader)`` tuples.

TODO: CORe50 loader (large download, complex session splits -- deferred from B1).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

DATA_ROOT = "research/cl/data/"


# ---------------------------------------------------------------------------
# Permuted-MNIST
# ---------------------------------------------------------------------------


def _make_permutation(task_id: int, n_pixels: int, seed: int) -> torch.Tensor:
    """Deterministic pixel permutation for *task_id*."""
    gen = torch.Generator().manual_seed(seed + task_id)
    return torch.randperm(n_pixels, generator=gen)


def permuted_mnist(
    n_tasks: int = 10,
    batch_size: int = 128,
    seed: int = 0,
) -> list[tuple[int, DataLoader, DataLoader]]:
    """Return a list of ``(task_id, train_loader, test_loader)`` for Permuted-MNIST.

    Task 0 uses the identity permutation (original MNIST).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    # Pre-load into tensors once (MNIST is small)
    train_x = train_ds.data.float().view(-1, 784) / 255.0
    train_y = train_ds.targets
    test_x = test_ds.data.float().view(-1, 784) / 255.0
    test_y = test_ds.targets

    n_pixels = 784
    tasks: list[tuple[int, DataLoader, DataLoader]] = []

    for t in range(n_tasks):
        perm = torch.arange(n_pixels) if t == 0 else _make_permutation(t, n_pixels, seed)

        tr_loader = DataLoader(
            TensorDataset(train_x[:, perm], train_y),
            batch_size=batch_size,
            shuffle=True,
        )
        te_loader = DataLoader(
            TensorDataset(test_x[:, perm], test_y),
            batch_size=batch_size,
            shuffle=False,
        )
        tasks.append((t, tr_loader, te_loader))

    return tasks


# ---------------------------------------------------------------------------
# Split-CIFAR-10
# ---------------------------------------------------------------------------

# 5 tasks, 2 classes each: (0,1), (2,3), (4,5), (6,7), (8,9)
CIFAR_SPLITS: list[tuple[int, ...]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]


def _filter_classes(
    images: torch.Tensor,
    labels: torch.Tensor,
    classes: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for c in classes:
        mask |= labels == c
    return images[mask], labels[mask]


def split_cifar10(
    batch_size: int = 128,
) -> list[tuple[int, DataLoader, DataLoader]]:
    """Return a list of ``(task_id, train_loader, test_loader)`` for Split-CIFAR-10.

    Each task contains 2 classes. Labels are the *original* CIFAR class ids
    (not remapped to 0/1), so the output layer should be 10-way.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)

    # Materialize
    train_x = torch.tensor(train_ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    train_x = train_x.reshape(len(train_x), -1)  # flatten to 3072
    train_y = torch.tensor(train_ds.targets, dtype=torch.long)

    test_x = torch.tensor(test_ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    test_x = test_x.reshape(len(test_x), -1)
    test_y = torch.tensor(test_ds.targets, dtype=torch.long)

    tasks: list[tuple[int, DataLoader, DataLoader]] = []
    for t, classes in enumerate(CIFAR_SPLITS):
        tr_imgs, tr_labs = _filter_classes(train_x, train_y, classes)
        te_imgs, te_labs = _filter_classes(test_x, test_y, classes)
        tr_loader = DataLoader(
            TensorDataset(tr_imgs, tr_labs),
            batch_size=batch_size,
            shuffle=True,
        )
        te_loader = DataLoader(
            TensorDataset(te_imgs, te_labs),
            batch_size=batch_size,
            shuffle=False,
        )
        tasks.append((t, tr_loader, te_loader))

    return tasks
