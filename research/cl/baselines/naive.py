"""Vanilla MLP baseline -- no continual-learning defense.

Trained sequentially on each task with plain SGD. This is the
*catastrophic-forgetting floor*: the worst a reasonable model can do
on CL benchmarks.

Published results: ~20 pct ACC on 10-task Permuted-MNIST.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer


class NaiveMLP(nn.Module):
    """Simple 2-layer MLP for classification."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mnist_factory(device: torch.device) -> tuple[nn.Module, Optimizer]:
    """Model factory for Permuted-MNIST (784 -> 256 -> 10)."""
    model = NaiveMLP(input_dim=784, hidden_dim=256, output_dim=10).to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    return model, optimizer


def cifar_factory(device: torch.device) -> tuple[nn.Module, Optimizer]:
    """Model factory for Split-CIFAR-10 (3072 -> 512 -> 10)."""
    model = NaiveMLP(input_dim=3072, hidden_dim=512, output_dim=10).to(device)
    optimizer = SGD(model.parameters(), lr=0.01)
    return model, optimizer
