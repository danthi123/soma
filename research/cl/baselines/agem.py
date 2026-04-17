"""Averaged Gradient Episodic Memory (A-GEM, Chaudhry et al., 2019).

Maintains a small episodic memory buffer of past-task samples. At each
training step, if the current gradient would increase loss on the buffer
(negative dot product with reference gradient), project the gradient to
the non-negative half-space.

Simpler than the original GEM (no per-task QP solve) and nearly as
effective on standard benchmarks.

References
----------
Chaudhry, A. et al. (2019). Efficient lifelong learning with A-GEM.
*ICLR*.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class AGEM:
    """A-GEM wrapper.

    Usage with the CL harness::

        agem = AGEM(buffer_per_task=256)
        result = harness.run(
            model_factory=factory,
            tasks=tasks,
            train_one_epoch=agem.train_one_epoch,
            on_task_end=agem.on_task_end,
        )
    """

    def __init__(
        self,
        buffer_per_task: int = 256,
        ref_batch_size: int = 128,
    ):
        """
        Parameters
        ----------
        buffer_per_task:
            Number of samples to store per completed task.
        ref_batch_size:
            Batch size when computing the reference gradient from the
            memory buffer.
        """
        self.buffer_per_task = buffer_per_task
        self.ref_batch_size = ref_batch_size

        # Memory buffer: list of (x, y) tensors from completed tasks
        self._buffer_x: list[torch.Tensor] = []
        self._buffer_y: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_grad(model: nn.Module) -> torch.Tensor:
        """Flatten all parameter gradients into a single vector."""
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(param.numel(), device=param.device))
        return torch.cat(grads)

    @staticmethod
    def _unflatten_grad(model: nn.Module, flat_grad: torch.Tensor) -> None:
        """Write a flat gradient vector back into parameter .grad fields."""
        offset = 0
        for param in model.parameters():
            n = param.numel()
            if param.grad is not None:
                param.grad.copy_(flat_grad[offset : offset + n].view_as(param))
            offset += n

    @staticmethod
    def _project_gradient(g: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
        """Project g so its dot product with g_ref is non-negative.

        If ``g . g_ref >= 0``, return g unchanged.
        Otherwise return ``g - (g.g_ref / g_ref.g_ref) * g_ref``.
        """
        dot = torch.dot(g, g_ref)
        if dot >= 0:
            return g
        ref_sq = torch.dot(g_ref, g_ref)
        if ref_sq < 1e-12:
            return g
        return g - (dot / ref_sq) * g_ref

    # ------------------------------------------------------------------
    # Reference gradient
    # ------------------------------------------------------------------

    def _compute_ref_gradient(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Compute the gradient on a random batch from the memory buffer."""
        if len(self._buffer_x) == 0:
            return None

        # Concatenate all buffer data and sample a batch
        all_x = torch.cat(self._buffer_x, dim=0)
        all_y = torch.cat(self._buffer_y, dim=0)
        n = all_x.size(0)
        indices = torch.randperm(n)[: self.ref_batch_size]
        bx = all_x[indices].to(device)
        by = all_y[indices].to(device)

        model.zero_grad()
        logits = model(bx)
        loss = nn.CrossEntropyLoss()(logits, by)
        loss.backward()

        return self._flatten_grad(model)

    # ------------------------------------------------------------------
    # Training step (plugs into harness)
    # ------------------------------------------------------------------

    def train_one_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """Train one epoch with A-GEM gradient projection."""
        model.train(True)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        n_batches = 0
        has_buffer = len(self._buffer_x) > 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 1. Compute current-task gradient
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # 2. A-GEM projection (only if we have a memory buffer)
            if has_buffer:
                g_task = self._flatten_grad(model)
                g_ref = self._compute_ref_gradient(model, device)
                if g_ref is not None:
                    g_proj = self._project_gradient(g_task, g_ref)
                    self._unflatten_grad(model, g_proj)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Post-task hook: populate memory buffer
    # ------------------------------------------------------------------

    def on_task_end(
        self,
        model: nn.Module,
        task_idx: int,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """Store a random subset of the completed task's training data."""
        # Collect all data from the loader
        all_x_parts: list[torch.Tensor] = []
        all_y_parts: list[torch.Tensor] = []
        for x, y in train_loader:
            all_x_parts.append(x)
            all_y_parts.append(y)
        all_x = torch.cat(all_x_parts, dim=0)
        all_y = torch.cat(all_y_parts, dim=0)

        # Random subset
        n = all_x.size(0)
        k = min(self.buffer_per_task, n)
        indices = torch.randperm(n)[:k]

        # Store on CPU to save GPU memory
        self._buffer_x.append(all_x[indices].cpu())
        self._buffer_y.append(all_y[indices].cpu())


# ------------------------------------------------------------------
# Convenience factories (mirror naive.py / ewc.py pattern)
# ------------------------------------------------------------------

def make_agem_components(
    buffer_per_task: int = 256,
    ref_batch_size: int = 128,
) -> tuple[Callable, Callable]:
    """Return ``(train_one_epoch, on_task_end)`` callables for the harness.

    Example::

        train_fn, hook = make_agem_components(buffer_per_task=256)
        result = harness.run(factory, tasks, train_one_epoch=train_fn, on_task_end=hook)
    """
    agem = AGEM(buffer_per_task=buffer_per_task, ref_batch_size=ref_batch_size)
    return agem.train_one_epoch, agem.on_task_end
