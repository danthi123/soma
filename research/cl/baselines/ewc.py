"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

Online EWC variant: maintains a running Fisher accumulator rather than
storing per-task Fisher matrices. After each task the diagonal Fisher
is computed from the training data and added to a running sum. The
penalty is ``(lambda / 2) * sum(F_running * (theta - theta_star)^2)``.

References
----------
Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in
neural networks. *PNAS*, 114(13), 3521-3526.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class EWC:
    """Online EWC wrapper.

    Usage with the CL harness::

        ewc = EWC(lam=400.0)
        result = harness.run(
            model_factory=factory,
            tasks=tasks,
            train_one_epoch=ewc.train_one_epoch,
            on_task_end=ewc.on_task_end,
        )
    """

    def __init__(
        self,
        lam: float = 400.0,
        fisher_samples: int | None = None,
        fisher_clamp: float = 100.0,
    ):
        """
        Parameters
        ----------
        lam:
            Regularization strength.
        fisher_samples:
            Number of samples used to estimate the Fisher diagonal.
            ``None`` means use the full training set.
        fisher_clamp:
            Upper bound for Fisher diagonal entries. Prevents numerical
            blow-up when Fisher values are very large (common on CIFAR
            with high-dimensional inputs).
        """
        self.lam = lam
        self.fisher_samples = fisher_samples
        self.fisher_clamp = fisher_clamp

        # Running Fisher accumulator (sum of per-task Fishers)
        self._fisher: dict[str, torch.Tensor] = {}
        # Optimal parameters snapshot (updated after each task)
        self._theta_star: dict[str, torch.Tensor] = {}
        # Number of tasks consolidated so far
        self._n_tasks_consolidated: int = 0

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC penalty term (scalar tensor).

        The accumulated Fisher is divided by the number of consolidated
        tasks so the penalty scale stays constant regardless of how many
        tasks have been seen.
        """
        if self._n_tasks_consolidated == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        scale = 1.0 / self._n_tasks_consolidated
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self._fisher:
                diff_sq = (param - self._theta_star[name]).pow(2)
                loss += (self._fisher[name] * diff_sq).sum()
        return loss * scale

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
        """Train one epoch with EWC penalty added to the cross-entropy loss."""
        model.train(True)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            ce_loss = criterion(logits, y)
            ewc_loss = (self.lam / 2.0) * self.penalty(model)
            loss = ce_loss + ewc_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Post-task hook: compute Fisher + snapshot parameters
    # ------------------------------------------------------------------

    def on_task_end(
        self,
        model: nn.Module,
        task_idx: int,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """Compute diagonal Fisher on the completed task and update accumulators."""
        fisher = self._compute_fisher(model, train_loader, device)

        # Online EWC: accumulate Fisher across tasks
        for name, f in fisher.items():
            if name in self._fisher:
                self._fisher[name] = self._fisher[name] + f
            else:
                self._fisher[name] = f.clone()

        # Clamp to prevent numerical blow-up
        if self.fisher_clamp is not None:
            for name in self._fisher:
                self._fisher[name].clamp_(max=self.fisher_clamp)

        # Snapshot optimal parameters
        self._theta_star = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        self._n_tasks_consolidated += 1

    def _compute_fisher(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Estimate diagonal Fisher using squared gradients of the log-likelihood."""
        model.train(False)
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }

        n_batches = 0
        n_samples = 0
        criterion = nn.CrossEntropyLoss()  # mean reduction per batch

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            logits = model(x)
            # Use the model's own predictions (empirical Fisher)
            loss = criterion(logits, logits.argmax(dim=1))
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2)
            n_batches += 1
            n_samples += x.size(0)

            if self.fisher_samples is not None and n_samples >= self.fisher_samples:
                break

        # Average over batches
        for name in fisher:
            fisher[name] /= max(n_batches, 1)

        model.train(True)
        return fisher


# ------------------------------------------------------------------
# Convenience factories (mirror naive.py pattern)
# ------------------------------------------------------------------

def make_ewc_components(
    lam: float = 400.0,
    fisher_samples: int | None = None,
) -> tuple[Callable, Callable]:
    """Return ``(train_one_epoch, on_task_end)`` callables for the harness.

    Example::

        train_fn, hook = make_ewc_components(lam=400)
        result = harness.run(factory, tasks, train_one_epoch=train_fn, on_task_end=hook)
    """
    ewc = EWC(lam=lam, fisher_samples=fisher_samples)
    return ewc.train_one_epoch, ewc.on_task_end
