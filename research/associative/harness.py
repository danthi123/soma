"""Generic recall evaluator for associative-memory benchmarks (D1).

Defines the ``RecallModel`` protocol and the ``evaluate_recall`` function
that scores any model on stored-pattern recall tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class RecallModel(Protocol):
    """Minimal interface that any associative-memory model must satisfy."""

    def store(self, patterns: Tensor) -> None:
        """Memorise *patterns* (N, D)."""
        ...

    def recall(self, probe: Tensor) -> Tensor:
        """Given a (possibly noisy) probe (M, D), return recalled patterns (M, D)."""
        ...


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class RecallMetrics:
    """Metrics returned by ``evaluate_recall``."""

    exact_match_rate: float = 0.0
    per_element_accuracy: float = 0.0
    mse: float = 0.0
    n_queries: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        d: dict[str, float | int] = {
            "exact_match_rate": self.exact_match_rate,
            "per_element_accuracy": self.per_element_accuracy,
            "mse": self.mse,
            "n_queries": self.n_queries,
        }
        d.update(self.extra)
        return d


def evaluate_recall(
    model: RecallModel,
    stored_patterns: Tensor,
    probes: Tensor,
    *,
    bipolar: bool = False,
) -> RecallMetrics:
    """Run *model* on *probes* and compare to *stored_patterns*.

    Parameters
    ----------
    bipolar:
        If True, binarise the recalled output via ``sign()`` before
        comparison (appropriate for Hopfield-style binary patterns).
    """
    model.store(stored_patterns)
    with torch.no_grad():
        recalled = model.recall(probes)

    if bipolar:
        recalled = recalled.sign()

    n, d = stored_patterns.shape

    # Per-element accuracy: fraction of elements that match exactly
    if bipolar:
        matches = (recalled == stored_patterns).float()
    else:
        # For continuous patterns, "match" = within a small tolerance
        matches = (torch.abs(recalled - stored_patterns) < 0.05).float()
    per_element_accuracy = matches.mean().item()

    # Exact match: every element in the pattern matches
    exact = matches.sum(dim=1) == d
    exact_match_rate = exact.float().mean().item()

    # MSE
    mse = ((recalled - stored_patterns) ** 2).mean().item()

    return RecallMetrics(
        exact_match_rate=exact_match_rate,
        per_element_accuracy=per_element_accuracy,
        mse=mse,
        n_queries=n,
    )


def evaluate_recall_nearest(
    model: RecallModel,
    stored_patterns: Tensor,
    stored_labels: Tensor,
    probes: Tensor,
) -> RecallMetrics:
    """Evaluate recall by finding the nearest stored pattern to each recalled output.

    Useful for MNIST-style benchmarks where we care about class accuracy,
    not exact pixel reconstruction.
    """
    model.store(stored_patterns)
    with torch.no_grad():
        recalled = model.recall(probes)

    n = stored_patterns.shape[0]

    # MSE against the original (un-noised) patterns
    mse = ((recalled - stored_patterns) ** 2).mean().item()

    # For each recalled pattern, find the nearest stored pattern
    # and check if the class label matches
    # recalled: (N, D), stored_patterns: (N, D)
    # We compare each recalled[i] against ALL stored patterns
    dists = torch.cdist(recalled.unsqueeze(0), stored_patterns.unsqueeze(0)).squeeze(0)  # (N, N)
    nearest_idx = dists.argmin(dim=1)  # (N,)
    predicted_labels = stored_labels[nearest_idx]

    class_accuracy = (predicted_labels == stored_labels).float().mean().item()

    # Per-pixel accuracy (within tolerance)
    per_pixel = (torch.abs(recalled - stored_patterns) < 0.1).float().mean().item()

    return RecallMetrics(
        exact_match_rate=class_accuracy,  # repurpose: class accuracy
        per_element_accuracy=per_pixel,
        mse=mse,
        n_queries=n,
        extra={"class_accuracy": class_accuracy},
    )
