"""Continual-learning metrics (Lopez-Paz & Ranzato, 2017).

All functions take an accuracy matrix ``A`` of shape ``(T, T)`` where
``A[i][j]`` is the accuracy on task *j*'s test set after the model has been
trained through task *i*.

References
----------
Lopez-Paz, D. & Ranzato, M. (2017). Gradient episodic memory for continual
learning. *NeurIPS*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def acc(A: NDArray[np.floating]) -> float:
    """Average accuracy after training on all tasks (last row mean)."""
    T = A.shape[0]
    return float(np.mean(A[T - 1, :]))


def bwt(A: NDArray[np.floating]) -> float:
    """Backward transfer: average accuracy drop on earlier tasks.

    BWT = (1 / (T-1)) * sum_{i=0}^{T-2} (A[T-1, i] - A[i, i])

    Negative BWT means forgetting.
    """
    T = A.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([A[T - 1, i] - A[i, i] for i in range(T - 1)]))


def fwt(A: NDArray[np.floating], random_baseline: NDArray[np.floating] | None = None) -> float:
    """Forward transfer: accuracy gain on unseen tasks vs random init.

    FWT = (1 / (T-1)) * sum_{i=1}^{T-1} (A[i-1, i] - b_i)

    where ``b_i`` is the random-init accuracy on task *i* (defaults to
    ``1/n_classes`` = 0.1 for 10-class problems if not provided).
    """
    T = A.shape[0]
    if T < 2:
        return 0.0
    if random_baseline is None:
        random_baseline = np.full(T, 0.1)
    return float(np.mean([A[i - 1, i] - random_baseline[i] for i in range(1, T)]))
