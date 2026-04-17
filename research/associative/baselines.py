"""Baseline associative-memory models for D1.

Three models, all satisfying the ``RecallModel`` protocol:
  1. ClassicalHopfield -- discrete bipolar Hopfield network (sign rule).
  2. ModernHopfield -- continuous Hopfield (Ramsauer et al. 2020, softmax energy).
  3. KNNRecall -- k-nearest-neighbour (k=1) lookup by Euclidean distance.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ClassicalHopfield:
    """Textbook discrete Hopfield network.

    Weight matrix: ``W = (1/N) sum_i xi_i @ xi_i^T``, diagonal zeroed.
    Recall: iterate ``x <- sign(W @ x)`` until convergence or *max_iter*.
    """

    def __init__(self, max_iter: int = 100) -> None:
        self.max_iter = max_iter
        self.W: Tensor | None = None
        self._patterns: Tensor | None = None

    def store(self, patterns: Tensor) -> None:
        """Store bipolar patterns (N, D) into the weight matrix."""
        n, d = patterns.shape
        self._patterns = patterns
        # W = (1/N) * X^T @ X, with diagonal zeroed
        self.W = (patterns.T @ patterns) / n  # (D, D)
        self.W.fill_diagonal_(0.0)

    def recall(self, probe: Tensor) -> Tensor:
        """Recall from (M, D) probes. Returns (M, D) bipolar patterns."""
        assert self.W is not None, "Must call store() first"
        x = probe.clone()
        # Replace zeros (masked bits) with random +/-1 for initial state
        zero_mask = x == 0
        if zero_mask.any():
            x[zero_mask] = torch.sign(torch.randn(zero_mask.sum().item(), device=x.device))
            # Ensure no zeros remain
            x[x == 0] = 1.0

        for _ in range(self.max_iter):
            x_new = torch.sign(x @ self.W.T)
            # sign(0) = 0 in PyTorch; replace with previous value
            zeros = x_new == 0
            x_new[zeros] = x[zeros]
            if torch.equal(x_new, x):
                break
            x = x_new
        return x


class ModernHopfield:
    """Modern continuous Hopfield network (Ramsauer et al. 2020).

    Energy-based recall with softmax attention:
        ``x_new = X @ softmax(beta * X^T @ x)``
    where X is (D, N) stored-pattern matrix.
    """

    def __init__(self, beta: float = 10.0, n_iter: int = 5) -> None:
        self.beta = beta
        self.n_iter = n_iter
        self.X: Tensor | None = None  # (D, N)

    def store(self, patterns: Tensor) -> None:
        """Store patterns (N, D)."""
        self.X = patterns.T  # (D, N)

    def recall(self, probe: Tensor) -> Tensor:
        """Recall from (M, D) probes. Returns (M, D) continuous patterns."""
        assert self.X is not None, "Must call store() first"
        x = probe.clone()  # (M, D)
        for _ in range(self.n_iter):
            # Attention scores: (M, N)
            scores = self.beta * (x @ self.X)
            # Numerical stability: subtract max
            scores = scores - scores.max(dim=1, keepdim=True).values
            attn = torch.softmax(scores, dim=1)  # (M, N)
            x = attn @ self.X.T  # (M, D)
        return x


class KNNRecall:
    """Trivial 1-nearest-neighbour recall by Euclidean distance.

    Lower bound on any model-based approach.
    """

    def __init__(self) -> None:
        self._patterns: Tensor | None = None

    def store(self, patterns: Tensor) -> None:
        self._patterns = patterns  # (N, D)

    def recall(self, probe: Tensor) -> Tensor:
        """Return the nearest stored pattern for each probe."""
        assert self._patterns is not None, "Must call store() first"
        # (M, N) pairwise distances
        dists = torch.cdist(probe, self._patterns)  # (M, N)
        nearest_idx = dists.argmin(dim=1)  # (M,)
        return self._patterns[nearest_idx]
