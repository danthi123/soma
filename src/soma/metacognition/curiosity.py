"""Curiosity module — intrinsic motivation via learning-progress signal.

Whitepaper Section 7.1.

Maintains a per-domain ring buffer of prediction errors. For each new
input, a learned ``domain_classifier`` assigns it to one of ``num_domains``
bins; the curiosity score is the rate of improvement in that bin's error
trace, multiplied by the current error magnitude (so solved-but-zero
domains fade out and impossible-but-stuck domains also fade out).
"""

from __future__ import annotations

import torch
from torch import nn

from soma.core.ring_buffer import RingBuffer


class CuriosityModule(nn.Module):
    """Tracks per-domain error progress and returns a curiosity score.

    Parameters
    ----------
    input_dim:
        Dimension of the representation passed to ``compute_curiosity``.
        The learnable ``domain_classifier`` maps ``input_dim`` -> ``num_domains``.
    num_domains:
        Number of input categories to track separately.
    window_size:
        Size of each domain's error history ring buffer.
    warmup:
        Return the uniform default curiosity (1.0) until a domain has
        seen at least this many samples.
    """

    def __init__(
        self,
        input_dim: int = 128,
        num_domains: int = 8,
        *,
        window_size: int = 100,
        warmup: int = 10,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_domains <= 0:
            raise ValueError(f"num_domains must be positive, got {num_domains}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if warmup <= 0:
            raise ValueError(f"warmup must be positive, got {warmup}")

        self.input_dim = input_dim
        self.num_domains = num_domains
        self.window_size = window_size
        self.warmup = warmup

        # Per-domain ring buffers of prediction errors.
        self.error_histories: list[RingBuffer] = [
            RingBuffer(window_size) for _ in range(num_domains)
        ]

        # Learnable domain classifier.
        self.domain_classifier = nn.Linear(input_dim, num_domains)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def classify_domain(self, input_repr: torch.Tensor) -> int:
        """Return the argmax domain label for ``input_repr``."""
        if input_repr.shape != (self.input_dim,):
            raise ValueError(
                f"CuriosityModule expects shape ({self.input_dim},), got {tuple(input_repr.shape)}"
            )
        with torch.no_grad():
            logits = self.domain_classifier(input_repr.detach())
            return int(logits.argmax().item())

    def compute_curiosity(
        self,
        input_repr: torch.Tensor,
        prediction_error: float,
    ) -> float:
        """Return a curiosity score in ``[0, +inf)`` for this input.

        High curiosity = the error is dropping fast in this domain (learning
        progress), scaled by the current error magnitude so trivially-solved
        domains still dampen to zero.
        """
        if prediction_error < 0.0:
            raise ValueError(f"prediction_error must be non-negative, got {prediction_error}")

        domain = self.classify_domain(input_repr)
        history = self.error_histories[domain]
        history.append(prediction_error)

        values = history.get_all()
        if len(values) < self.warmup:
            # Uniform default until enough data — matches whitepaper's
            # "cold-start" behavior.
            return 1.0

        # Compare recent vs older window to estimate learning progress.
        recent_window = max(1, self.window_size // 10)
        older_window = max(recent_window + 1, self.window_size // 2)

        recent_slice = values[-recent_window:]
        older_slice = values[-older_window:-recent_window] or values[-recent_window:]

        recent_mean = sum(recent_slice) / len(recent_slice)
        older_mean = sum(older_slice) / len(older_slice)

        learning_progress = (older_mean - recent_mean) / older_mean if older_mean > 0.0 else 0.0

        learning_progress = max(0.0, learning_progress)
        magnitude_factor = recent_mean / (recent_mean + 0.1)
        return float(learning_progress * magnitude_factor)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def domain_mean_error(self, domain: int) -> float:
        """Mean of the stored error history for ``domain`` (0 if empty)."""
        if not 0 <= domain < self.num_domains:
            raise IndexError(f"domain {domain} out of range [0, {self.num_domains})")
        return self.error_histories[domain].mean()

    def reset_histories(self) -> None:
        for history in self.error_histories:
            history.clear()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, list[float]]:
        """Return the per-domain error histories as plain lists."""
        return {str(i): h.to_list() for i, h in enumerate(self.error_histories)}

    def load_histories(self, data: dict[str, list[float]]) -> None:
        for key, values in data.items():
            idx = int(key)
            if not 0 <= idx < self.num_domains:
                continue
            self.error_histories[idx] = (
                RingBuffer.from_list(values, capacity=self.window_size)
                if values
                else RingBuffer(self.window_size)
            )
