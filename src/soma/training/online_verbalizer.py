"""Online verbalizer training infrastructure.

Each ChatSession turn becomes a tiny gradient step. A ReplayBuffer
retains prior (user_text, response) exchanges so the verbalizer doesn't
overfit to the latest turn. A divergence monitor (T6) freezes updates
if loss starts rising monotonically.

Phase 6 Task 2 scope: ChatExchange dataclass + ReplayBuffer only. The
OnlineVerbalizerTrainer that uses them lives in T3+.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from soma.core.config import SOMAConfig

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatExchange:
    """One (user_text, response) pair in the replay buffer."""

    user_text: str
    response: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


class ReplayBuffer:
    """Fixed-size FIFO buffer of ChatExchange for online training replay.

    Entries evict oldest-first when ``add`` pushes past ``capacity``.
    ``sample`` draws uniformly without replacement.
    """

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.entries: deque[ChatExchange] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, exchange: ChatExchange) -> None:
        self.entries.append(exchange)

    def sample(self, *, n: int) -> list[ChatExchange]:
        """Return up to ``n`` entries, uniform random, no replacement.

        When ``n`` >= len(buffer), returns all entries. Order of returned
        list is unspecified (sample order).
        """
        buf_len = len(self.entries)
        if buf_len == 0:
            return []
        if n >= buf_len:
            return list(self.entries)
        return random.sample(list(self.entries), n)


class OnlineVerbalizerTrainer:
    """Wraps VerbalizerTrainer with replay + divergence for online updates.

    Each ChatSession turn calls ``step(user_text, response)`` which:
      1. Records the exchange in the replay buffer
      2. Samples a small batch (latest + random prior turns)
      3. Runs a tiny training step on each sample via ``inner.train_step``
      4. Appends mean loss to the history and checks for divergence

    The inner trainer's optimizer LR is reset at construction to
    ``config.online_verbalizer_lr`` — intentionally 10x smaller than
    bootstrap so per-turn updates are noise-robust. Adam's first/second
    moments carry over from bootstrap (same param_group), which is
    desirable: bootstrap's accumulated gradient history still primes the
    projector.
    """

    def __init__(
        self,
        *,
        inner: Any,  # VerbalizerTrainer — typed Any to avoid circular imports
        config: SOMAConfig,
    ) -> None:
        self.inner = inner
        self.config = config
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)
        self._loss_history: deque[float] = deque(maxlen=config.divergence_window)
        self.is_diverged: bool = False

        # Swap the inner optimizer's LR to the online rate. Preserves Adam
        # moment state from bootstrap — we want that history to keep
        # priming the projector.
        for group in self.inner.optim.param_groups:
            group["lr"] = config.online_verbalizer_lr

    def record(self, *, user_text: str, response: str) -> None:
        """Append a new exchange to the replay buffer."""
        self.replay_buffer.add(ChatExchange(user_text=user_text, response=response))

    def sample_batch(self, *, batch_size: int) -> list[str]:
        """Return user-text samples for training.

        Layout: [latest, random_0, random_1, ..., random_{batch_size-2}].
        If the buffer has fewer than ``batch_size`` entries, returns all
        of them with the latest first.
        """
        buf_len = len(self.replay_buffer)
        if buf_len == 0:
            return []

        latest = self.replay_buffer.entries[-1].user_text
        remaining = batch_size - 1
        if remaining <= 0:
            return [latest]
        # Exclude the latest from the random pool so it doesn't duplicate.
        pool = list(self.replay_buffer.entries)[:-1]
        if not pool:
            return [latest]
        sampled = random.sample(pool, min(remaining, len(pool)))
        return [latest] + [ex.user_text for ex in sampled]
