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
import math
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

    def step(self, *, user_text: str, response: str) -> None:
        """Record the exchange and (if not diverged) run one training step.

        The training step runs ``inner.train_step`` on each text in the
        sampled batch (latest + replays). Mean of FINITE losses is appended
        to the rolling window; then divergence is checked.
        """
        self.record(user_text=user_text, response=response)

        if self.is_diverged:
            return

        batch = self.sample_batch(batch_size=self.config.online_batch_size)
        step_losses = []
        for text in batch:
            loss = self.inner.train_step(text=text)
            if math.isfinite(loss):
                step_losses.append(loss)

        if step_losses:
            mean_loss = sum(step_losses) / len(step_losses)
            self._loss_history.append(mean_loss)
            self._check_divergence()

    def _check_divergence(self) -> None:
        """Rolling-window rise detector.

        If the loss history is at least ``divergence_window`` long AND
        ``mean(second half) - mean(first half) > divergence_threshold``,
        freeze further training and log a warning. Only runs on a FULL
        window — partial histories are benign.
        """
        history = list(self._loss_history)
        window = self.config.divergence_window
        if len(history) < window:
            return
        half = window // 2
        mean_first = sum(history[:half]) / half
        mean_second = sum(history[half:]) / (len(history) - half)
        rise = mean_second - mean_first
        if rise > self.config.divergence_threshold:
            self.is_diverged = True
            _log.warning(
                "online verbalizer training diverged: mean(first half)=%.4f "
                "mean(second half)=%.4f rise=%.4f > threshold=%.4f. "
                "Further steps frozen; call reset_divergence_guard() to resume.",
                mean_first,
                mean_second,
                rise,
                self.config.divergence_threshold,
            )

    def reset_divergence_guard(self) -> None:
        """Clear the diverged flag and wipe the loss history."""
        self.is_diverged = False
        self._loss_history.clear()
