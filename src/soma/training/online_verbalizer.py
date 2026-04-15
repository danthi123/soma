"""Online verbalizer training infrastructure.

Each ChatSession turn becomes a tiny gradient step. A ReplayBuffer
retains prior (user_text, response) exchanges so the verbalizer doesn't
overfit to the latest turn. A divergence monitor (T6) freezes updates
if loss starts rising monotonically.

Phase 6 Task 2 scope: ChatExchange dataclass + ReplayBuffer only. The
OnlineVerbalizerTrainer that uses them lives in T3+.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime


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
