"""Thread-safe pub/sub data bus with per-channel ring buffers.

Adapted from ``sim/sim/data_bus.py`` (neural-simulator project) — the
substrate is different (SOMA.step results vs. Izhikevich spike events),
but the mechanism is the same: a producer thread publishes onto named
channels, and any number of consumers subscribe to get callbacks or read
the rolling history.

Thread model:
- ``publish`` is called from the training worker thread.
- ``subscribe`` / ``get_history`` / ``latest`` are called from the UI
  main thread during DPG render callbacks.
- An RLock protects the buffer + subscriber list per channel.
- Subscribers run inline on the publisher thread; they MUST be cheap.
  Heavy UI work should stash data into an arg-less ``deque`` and let the
  main thread's per-frame callback drain it.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
Subscriber = Callable[[Any], None]


class DataChannel:
    """Named stream with a ring-buffer history and inline subscribers.

    Parameters
    ----------
    name:
        Human-readable channel identifier (e.g., ``"metrics"``).
    max_history:
        Maximum number of entries retained in the rolling buffer.
    throttle_steps:
        Only store every Nth publish. ``1`` keeps everything; ``10``
        drops 90% of publishes (useful when the producer is faster than
        the UI can render).
    """

    def __init__(self, name: str, max_history: int = 1000, throttle_steps: int = 1) -> None:
        if max_history <= 0:
            raise ValueError(f"max_history must be positive, got {max_history}")
        if throttle_steps <= 0:
            raise ValueError(f"throttle_steps must be positive, got {throttle_steps}")
        self.name = name
        self.max_history = max_history
        self.throttle_steps = throttle_steps
        self._buffer: deque[Any] = deque(maxlen=max_history)
        self._subscribers: list[Subscriber] = []
        self._step_counter = 0
        self._lock = threading.RLock()

    def publish(self, data: Any) -> None:
        """Append to the buffer (respecting throttle) and fan out to subscribers."""
        with self._lock:
            self._step_counter += 1
            if self._step_counter % self.throttle_steps != 0:
                return
            self._buffer.append(data)
            subscribers = list(self._subscribers)  # snapshot outside the lock
        for cb in subscribers:
            # Subscribers run on the publisher thread. Swallow exceptions so
            # one misbehaving listener can't kill the training loop.
            # Intentionally silent — see module docstring. UI side-effects
            # that need to surface errors should log them inside the cb.
            with contextlib.suppress(Exception):
                cb(data)

    def subscribe(self, callback: Subscriber) -> None:
        """Register a callback invoked on every (non-throttled) publish."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        """Remove a previously-registered callback (silent if not found)."""
        with self._lock, contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    def get_history(self, n: int | None = None) -> list[Any]:
        """Return the last ``n`` (or all) entries from the ring buffer."""
        with self._lock:
            items = list(self._buffer)
        if n is None:
            return items
        return items[-n:]

    def latest(self) -> Any | None:
        """Return the most-recently published value, or ``None`` if empty."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        """Drop all buffered entries. Subscribers are left untouched."""
        with self._lock:
            self._buffer.clear()

    @property
    def size(self) -> int:
        """Number of entries currently in the ring buffer."""
        with self._lock:
            return len(self._buffer)


class DataBus:
    """Central registry of named :class:`DataChannel` instances."""

    def __init__(self) -> None:
        self._channels: dict[str, DataChannel] = {}
        self._lock = threading.RLock()

    def create_channel(
        self,
        name: str,
        *,
        max_history: int = 1000,
        throttle_steps: int = 1,
    ) -> DataChannel:
        """Create (or fetch) a channel by name.

        Repeated calls with the same ``name`` return the same channel —
        subscribers from earlier calls stay registered. This matters when
        panels are rebuilt during hot reload.
        """
        with self._lock:
            existing = self._channels.get(name)
            if existing is not None:
                return existing
            channel = DataChannel(name, max_history=max_history, throttle_steps=throttle_steps)
            self._channels[name] = channel
            return channel

    def get(self, name: str) -> DataChannel:
        """Return the channel named ``name``; raise if missing."""
        with self._lock:
            ch = self._channels.get(name)
        if ch is None:
            raise KeyError(f"No channel named {name!r}")
        return ch

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._channels

    def publish(self, name: str, data: Any) -> None:
        """Publish to ``name``; creates the channel on first use."""
        with self._lock:
            ch = self._channels.get(name)
            if ch is None:
                ch = DataChannel(name)
                self._channels[name] = ch
        ch.publish(data)

    def subscribe(self, name: str, callback: Subscriber) -> None:
        """Subscribe to ``name``; creates the channel on first use."""
        with self._lock:
            ch = self._channels.get(name)
            if ch is None:
                ch = DataChannel(name)
                self._channels[name] = ch
        ch.subscribe(callback)

    @property
    def channel_names(self) -> list[str]:
        with self._lock:
            return list(self._channels.keys())
