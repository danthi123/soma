"""A fixed-capacity circular buffer for scalar float statistics.

Used by:
- ``Node.activation_history`` — recent activation magnitudes per node.
- ``CuriosityModule.error_histories`` — prediction errors per domain.
- Any other module that needs a bounded rolling window of scalars.

Design notes:
- Stores plain Python floats (not tensors) — the consumers always call
  ``.item()`` on their tensor inputs before appending, and reads are for
  CPU-side statistics (means, progress metrics).
- Not an ``nn.Module``: holds no learnable parameters. Node/Edge modules
  own their ring buffers as attributes. Serialization is explicit via
  ``to_list``/``from_list``.
- Fixed capacity avoids unbounded memory growth over long runs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence


class RingBuffer:
    """Fixed-size circular buffer over float scalars.

    Newest element is at the end when iterated via ``get_all()``. Once the
    buffer is full, each append drops the oldest element.
    """

    __slots__ = ("_buffer", "_capacity", "_write_index", "_size")

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"RingBuffer capacity must be a positive int, got {capacity!r}")
        self._capacity: int = capacity
        # Pre-allocate to avoid list growth cost; unused slots hold 0.0.
        self._buffer: list[float] = [0.0] * capacity
        self._write_index: int = 0
        self._size: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def append(self, value: float) -> None:
        """Append ``value`` to the buffer, evicting the oldest if full."""
        self._buffer[self._write_index] = float(value)
        self._write_index = (self._write_index + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def extend(self, values: Sequence[float]) -> None:
        """Append each element of ``values`` in order."""
        for v in values:
            self.append(v)

    def clear(self) -> None:
        """Reset the buffer to empty (capacity unchanged)."""
        self._write_index = 0
        self._size = 0
        for i in range(self._capacity):
            self._buffer[i] = 0.0

    # ------------------------------------------------------------------
    # Read views
    # ------------------------------------------------------------------
    def get_all(self) -> list[float]:
        """Return a new list of entries in insertion order (oldest first)."""
        if self._size < self._capacity:
            # Buffer not yet wrapped — entries sit in [0, size).
            return self._buffer[: self._size]
        # Wrapped: oldest entry is at ``_write_index``.
        return self._buffer[self._write_index :] + self._buffer[: self._write_index]

    def last(self) -> float:
        """Return the most recently appended value. Raises if empty."""
        if self._size == 0:
            raise IndexError("RingBuffer is empty")
        last_idx = (self._write_index - 1) % self._capacity
        return self._buffer[last_idx]

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------
    def mean(self) -> float:
        """Mean of stored values; returns 0.0 if empty."""
        if self._size == 0:
            return 0.0
        total = 0.0
        for v in self._iter_stored():
            total += v
        return total / self._size

    def variance(self) -> float:
        """Population variance of stored values; 0.0 if fewer than 2 entries."""
        if self._size < 2:
            return 0.0
        mu = self.mean()
        total_sq = 0.0
        for v in self._iter_stored():
            diff = v - mu
            total_sq += diff * diff
        return total_sq / self._size

    def max(self) -> float:
        """Max of stored values; raises if empty."""
        if self._size == 0:
            raise ValueError("RingBuffer is empty; max undefined")
        return max(self._iter_stored())

    def min(self) -> float:
        """Min of stored values; raises if empty."""
        if self._size == 0:
            raise ValueError("RingBuffer is empty; min undefined")
        return min(self._iter_stored())

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[float]:
        return iter(self.get_all())

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, int | float):
            return False
        return any(v == value for v in self._iter_stored())

    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self._capacity}, size={self._size}, values={self.get_all()!r})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return self._size == self._capacity

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_list(self) -> list[float]:
        """Alias of ``get_all()``; provided for serialization clarity."""
        return self.get_all()

    @classmethod
    def from_list(cls, values: Sequence[float], capacity: int | None = None) -> RingBuffer:
        """Build a buffer from a list of values.

        If ``capacity`` is omitted, it defaults to ``len(values)`` (must be > 0).
        If ``capacity`` is smaller than ``len(values)``, the oldest entries
        (front of the list) are dropped to fit.
        """
        if capacity is None:
            if len(values) == 0:
                raise ValueError("Cannot infer capacity from empty values; pass capacity=...")
            capacity = len(values)
        rb = cls(capacity)
        # If values exceed capacity, only the last ``capacity`` entries survive.
        tail = values[-capacity:] if len(values) > capacity else values
        for v in tail:
            rb.append(v)
        return rb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _iter_stored(self) -> Iterator[float]:
        """Iterate over only the stored entries (avoids allocating a list)."""
        if self._size < self._capacity:
            for i in range(self._size):
                yield self._buffer[i]
        else:
            # Oldest first, wrapped.
            start = self._write_index
            for i in range(self._capacity):
                yield self._buffer[(start + i) % self._capacity]
