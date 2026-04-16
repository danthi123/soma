"""Base adapter interface for memory systems.

Every benchmark-testable system (SOMA MemoryLayer, Chroma+RAG, Mem0,
Zep, etc.) implements :class:`BaseMemorySystem`. The harness calls
the same methods on each adapter so comparisons are apples-to-apples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkHit:
    """One retrieved entry, normalized across systems."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    node_id: str = ""


class BaseMemorySystem(ABC):
    """Common interface all benchmarked systems expose.

    Adapters must be cheap to instantiate (no benchmark timing should
    include setup cost unless measured separately). Heavy init (model
    download, index build) should happen in :meth:`prepare`.
    """

    name: str = "base"

    @abstractmethod
    def prepare(self) -> None:
        """One-time setup before any store/retrieve."""

    @abstractmethod
    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Persist an entry. Returns a stable node_id."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[BenchmarkHit]:
        """Return up to k entries most similar to ``query``."""

    @abstractmethod
    def consolidate(self) -> None:
        """Trigger any maintenance the system exposes (may be a no-op)."""

    @abstractmethod
    def clear(self) -> None:
        """Reset the store so the adapter can be reused for multiple runs."""

    def teardown(self) -> None:
        """Release any resources (temp dirs, clients). Default: no-op."""
        return None

    def disk_footprint_bytes(self) -> int:
        """Best-effort on-disk size. Default: 0 (in-memory only)."""
        return 0
