"""SOMA memory subsystems.

Internal substrate:
    - :class:`WorkingMemory`: transient slot-based context buffer.
    - :class:`EpisodicMemory`: fast, content-addressable one-shot store
      over tensors.

Public API (see ``docs/positioning.md``):
    - :class:`MemoryLayer`: vector-DB-shaped interface agent developers
      interact with. Wraps the substrate above and exposes a
      ``store``/``retrieve`` surface that can be dropped in where a
      vector store would otherwise go.
"""

from soma.memory.api import MemoryHit, MemoryLayer
from soma.memory.episodic_memory import EpisodicMemory, Retrieval
from soma.memory.working_memory import WorkingMemory

__all__ = [
    "EpisodicMemory",
    "MemoryHit",
    "MemoryLayer",
    "Retrieval",
    "WorkingMemory",
]
