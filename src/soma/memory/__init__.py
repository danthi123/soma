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
    - :class:`ConversationalMemory`: Mem0/Zep-style wrapper that adds
      LLM-driven fact extraction, reconciliation, and rolling session
      summaries on top of a :class:`MemoryLayer`. Raw store/retrieve
      API stays unchanged.
"""

from soma.memory.api import MemoryHit, MemoryLayer
from soma.memory.conversational import ConversationalMemory, ExtractedFact
from soma.memory.episodic_memory import EpisodicMemory, Retrieval
from soma.memory.working_memory import WorkingMemory

__all__ = [
    "ConversationalMemory",
    "EpisodicMemory",
    "ExtractedFact",
    "MemoryHit",
    "MemoryLayer",
    "Retrieval",
    "WorkingMemory",
]
