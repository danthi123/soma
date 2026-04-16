"""Vector-backend adapters shipped with SOMA.

The default adapter is :class:`soma.memory.backends.inproc.InProcBackend`
— pure in-process, lazy FAISS build, zero external deps beyond what
SOMA already ships. Additional adapters (Qdrant, LanceDB, ...) live
alongside it and are imported on demand.
"""

from __future__ import annotations

from soma.memory.backends.inproc import InProcBackend

__all__ = ["InProcBackend"]
