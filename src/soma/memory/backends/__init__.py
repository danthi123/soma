"""Vector-backend adapters shipped with SOMA.

The default adapter is :class:`soma.memory.backends.inproc.InProcBackend`
— pure in-process, lazy FAISS build, zero external deps beyond what
SOMA already ships. Additional adapters (Qdrant, LanceDB, ...) live
alongside it and are imported on demand.

``QdrantBackend`` is an optional dep (``pip install soma[qdrant]``);
``LanceDBBackend`` is an optional dep (``pip install soma[lancedb]``).
Both are re-exported from this module when the underlying dep is
installed and silently skipped otherwise so SOMA still imports on
bare environments.
"""

from __future__ import annotations

from soma.memory.backends.inproc import InProcBackend

__all__ = ["InProcBackend"]

try:
    from soma.memory.backends.qdrant import QdrantBackend  # noqa: F401

    __all__.append("QdrantBackend")
except ImportError:  # pragma: no cover — optional dep
    pass

try:
    from soma.memory.backends.lancedb import LanceDBBackend  # noqa: F401

    __all__.append("LanceDBBackend")
except ImportError:  # pragma: no cover — optional dep
    pass
