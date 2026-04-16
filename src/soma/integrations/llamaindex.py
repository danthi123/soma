"""LlamaIndex retriever adapter for SOMA MemoryLayer.

Drop-in ``BaseRetriever`` that delegates to a :class:`MemoryLayer`
instance::

    from soma.memory import MemoryLayer
    from soma.integrations.llamaindex import SomaRetriever

    mem = MemoryLayer.with_sbert()
    mem.store("user lives in Portland")
    retriever = SomaRetriever(memory=mem, k=3)
    nodes = retriever.retrieve("where does the user live?")

Requires ``llama-index-core`` (optional dep).
"""

from __future__ import annotations

from typing import Any

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False

from soma.memory.api import MemoryLayer


def _check_llamaindex() -> None:
    if not _HAS_LLAMAINDEX:
        raise ImportError(
            "soma.integrations.llamaindex requires llama-index-core. "
            "Install with: pip install llama-index-core"
        )


if _HAS_LLAMAINDEX:

    class SomaRetriever(BaseRetriever):  # type: ignore[misc]
        """LlamaIndex retriever backed by a SOMA MemoryLayer.

        Pass through to :meth:`MemoryLayer.retrieve`: ``where``,
        ``hybrid_alpha``, and ``rerank_top_n`` all supported.
        """

        def __init__(
            self,
            memory: MemoryLayer,
            k: int = 5,
            *,
            where: dict[str, Any] | None = None,
            hybrid_alpha: float | None = None,
            rerank_top_n: int | None = None,
        ) -> None:
            super().__init__()
            self._memory = memory
            self._k = k
            self._where = where
            self._hybrid_alpha = hybrid_alpha
            self._rerank_top_n = rerank_top_n

        def _retrieve(self, query_bundle: QueryBundle, **kwargs: Any) -> list[NodeWithScore]:  # type: ignore[name-defined,override]
            rkwargs: dict[str, Any] = {}
            if self._where is not None:
                rkwargs["where"] = self._where
            if self._hybrid_alpha is not None:
                rkwargs["hybrid_alpha"] = self._hybrid_alpha
            if self._rerank_top_n is not None:
                rkwargs["rerank_top_n"] = self._rerank_top_n
            hits = self._memory.retrieve(query_bundle.query_str, k=self._k, **rkwargs)
            return [
                NodeWithScore(
                    node=TextNode(
                        text=hit.text,
                        metadata={
                            **hit.metadata,
                            "node_id": hit.node_id,
                            "timestamp_step": hit.timestamp_step,
                        },
                    ),
                    score=hit.score,
                )
                for hit in hits
            ]

else:

    class SomaRetriever:  # type: ignore[no-redef]
        """Stub that raises ImportError when llama-index-core is missing."""

        def __init__(self, **kwargs: Any) -> None:
            _check_llamaindex()
