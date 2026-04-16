"""LangChain retriever adapter for SOMA MemoryLayer.

Drop-in ``BaseRetriever`` that delegates to a :class:`MemoryLayer`
instance. Lets LangChain chains and agents use SOMA as their retrieval
backend with zero glue code::

    from soma.memory import MemoryLayer
    from soma.integrations.langchain import SomaRetriever

    mem = MemoryLayer.with_sbert()
    mem.store("user lives in Portland")
    retriever = SomaRetriever(memory=mem, k=3)

    # Use in any LangChain chain:
    docs = retriever.invoke("where does the user live?")

Requires ``langchain-core`` (optional dep).
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

def _check_langchain() -> None:
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "soma.integrations.langchain requires langchain-core. "
            "Install with: pip install langchain-core"
        )


if _HAS_LANGCHAIN:

    class SomaRetriever(BaseRetriever):  # type: ignore[misc]
        """LangChain retriever backed by a SOMA MemoryLayer.

        Parameters
        ----------
        memory : MemoryLayer
            The memory layer to search.
        k : int
            Number of results to return per query (default 5).
        """

        memory: Any  # MemoryLayer — typed Any to satisfy Pydantic v2
        k: int = 5
        model_config = {"arbitrary_types_allowed": True}

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,  # type: ignore[name-defined]
        ) -> list[Document]:  # type: ignore[name-defined]
            hits = self.memory.retrieve(query, k=self.k)
            return [
                Document(
                    page_content=hit.text,
                    metadata={
                        **hit.metadata,
                        "node_id": hit.node_id,
                        "score": hit.score,
                        "timestamp_step": hit.timestamp_step,
                    },
                )
                for hit in hits
            ]

else:

    class SomaRetriever:  # type: ignore[no-redef]
        """Stub that raises ImportError when langchain-core is missing."""

        def __init__(self, **kwargs: Any) -> None:
            _check_langchain()
