"""Chroma + default-embedder adapter (the "plain RAG" baseline)."""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .base import BaseMemorySystem, BenchmarkHit


class ChromaAdapter(BaseMemorySystem):
    """Wraps chromadb's PersistentClient with its default embedder."""

    name = "chroma"

    def __init__(self, collection_name: str = "bench") -> None:
        self._collection_name = collection_name
        self._client: Any = None
        self._col: Any = None
        self._persist_dir: Path | None = None

    def prepare(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaAdapter requires chromadb. Install: pip install chromadb"
            ) from exc
        self._persist_dir = Path(tempfile.mkdtemp(prefix="chroma-bench-"))
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._col = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        assert self._col is not None
        nid = uuid.uuid4().hex
        self._col.add(
            ids=[nid],
            documents=[text],
            metadatas=[metadata] if metadata else None,
        )
        return nid

    def store_with_embedding(
        self,
        text: str,
        embedding: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Use Chroma's direct-embedding path so it skips its internal
        embed model. Same chromadb API, just providing the vector."""
        assert self._col is not None
        nid = uuid.uuid4().hex
        emb = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        self._col.add(
            ids=[nid],
            documents=[text],
            embeddings=[emb],
            metadatas=[metadata] if metadata else None,
        )
        return nid

    def retrieve(self, query: str, k: int = 5) -> list[BenchmarkHit]:
        assert self._col is not None
        n = min(k, self._col.count())
        if n <= 0:
            return []
        result = self._col.query(query_texts=[query], n_results=n)
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0] or [{}] * len(docs)
        hits = []
        for doc, dist, nid, meta in zip(docs, distances, ids, metadatas, strict=False):
            hits.append(
                BenchmarkHit(
                    text=doc,
                    score=1.0 - float(dist),  # cosine distance → similarity
                    metadata=dict(meta) if meta else {},
                    node_id=str(nid),
                )
            )
        return hits

    def consolidate(self) -> None:
        return None

    def clear(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.delete_collection(self._collection_name)
            self._col = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def disk_footprint_bytes(self) -> int:
        if self._persist_dir is None or not self._persist_dir.exists():
            return 0
        return sum(
            f.stat().st_size for f in self._persist_dir.rglob("*") if f.is_file()
        )

    def teardown(self) -> None:
        self._col = None
        self._client = None
        if self._persist_dir is not None:
            shutil.rmtree(self._persist_dir, ignore_errors=True)
            self._persist_dir = None
