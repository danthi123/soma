"""SOMA MemoryLayer adapter for the benchmark harness."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from soma.memory import MemoryLayer

from .base import BaseMemorySystem, BenchmarkHit


class SomaAdapter(BaseMemorySystem):
    """Wraps ``soma.memory.MemoryLayer``.

    By default uses sentence-transformers for embeddings (strongest
    baseline; same as Chroma's default). Set ``use_sbert=False`` to fall
    back to SOMA's TextEncoder for fully-offline/tiny footprint runs.
    """

    name = "soma"

    def __init__(
        self,
        *,
        use_sbert: bool = True,
        attach_soma: bool = False,
        embed_model: str = "all-MiniLM-L6-v2",
        auto_consolidate_every: int = 0,
        graph_rerank_alpha: float = 0.0,
        graph_rerank_stable_capture: bool = True,
        faiss_index_type: str = "flat",
        faiss_threshold: int = 10_000,
    ) -> None:
        self._use_sbert = use_sbert
        self._attach_soma = attach_soma
        self._embed_model = embed_model
        self._auto_consolidate_every = auto_consolidate_every
        self._graph_rerank_alpha = graph_rerank_alpha
        self._graph_rerank_stable_capture = graph_rerank_stable_capture
        self._faiss_index_type = faiss_index_type
        self._faiss_threshold = faiss_threshold
        self._mem: MemoryLayer | None = None
        self._bundle_path: Path | None = None

    def prepare(self) -> None:
        if self._use_sbert:
            self._mem = MemoryLayer.with_sbert(self._embed_model)
            self._mem._auto_consolidate_every = self._auto_consolidate_every
            self._mem._graph_rerank_alpha = self._graph_rerank_alpha
            self._mem._graph_rerank_stable_capture = self._graph_rerank_stable_capture
            self._mem._faiss_index_type = self._faiss_index_type
            self._mem._faiss_threshold = self._faiss_threshold
        else:
            from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

            tokenizer = train_bpe_tokenizer(["placeholder"], vocab_size=128)
            encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=128)
            self._mem = MemoryLayer(
                tokenizer=tokenizer,
                encoder=encoder,
                auto_consolidate_every=self._auto_consolidate_every,
                graph_rerank_alpha=self._graph_rerank_alpha,
                graph_rerank_stable_capture=self._graph_rerank_stable_capture,
                faiss_index_type=self._faiss_index_type,
                faiss_threshold=self._faiss_threshold,
            )
        if self._attach_soma:
            from soma.core.config import SOMAConfig
            from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
            from soma.system import SOMA

            config = SOMAConfig(
                vocab_size=128,
                text_embed_dim=32,
                sensor_output_dim=32,
                max_input_tokens=128,
            )
            soma = SOMA(config)
            # SOMA's graph operates on its own small TextEncoder regardless
            # of what the MemoryLayer embeds with for cosine. Keeping them
            # independent lets sbert (384-d) drive retrieval while the 32-d
            # SOMA substrate drives graph-based re-ranking.
            if self._use_sbert:
                soma_tokenizer = train_bpe_tokenizer(["placeholder"], vocab_size=128)
                soma_encoder = TextEncoder(
                    soma_tokenizer, embed_dim=32, max_seq_len=128,
                )
            else:
                assert self._mem._encoder is not None
                soma_tokenizer = self._mem._tokenizer
                soma_encoder = self._mem._encoder
            self._mem.attach_soma(soma, soma_tokenizer, soma_encoder)

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        assert self._mem is not None
        return self._mem.store(text, metadata=metadata)

    def store_with_embedding(
        self,
        text: str,
        embedding: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Skip the per-call embed by injecting the precomputed vector
        through MemoryLayer's internals — we set the custom embed_fn to
        a closure that returns this exact vector for this exact text."""
        assert self._mem is not None
        import torch

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
        # The MemoryLayer's _embed() routes through _custom_embed_fn when
        # set. Temporarily override for this single store; the sbert
        # path remains the default for retrieve.
        prev_fn = self._mem._custom_embed_fn
        self._mem._custom_embed_fn = lambda _t: embedding
        try:
            return self._mem.store(text, metadata=metadata)
        finally:
            self._mem._custom_embed_fn = prev_fn

    def retrieve(self, query: str, k: int = 5) -> list[BenchmarkHit]:
        assert self._mem is not None
        hits = self._mem.retrieve(query, k=k)
        return [
            BenchmarkHit(
                text=h.text, score=h.score, metadata=h.metadata, node_id=h.node_id,
            )
            for h in hits
        ]

    def consolidate(self) -> None:
        assert self._mem is not None
        self._mem.consolidate()

    def clear(self) -> None:
        self._mem = None
        if self._bundle_path and self._bundle_path.exists():
            shutil.rmtree(self._bundle_path, ignore_errors=True)
            self._bundle_path = None

    def disk_footprint_bytes(self) -> int:
        if self._mem is None:
            return 0
        self._bundle_path = Path(tempfile.mkdtemp()) / "mem"
        self._mem.save(self._bundle_path)
        return sum(
            f.stat().st_size for f in self._bundle_path.rglob("*") if f.is_file()
        )

    def teardown(self) -> None:
        self.clear()
