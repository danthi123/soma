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
    ) -> None:
        self._use_sbert = use_sbert
        self._attach_soma = attach_soma
        self._embed_model = embed_model
        self._auto_consolidate_every = auto_consolidate_every
        self._mem: MemoryLayer | None = None
        self._bundle_path: Path | None = None

    def prepare(self) -> None:
        if self._use_sbert:
            self._mem = MemoryLayer.with_sbert(self._embed_model)
            self._mem._auto_consolidate_every = self._auto_consolidate_every
        else:
            from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

            tokenizer = train_bpe_tokenizer(["placeholder"], vocab_size=128)
            encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=128)
            self._mem = MemoryLayer(
                tokenizer=tokenizer,
                encoder=encoder,
                auto_consolidate_every=self._auto_consolidate_every,
            )
        if self._attach_soma:
            from soma.core.config import SOMAConfig
            from soma.system import SOMA

            if self._use_sbert:
                raise ValueError(
                    "attach_soma=True requires use_sbert=False so the "
                    "TextEncoder feeding SOMA matches the configured SENSOR dim"
                )
            config = SOMAConfig(
                vocab_size=128,
                text_embed_dim=32,
                sensor_output_dim=32,
                max_input_tokens=128,
            )
            soma = SOMA(config)
            assert self._mem._encoder is not None
            self._mem.attach_soma(soma, self._mem._tokenizer, self._mem._encoder)

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        assert self._mem is not None
        return self._mem.store(text, metadata=metadata)

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
