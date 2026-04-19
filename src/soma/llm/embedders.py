"""Teacher embedders for LLM-distilled projections (Direction 4a).

``LLMTeacher`` defines the minimal protocol. ``OllamaEmbedder`` is the
concrete adapter for a locally-running Ollama server. ``CachedEmbedder``
(added separately) wraps any teacher with an on-disk cache.

Why split from ``backends.py``: the generative/chat backends there
produce text completions; these classes return dense vectors. Different
protocol, different endpoints, different use sites — the separation
keeps each file small and single-purpose.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LLMTeacher(Protocol):
    """Minimal protocol: embed one string or a batch of strings."""

    name: str

    def embed(self, text: str) -> torch.Tensor: ...

    def embed_batch(self, texts: list[str]) -> torch.Tensor: ...


@dataclass
class OllamaEmbedder:
    """Calls Ollama's ``POST /api/embeddings`` endpoint.

    Returns a 1-D ``torch.Tensor`` (float32, shape ``(dim,)``) for
    single inputs and a 2-D tensor (shape ``(B, dim)``) for batches.
    The embedding dimension depends on the model:
    ``mxbai-embed-large`` → 1024, ``nomic-embed-text`` → 768.
    """

    model: str = "mxbai-embed-large"
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"ollama-embed:{self.model}"

    def embed(self, text: str) -> torch.Tensor:
        body = json.dumps(
            {"model": self.model, "prompt": text}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embeddings server unreachable at {self.base_url}: "
                f"{exc}. Install Ollama and run `ollama serve`, then "
                f"`ollama pull {self.model}`."
            ) from exc
        vec = payload.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError(
                f"Ollama returned malformed embedding payload: {payload!r}"
            )
        return torch.tensor(vec, dtype=torch.float32)

    def embed_batch(self, texts: list[str]) -> torch.Tensor:
        # Ollama's current /api/embeddings is one-at-a-time. Future
        # optimization: newer /api/embed supports batch. For now the
        # caching layer (CachedEmbedder) amortizes the HTTP cost across
        # repeat runs.
        return torch.stack([self.embed(t) for t in texts])
