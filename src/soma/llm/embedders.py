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

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class CachedEmbedder:
    """Wraps any ``LLMTeacher`` with an on-disk + memory cache.

    Keys by ``sha256(text)``. Embeddings are stored as ``.pt`` files
    under ``<cache_dir>/<teacher-name-slug>/<hash>.pt``. Safe to share
    a cache dir across models — the name slug isolates them.

    Memory cache is kept per-instance; disk cache persists across runs
    so repeated benchmarks (e.g. LoCoMo's 5882 turns) don't re-hit
    the embedding model.
    """

    teacher: LLMTeacher
    cache_dir: str
    _mem_cache: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False,
    )
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"cached:{self.teacher.name}"
        slug = self.teacher.name.replace(":", "_").replace("/", "_")
        self._cache_path = Path(self.cache_dir) / slug
        self._cache_path.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_file(self, key: str) -> Path:
        return self._cache_path / f"{key}.pt"

    def embed(self, text: str) -> torch.Tensor:
        key = self._key(text)
        if key in self._mem_cache:
            return self._mem_cache[key]
        disk_file = self._cache_file(key)
        if disk_file.exists():
            tensor = torch.load(disk_file, map_location="cpu", weights_only=True)
            self._mem_cache[key] = tensor
            return tensor
        tensor = self.teacher.embed(text)
        self._mem_cache[key] = tensor
        torch.save(tensor, disk_file)
        return tensor

    def embed_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.embed(t) for t in texts])
