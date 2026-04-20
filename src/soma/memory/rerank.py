"""Cross-encoder re-ranking for SOMA retrievals.

Vector retrieval (bi-encoder cosine) is fast but coarse — it scores
each document against the query *independently*. Cross-encoder
re-ranking scores (query, document) pairs *jointly* so the model can
attend across both. The cost is latency (a small transformer pass per
candidate) but the recall lift on real queries is usually large —
BEIR-style evaluations routinely show +5-15% Recall@5 over pure
cosine.

Pattern: over-fetch top-N with cheap cosine, re-rank those N with
a cross-encoder, return top-k of the re-ranked list. ``N`` should be
3-5× ``k``.

Requires ``sentence-transformers`` (same optional dep as
:meth:`MemoryLayer.with_sbert`). The default model,
``cross-encoder/ms-marco-MiniLM-L-6-v2``, is 23 MB and runs in <10 ms
per candidate on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Minimal protocol: score (query, candidate_texts) → list[float]."""

    name: str = "base"

    def score(self, query: str, candidates: list[str]) -> list[float]:
        raise NotImplementedError


@dataclass
class CrossEncoderReranker(Reranker):
    """Sentence-transformers CrossEncoder wrapper.

    Lazy-loads the model on first ``score()`` call so construction is
    cheap and offline-friendly when the cross-encoder isn't used.
    """

    model_name: str = DEFAULT_MODEL
    device: str | None = None
    batch_size: int = 32
    _model: Any = None

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"cross-encoder:{self.model_name}"

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker needs sentence-transformers. "
                "Install: pip install 'soma-memory[sbert]'"
            ) from exc
        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        raw = self._load().predict(pairs, batch_size=self.batch_size)
        # predict() returns np.ndarray or list; normalise to list[float]
        return [float(x) for x in raw]


@dataclass
class StubReranker(Reranker):
    """Deterministic scoring by token-overlap — for tests only."""

    name: str = "stub-token-overlap"

    def score(self, query: str, candidates: list[str]) -> list[float]:
        q_tokens = set(query.lower().split())
        out = []
        for c in candidates:
            c_tokens = set(c.lower().split())
            denom = max(1, len(q_tokens | c_tokens))
            out.append(len(q_tokens & c_tokens) / denom)
        return out
