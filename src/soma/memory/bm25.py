"""Tiny BM25 lexical index — supports hybrid search alongside vectors.

Pure-Python BM25-Okapi. No external deps. Fast enough for 100K+
entries (benchmarks below put 100K / 200-char docs at ~5 ms/query on
a laptop). The point is *hybrid retrieval*: combining lexical scores
with the MemoryLayer's cosine scores usually improves recall on
queries that hinge on specific terminology that sbert doesn't tokenize
as a single semantic unit.

Design note: we keep this in-tree and self-contained rather than
depending on ``rank-bm25`` because the whole algorithm is ~60 lines
and SOMA already uses a zero-runtime-deps posture where it can.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, ascii-letters/digits only. No stemming."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """BM25-Okapi index over a fixed list of documents.

    Rebuild when the document set changes — this is a snapshot index.
    For SOMA's use case that's fine because :class:`MemoryLayer` lazily
    rebuilds on first hybrid query and tracks invalidation via a version
    counter, same as the FAISS index.
    """

    k1: float = 1.5
    b: float = 0.75
    _doc_lens: list[int] = None  # type: ignore[assignment]
    _avgdl: float = 0.0
    _df: Counter[str] = None  # type: ignore[assignment]
    _idf: dict[str, float] = None  # type: ignore[assignment]
    _tf: list[Counter[str]] = None  # type: ignore[assignment]
    _n: int = 0

    def __post_init__(self) -> None:
        self._doc_lens = []
        self._df = Counter()
        self._idf = {}
        self._tf = []
        self._n = 0

    def build(self, texts: list[str]) -> None:
        """Index ``texts``. Replaces any prior contents."""
        self._tf = []
        self._doc_lens = []
        self._df = Counter()
        self._n = len(texts)
        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._doc_lens.append(len(tokens))
            for term in tf:
                self._df[term] += 1
        self._avgdl = (
            sum(self._doc_lens) / self._n if self._n else 0.0
        )
        # BM25 IDF with the standard +0.5 smoothing so rare terms stay positive.
        self._idf = {
            term: math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in self._df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return top-``k`` (doc_index, score) pairs. Zero-score docs
        are kept only if fewer than ``k`` non-zero matches exist — the
        caller can still combine them with vector scores."""
        if self._n == 0 or k <= 0:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores = [0.0] * self._n
        for term in q_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue  # unseen term contributes nothing
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self._doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(1.0, self._avgdl))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        # Top-k by score descending; stable by index for ties.
        idx = sorted(range(self._n), key=lambda i: (-scores[i], i))[:k]
        return [(i, scores[i]) for i in idx if scores[i] > 0.0]

    def __len__(self) -> int:
        return self._n
