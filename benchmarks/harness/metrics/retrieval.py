"""Standard retrieval metrics: Recall@k, MRR@k, NDCG@k."""

from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(retrieved_texts: Iterable[str], relevant_texts: list[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-k."""
    if not relevant_texts:
        return 0.0
    top = list(retrieved_texts)[:k]
    found = sum(1 for r in relevant_texts if r in top)
    return found / len(relevant_texts)


def mrr_at_k(retrieved_texts: Iterable[str], relevant_texts: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant hit in the top-k."""
    if not relevant_texts:
        return 0.0
    relevant_set = set(relevant_texts)
    for rank, text in enumerate(list(retrieved_texts)[:k], start=1):
        if text in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_texts: Iterable[str], relevant_texts: list[str], k: int) -> float:
    """Normalized discounted cumulative gain. Binary relevance: 1 if relevant, 0 else."""
    if not relevant_texts:
        return 0.0
    top = list(retrieved_texts)[:k]
    relevant_set = set(relevant_texts)
    dcg = sum(
        (1.0 / math.log2(rank + 1)) if text in relevant_set else 0.0
        for rank, text in enumerate(top, start=1)
    )
    ideal_hits = min(len(relevant_texts), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
