"""Tests for the benchmark harness contract."""

from __future__ import annotations

from typing import Any

from benchmarks.harness.adapters.base import BaseMemorySystem, BenchmarkHit
from benchmarks.harness.metrics.retrieval import mrr_at_k, ndcg_at_k, recall_at_k
from benchmarks.harness.runner import LabeledQuery, run_retrieval_benchmark


class MockAdapter(BaseMemorySystem):
    """Trivial in-memory adapter for harness tests."""

    name = "mock"

    def __init__(self) -> None:
        self._entries: list[tuple[str, str, dict[str, Any]]] = []

    def prepare(self) -> None:
        self._entries = []

    def store(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        nid = f"id-{len(self._entries)}"
        self._entries.append((nid, text, metadata or {}))
        return nid

    def retrieve(self, query: str, k: int = 5) -> list[BenchmarkHit]:
        # Naive: return entries containing any query token
        tokens = query.lower().split()
        scored = []
        for nid, text, meta in self._entries:
            score = sum(1 for t in tokens if t in text.lower()) / max(len(tokens), 1)
            if score > 0:
                scored.append(BenchmarkHit(text=text, score=score, metadata=meta, node_id=nid))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def consolidate(self) -> None:
        return None

    def clear(self) -> None:
        self._entries = []


def test_harness_runs_mock_adapter() -> None:
    adapter = MockAdapter()
    dataset = ["the cat sat on the mat", "the dog ran fast", "quantum physics is hard"]
    queries = [LabeledQuery(query="cat on mat", relevant_texts=["the cat sat on the mat"])]
    results = run_retrieval_benchmark(systems=[adapter], dataset=dataset, queries=queries, k=3)
    assert len(results) == 1
    assert results[0].system == "mock"
    assert results[0].num_entries == 3
    assert results[0].recall_at_k == 1.0


def test_recall_metric() -> None:
    assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["d"], k=3) == 0.0
    assert recall_at_k(["a", "b", "c"], ["b", "c"], k=2) == 0.5


def test_mrr_metric() -> None:
    assert mrr_at_k(["a", "b", "c"], ["b"], k=3) == 0.5
    assert mrr_at_k(["a", "b", "c"], ["a"], k=3) == 1.0
    assert mrr_at_k(["a", "b", "c"], ["d"], k=3) == 0.0


def test_ndcg_metric() -> None:
    assert ndcg_at_k(["a", "b"], ["a"], k=2) == 1.0
    assert ndcg_at_k([], ["a"], k=2) == 0.0
    assert ndcg_at_k(["x"], ["a"], k=2) == 0.0
