"""Tests for the Phase 3 distillation runner — scoring logic.

The actual end-to-end runs require Ollama + Chroma; those are
integration-tested ad-hoc. Here we cover the pure scoring path that
could otherwise silently produce wrong R@k numbers.
"""

from __future__ import annotations

import pytest

from benchmarks.datasets.locomo import LoCoMoQuery
from benchmarks.run_locomo_distill import (
    _aggregate_recall,
    _score_sample_queries,
)


def _mk_query(
    sample_id: str,
    q_text: str,
    evidence: list[str],
    category: int = 1,
) -> LoCoMoQuery:
    return LoCoMoQuery(
        sample_id=sample_id,
        question=q_text,
        answer="",
        evidence=evidence,
        category=category,
    )


class TestScoreSampleQueries:
    def test_perfect_recall_at_1(self) -> None:
        """Every query has its evidence as the top hit → R@1 = 1.0."""
        queries = [
            _mk_query("s1", "q1", ["D1:1"]),
            _mk_query("s1", "q2", ["D1:2"]),
        ]
        # retrieved_per_q: for each query, the list of retrieved dia_ids
        hits = [
            ["D1:1", "D1:2", "D1:3"],
            ["D1:2", "D1:3", "D1:4"],
        ]
        hit_counts, totals, cat_hits, cat_totals = _score_sample_queries(queries, hits)
        assert hit_counts[1] == 2
        assert hit_counts[5] == 2
        assert totals[1] == 2
        assert totals[5] == 2
        assert cat_totals["single-hop"] == 2
        assert cat_hits["single-hop"][1] == 2

    def test_no_recall_when_hits_miss_evidence(self) -> None:
        queries = [_mk_query("s1", "q1", ["D1:1"])]
        hits = [["D1:99", "D1:98"]]
        hit_counts, totals, _, _ = _score_sample_queries(queries, hits)
        assert hit_counts[1] == 0
        assert hit_counts[5] == 0
        assert totals[1] == 1

    def test_recall_at_k_gates_by_position(self) -> None:
        queries = [_mk_query("s1", "q1", ["D1:5"])]
        # Evidence is at position 4 (index 4) in hit list
        hits = [["D1:0", "D1:1", "D1:2", "D1:3", "D1:5"]]
        hit_counts, _, _, _ = _score_sample_queries(queries, hits)
        assert hit_counts[1] == 0
        assert hit_counts[5] == 1
        assert hit_counts[10] == 1

    def test_per_category_breakdown(self) -> None:
        queries = [
            _mk_query("s1", "q1", ["D1:1"], category=1),   # single-hop
            _mk_query("s1", "q2", ["D1:2"], category=2),   # multi-hop
            _mk_query("s1", "q3", ["D1:3"], category=2),   # multi-hop
        ]
        hits = [
            ["D1:1"],       # q1: hit
            ["D1:2"],       # q2: hit
            ["D1:99"],      # q3: miss
        ]
        _, _, cat_hits, cat_totals = _score_sample_queries(queries, hits)
        assert cat_totals["single-hop"] == 1
        assert cat_hits["single-hop"][1] == 1
        assert cat_totals["multi-hop"] == 2
        assert cat_hits["multi-hop"][1] == 1


class TestAggregateRecall:
    def test_single_sample_passes_through(self) -> None:
        queries = [_mk_query("s1", "q1", ["D1:1"])]
        hits = [["D1:1"]]
        per_sample = [_score_sample_queries(queries, hits)]
        recall, _ = _aggregate_recall(per_sample)
        assert recall[1] == 1.0

    def test_multiple_samples_sum_correctly(self) -> None:
        # Sample 1: 1/1 hit
        s1_queries = [_mk_query("s1", "q1", ["D1:1"])]
        s1_hits = [["D1:1"]]
        # Sample 2: 0/2 hit
        s2_queries = [
            _mk_query("s2", "q1", ["D1:1"]),
            _mk_query("s2", "q2", ["D1:2"]),
        ]
        s2_hits = [["D1:99"], ["D1:99"]]
        per_sample = [
            _score_sample_queries(s1_queries, s1_hits),
            _score_sample_queries(s2_queries, s2_hits),
        ]
        recall, _ = _aggregate_recall(per_sample)
        # 1/3 total = 0.333
        assert recall[1] == pytest.approx(1 / 3, abs=1e-6)
