"""Tests for the Phase 3 distillation runner — scoring logic.

The actual end-to-end runs require Ollama + Chroma; those are
integration-tested ad-hoc. Here we cover the pure scoring path that
could otherwise silently produce wrong R@k numbers.
"""

from __future__ import annotations

import pytest

from benchmarks.datasets.locomo import CATEGORY_NAMES, LoCoMoQuery
from benchmarks.run_locomo_distill import _score_queries


def _mk_query(
    sample_id: int,
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


class TestScoreQueries:
    def test_perfect_recall_at_1(self) -> None:
        """Every query has its evidence as the top hit → R@1 = 1.0."""
        queries = [
            _mk_query(0, "q1", ["t1"]),
            _mk_query(0, "q2", ["t2"]),
        ]
        turn_id_to_step = {"t1": 0, "t2": 1}
        # Each query's top hit is its evidence step
        hits_list = [
            [(0, "t1 text", 0.9)],   # q1 → step 0
            [(1, "t2 text", 0.9)],   # q2 → step 1
        ]
        recall, by_cat = _score_queries(hits_list, queries, turn_id_to_step)
        assert recall[1] == 1.0
        assert recall[5] == 1.0

    def test_no_recall_when_hits_miss_evidence(self) -> None:
        queries = [_mk_query(0, "q1", ["t1"])]
        turn_id_to_step = {"t1": 0, "t99": 99}
        hits_list = [
            [(99, "wrong", 0.9)],   # wrong step
        ]
        recall, _ = _score_queries(hits_list, queries, turn_id_to_step)
        assert recall[1] == 0.0
        assert recall[5] == 0.0

    def test_recall_at_k_gates_by_position(self) -> None:
        queries = [_mk_query(0, "q1", ["t5"])]
        turn_id_to_step = {f"t{i}": i for i in range(10)}
        # Evidence is step 5 — it's at position 4 (index 4) in hit list
        hits_list = [
            [
                (0, "x", 0.9), (1, "x", 0.8), (2, "x", 0.7),
                (3, "x", 0.6), (5, "correct", 0.5),
            ],
        ]
        recall, _ = _score_queries(hits_list, queries, turn_id_to_step)
        assert recall[1] == 0.0   # Not in top 1
        assert recall[5] == 1.0   # In top 5
        assert recall[10] == 1.0  # In top 10

    def test_per_category_breakdown(self) -> None:
        queries = [
            _mk_query(0, "q1", ["t1"], category=1),  # single-hop
            _mk_query(0, "q2", ["t2"], category=2),  # multi-hop
            _mk_query(0, "q3", ["t3"], category=2),  # multi-hop, miss
        ]
        turn_id_to_step = {"t1": 0, "t2": 1, "t3": 2}
        hits_list = [
            [(0, "x", 0.9)],          # q1: hit
            [(1, "x", 0.9)],          # q2: hit
            [(99, "x", 0.9)],         # q3: miss
        ]
        _, by_cat = _score_queries(hits_list, queries, turn_id_to_step)
        assert by_cat["single-hop"][1] == 1.0    # 1/1 in single-hop
        assert by_cat["multi-hop"][1] == pytest.approx(0.5)  # 1/2 in multi-hop

    def test_evidence_not_in_corpus_is_not_credited(self) -> None:
        """Queries whose evidence turn was dropped from the corpus
        shouldn't crash — they count as "no possible hit" rather than
        erroring."""
        queries = [_mk_query(0, "q1", ["t_missing"])]
        turn_id_to_step = {"t1": 0}   # t_missing not in map
        hits_list = [[(0, "x", 0.9)]]
        recall, _ = _score_queries(hits_list, queries, turn_id_to_step)
        # Recall counted over all queries, including the one with no
        # matchable evidence — so R@1 = 0
        assert recall[1] == 0.0
