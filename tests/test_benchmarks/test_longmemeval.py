"""Tests for the LongMemEval benchmark adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.industry.longmemeval.data_loader import (
    LongMemEvalItem,
    Turn,
    load_dataset,
)
from benchmarks.industry.longmemeval.evaluate_baseline import (
    _format_history,
)
from benchmarks.industry.longmemeval.evaluate_soma import (
    _build_memory_layer,
    ingest_conversations,
)
from benchmarks.industry.longmemeval.metrics import (
    compute_scores,
    exact_match,
    rouge_l,
    token_f1,
)

DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "benchmarks"
    / "industry"
    / "longmemeval"
    / "data"
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_item(
    question_id: str = "test_001",
    question_type: str = "single-session-user",
    question: str = "What is my favorite color?",
    answer: str = "blue",
    num_sessions: int = 2,
    turns_per_session: int = 4,
) -> LongMemEvalItem:
    sessions: list[list[Turn]] = []
    for s in range(num_sessions):
        turns: list[Turn] = []
        for t in range(turns_per_session):
            role = "user" if t % 2 == 0 else "assistant"
            content = f"Session {s} turn {t} content."
            if s == 0 and t == 0:
                content = "My favorite color is blue."
            turns.append(Turn(role=role, content=content, has_answer=(s == 0 and t == 0)))
        sessions.append(turns)

    return LongMemEvalItem(
        question_id=question_id,
        question_type=question_type,
        question=question,
        answer=answer,
        question_date="2023/04/10 (Mon) 23:07",
        haystack_dates=[f"2023/04/{10 + s} (Mon) 14:00" for s in range(num_sessions)],
        haystack_session_ids=[f"sess_{s}" for s in range(num_sessions)],
        haystack_sessions=sessions,
        answer_session_ids=["sess_0"],
    )


# ── Metrics tests ─────────────────────────────────────────────────────


class TestTokenF1:
    def test_exact(self) -> None:
        assert token_f1("blue", "blue") == 1.0

    def test_partial(self) -> None:
        score = token_f1("the color blue", "blue")
        assert 0.0 < score < 1.0

    def test_no_overlap(self) -> None:
        assert token_f1("red", "blue") == 0.0

    def test_empty(self) -> None:
        assert token_f1("", "") == 1.0
        assert token_f1("hello", "") == 0.0
        assert token_f1("", "hello") == 0.0


class TestExactMatch:
    def test_match(self) -> None:
        assert exact_match("Blue", "blue") == 1.0

    def test_no_match(self) -> None:
        assert exact_match("red", "blue") == 0.0

    def test_punctuation_ignored(self) -> None:
        assert exact_match("blue!", "blue") == 1.0


class TestRougeL:
    def test_identical(self) -> None:
        assert rouge_l("the quick brown fox", "the quick brown fox") == 1.0

    def test_partial(self) -> None:
        score = rouge_l("the quick fox", "the quick brown fox")
        assert 0.0 < score < 1.0

    def test_empty(self) -> None:
        assert rouge_l("", "") == 1.0


class TestComputeScores:
    def test_basic(self) -> None:
        preds = [
            {"question_id": "q1", "hypothesis": "blue"},
            {"question_id": "q2", "hypothesis": "paris"},
        ]
        refs = [
            {"question_id": "q1", "answer": "blue", "question_type": "single-session-user"},
            {"question_id": "q2", "answer": "paris", "question_type": "multi-session"},
        ]
        scores = compute_scores(preds, refs)
        assert scores.f1 == 1.0
        assert scores.em == 1.0
        assert scores.n_total == 2
        assert "single-session-user" in scores.per_type
        assert "multi-session" in scores.per_type

    def test_missing_prediction(self) -> None:
        preds = [{"question_id": "q1", "hypothesis": "blue"}]
        refs = [
            {"question_id": "q1", "answer": "blue", "question_type": "x"},
            {"question_id": "q2", "answer": "red", "question_type": "x"},
        ]
        scores = compute_scores(preds, refs)
        assert scores.n_total == 1


# ── Data loader tests ─────────────────────────────────────────────────


class TestDataLoader:
    @pytest.mark.skipif(
        not (DATA_DIR / "longmemeval_oracle.json").exists(),
        reason="LongMemEval oracle data not downloaded",
    )
    def test_load_oracle(self) -> None:
        items = load_dataset("oracle", limit=5)
        assert len(items) == 5
        item = items[0]
        assert item.question_id
        assert item.question_type
        assert item.question
        assert item.answer
        assert item.num_sessions > 0
        assert item.total_turns > 0

    @pytest.mark.skipif(
        not (DATA_DIR / "longmemeval_oracle.json").exists(),
        reason="LongMemEval oracle data not downloaded",
    )
    def test_question_types(self) -> None:
        items = load_dataset("oracle")
        types = {item.question_type for item in items}
        assert "temporal-reasoning" in types
        assert "multi-session" in types

    def test_load_missing_variant(self) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            load_dataset("nonexistent")

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset("oracle", data_dir="/tmp/does_not_exist_longmemeval")


# ── SOMA adapter tests ────────────────────────────────────────────────


class TestSomaAdapter:
    def test_ingest_conversations(self) -> None:
        item = _make_item(num_sessions=2, turns_per_session=3)
        mem = _build_memory_layer()
        ingest_conversations(mem, item)
        assert len(mem._ids) == 6  # 2 sessions x 3 turns

    def test_ingest_and_retrieve(self) -> None:
        item = _make_item()
        mem = _build_memory_layer()
        ingest_conversations(mem, item)
        hits = mem.retrieve("favorite color", k=5)
        assert len(hits) > 0
        # Stub embedder is hash-based so ordering is non-semantic;
        # just verify we get real text back from the stored turns.
        assert all(h.text for h in hits)


# ── Baseline adapter tests ───────────────────────────────────────────


class TestBaselineAdapter:
    def test_format_history(self) -> None:
        item = _make_item(num_sessions=2, turns_per_session=2)
        history = _format_history(item, max_chars=10000)
        assert "Session" in history
        assert "blue" in history

    def test_format_history_truncation(self) -> None:
        item = _make_item(num_sessions=5, turns_per_session=10)
        history = _format_history(item, max_chars=200)
        assert len(history) <= 200


# ── Smoke test (requires Ollama running) ─────────────────────────────


class TestSmokeEndToEnd:
    @pytest.mark.skipif(
        not (DATA_DIR / "longmemeval_oracle.json").exists(),
        reason="LongMemEval oracle data not downloaded",
    )
    @pytest.mark.skip(reason="Requires running Ollama instance")
    def test_soma_single_item(self) -> None:
        from benchmarks.industry.longmemeval.evaluate_soma import evaluate_item

        items = load_dataset("oracle", limit=1)
        result = evaluate_item(items[0])
        assert "question_id" in result
        assert "hypothesis" in result
        assert result["hypothesis"]  # non-empty
