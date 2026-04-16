"""Tests for the LoCoMo QA-eval harness.

Task A — `--run-qa-eval` on `run_locomo.py`. The harness has three
moving parts:

1. `answer_question(query, hits, llm)` — concatenates retrieved hits
   into a short context, prompts the responder LLM, returns the
   candidate answer string.
2. `judge_answer(question, gold, candidate, judge_llm)` — asks a
   judge LLM for a JSON {match: bool, reason: str} verdict.
3. Wiring into `run_locomo.py` so a `--run-qa-eval` column appears on
   the main table per-arm.

Tests use a CapturingBackend that records prompts and returns
scripted replies so they're hermetic (no Ollama, no network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from benchmarks.harness.adapters.base import BenchmarkHit
from benchmarks.harness.qa_eval import (
    QAEvalResult,
    answer_question,
    evaluate_qa,
    judge_answer,
)


@dataclass
class CapturingBackend:
    """Records every prompt it sees; returns scripted replies in order."""

    replies: list[str] = field(default_factory=list)
    name: str = "capturing"
    prompts: list[str] = field(default_factory=list)
    _call_idx: int = 0

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        i = self._call_idx
        self._call_idx += 1
        if i < len(self.replies):
            return self.replies[i]
        return ""


def _hit(text: str, score: float = 0.5, dia_id: str = "D1:0") -> BenchmarkHit:
    return BenchmarkHit(
        text=text,
        score=score,
        metadata={"dia_id": dia_id, "sample_id": "s"},
        node_id=dia_id,
    )


def test_answer_question_calls_llm_once_per_question() -> None:
    llm = CapturingBackend(replies=["Paris."])
    hits = [
        _hit("User lives in Paris", dia_id="D1:0"),
        _hit("User speaks French", dia_id="D1:1"),
    ]
    answer = answer_question("Where does the user live?", hits, llm)

    assert answer == "Paris."
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    # Context + question are both embedded.
    assert "User lives in Paris" in prompt
    assert "Where does the user live?" in prompt


def test_answer_question_returns_idk_on_empty_hits() -> None:
    """With no retrieved context the prompt still renders; the LLM
    can decide (honestly) that it doesn't know."""
    llm = CapturingBackend(replies=["I don't know."])
    answer = answer_question("Where does the user live?", [], llm)

    assert "don't know" in answer.lower()
    assert len(llm.prompts) == 1


def test_judge_accepts_paraphrase() -> None:
    """Gold 'Paris' should match candidate 'the French capital' semantically."""
    judge = CapturingBackend(
        replies=[json.dumps({"match": True, "reason": "same entity"})]
    )
    verdict = judge_answer(
        "Where does the user live?",
        gold_answer="Paris",
        candidate_answer="the French capital",
        judge_llm=judge,
    )
    assert verdict is True


def test_judge_rejects_unrelated_answer() -> None:
    """Gold 'Paris', candidate 'Berlin' — judge must return False."""
    judge = CapturingBackend(
        replies=[json.dumps({"match": False, "reason": "different cities"})]
    )
    verdict = judge_answer(
        "Where does the user live?",
        gold_answer="Paris",
        candidate_answer="Berlin",
        judge_llm=judge,
    )
    assert verdict is False


def test_judge_malformed_json_falls_back_to_false() -> None:
    """A judge that returns prose (ignoring the JSON-only instruction)
    must not crash the run — default to False so the eval stays strict."""
    judge = CapturingBackend(replies=["the candidate is kinda close but..."])
    verdict = judge_answer(
        "Where does the user live?",
        gold_answer="Paris",
        candidate_answer="France somewhere",
        judge_llm=judge,
    )
    assert verdict is False


def test_evaluate_qa_counts_matches() -> None:
    """evaluate_qa wires answer_question + judge_answer for every (query, hits,
    gold) triple and returns a QAEvalResult with accuracy + call count."""
    responder = CapturingBackend(replies=["Paris", "pizza"])
    judge = CapturingBackend(
        replies=[
            json.dumps({"match": True, "reason": "correct"}),
            json.dumps({"match": False, "reason": "wrong dish"}),
        ]
    )
    triples = [
        ("Where does the user live?", [_hit("User lives in Paris")], "Paris"),
        ("Favorite food?", [_hit("User likes sushi")], "sushi"),
    ]
    result = evaluate_qa(triples, responder_llm=responder, judge_llm=judge)
    assert isinstance(result, QAEvalResult)
    assert result.n_questions == 2
    assert result.n_correct == 1
    assert result.accuracy == 0.5
    # Two responder calls + two judge calls = 4 LLM calls.
    assert result.llm_calls == 4


def test_run_locomo_with_qa_eval_smoke(tmp_path, monkeypatch) -> None:
    """End-to-end: run_locomo with --run-qa-eval on a tiny 3-turn fixture
    produces a report that mentions the QA accuracy column, using a
    DryRunBackend so nothing hits the network."""
    # Patch the dataset loader to a tiny fixture so the smoke test
    # runs in under a second.
    from benchmarks import run_locomo
    from benchmarks.datasets.locomo import LoCoMoQuery, LoCoMoTurn

    tiny_turns = [
        LoCoMoTurn("s1", "D1:0", "Alice", "I live in Boston."),
        LoCoMoTurn("s1", "D1:1", "Bob", "How's the weather?"),
        LoCoMoTurn("s1", "D1:2", "Alice", "I love Thai food."),
    ]
    tiny_queries = [
        LoCoMoQuery(
            sample_id="s1",
            question="Where does Alice live?",
            answer="Boston",
            evidence=["D1:0"],
            category=1,
        ),
        LoCoMoQuery(
            sample_id="s1",
            question="What food does Alice love?",
            answer="Thai food",
            evidence=["D1:2"],
            category=1,
        ),
    ]
    monkeypatch.setattr(
        run_locomo, "load_locomo", lambda: (tiny_turns, tiny_queries)
    )

    # Force DryRunBackend — the test must not hit Ollama / OpenAI /
    # Anthropic even if the dev machine has them configured.
    from soma.llm.backends import DryRunBackend

    monkeypatch.setattr(
        "benchmarks.run_locomo.backend_from_env",
        lambda: DryRunBackend(),
    )

    out = tmp_path / "locomo_qa.md"
    # argv simulates `python -m benchmarks.run_locomo --out ... --run-qa-eval`
    import sys

    argv = sys.argv.copy()
    sys.argv = [
        "run_locomo",
        "--out",
        str(out),
        "--run-qa-eval",
        "--qa-eval-max-questions",
        "2",
    ]
    try:
        run_locomo.main()
    finally:
        sys.argv = argv

    assert out.exists()
    report = out.read_text(encoding="utf-8")
    assert "QA" in report  # "qa_accuracy" column header present
