"""LLM-as-judge QA evaluation for LoCoMo — the retrieval-quality
complement to Recall@k.

Recall@k measures whether the memory layer *surfaces* the right
evidence; QA accuracy measures whether the LLM *uses* that evidence
to produce an answer that matches the gold annotation. Together the
two numbers separate memory mistakes (missing evidence) from LLM
mistakes (evidence present, wrong answer) — which is the point of
the LoCoMo benchmark in Mem0's paper (arXiv 2504.19413).

The pipeline, per question:

1. Build a short context from the retrieved hits' texts.
2. Prompt the responder LLM with a short instruction + context +
   question, at ``temperature=0.0`` for determinism.
3. Prompt the judge LLM with {question, gold, candidate} and parse a
   JSON ``{match: bool, reason: ...}`` reply.

The judge prompt asks for JSON-only output; malformed replies
default to ``match=False`` so the eval stays strict.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.harness.adapters.base import BenchmarkHit
    from soma.llm.backends import LLMBackend

logger = logging.getLogger("benchmarks.qa_eval")


ANSWER_PROMPT = """Answer the question using only the context below. \
If the context doesn't cover it, say "I don't know."

Context:
{context}

Question: {question}

Answer (one sentence):"""


JUDGE_PROMPT = """You are a strict answer grader. Does the candidate \
answer match the gold answer in meaning? Ignore wording differences.

Question: {question}
Gold answer: {gold}
Candidate: {candidate}

Output JSON only, no prose: {{"match": true|false, "reason": "..."}}"""


# Very rough: ~4 chars/token heuristic. Good enough for a cost-guard
# report — we aren't billing on this.
_CHARS_PER_TOKEN = 4


@dataclass
class QAEvalResult:
    """Aggregate QA-eval numbers for one arm (one adapter x one run)."""

    n_questions: int
    n_correct: int
    llm_calls: int
    est_total_tokens: int = 0

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(1, self.n_questions)


def _render_context(hits: list[BenchmarkHit], *, max_hits: int = 10) -> str:
    """Concatenate hits' texts into a short numbered context block.

    Truncates to the top ``max_hits`` entries so local-model context
    windows don't blow up on long retrievals (LoCoMo's k=10 already
    fits but the cap leaves room for experimentation).
    """
    lines: list[str] = []
    for i, h in enumerate(hits[:max_hits], start=1):
        lines.append(f"[{i}] {h.text}")
    return "\n".join(lines) if lines else "(no context available)"


def answer_question(
    query: str,
    hits: list[BenchmarkHit],
    llm: LLMBackend,
    *,
    max_tokens: int = 128,
) -> str:
    """Generate a candidate answer using ``llm`` grounded in ``hits``.

    Deterministic (``temperature=0.0`` is set by every backend's
    ``generate`` by default). Returns the LLM's stripped reply. One
    LLM call per question.
    """
    prompt = ANSWER_PROMPT.format(
        context=_render_context(hits),
        question=query,
    )
    return llm.generate(prompt, max_tokens=max_tokens).strip()


def judge_answer(
    question: str,
    gold_answer: str,
    candidate_answer: str,
    judge_llm: LLMBackend,
    *,
    max_tokens: int = 128,
) -> bool:
    """Ask ``judge_llm`` whether the candidate matches the gold.

    Returns ``True`` iff the judge replies with valid JSON containing
    ``"match": true``. Anything else (prose, malformed JSON, missing
    key, non-bool value) is treated as a no-match so the eval stays
    strict — a broken judge shouldn't inflate the reported accuracy.
    """
    prompt = JUDGE_PROMPT.format(
        question=question, gold=gold_answer, candidate=candidate_answer,
    )
    raw = judge_llm.generate(prompt, max_tokens=max_tokens).strip()
    # Some models wrap JSON in code fences; strip a leading/trailing fence.
    if raw.startswith("```"):
        # Drop the first line (``` or ```json) and the last (```) if present.
        lines = raw.split("\n")
        if len(lines) >= 2:
            raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("judge: non-JSON reply (%s); default False", raw[:80])
        return False
    if not isinstance(parsed, dict):
        return False
    match = parsed.get("match")
    if isinstance(match, bool):
        return match
    # Some small models emit "true" / "false" as strings; be forgiving.
    if isinstance(match, str) and match.lower() in {"true", "false"}:
        return match.lower() == "true"
    return False


def evaluate_qa(
    triples: list[tuple[str, list[BenchmarkHit], str]],
    *,
    responder_llm: LLMBackend,
    judge_llm: LLMBackend,
    max_tokens_answer: int = 128,
    max_tokens_judge: int = 128,
) -> QAEvalResult:
    """Run the full (answer + judge) loop over every ``(question, hits, gold)``.

    ``responder_llm`` generates the candidate; ``judge_llm`` scores it
    against the gold. Counts total LLM calls + a rough token estimate
    (based on prompt + reply char lengths) so the runner can print a
    cost-guard summary at the end.
    """
    n_questions = len(triples)
    n_correct = 0
    llm_calls = 0
    est_tokens = 0

    for question, hits, gold in triples:
        # Answer step.
        prompt_a = ANSWER_PROMPT.format(
            context=_render_context(hits),
            question=question,
        )
        candidate = responder_llm.generate(
            prompt_a, max_tokens=max_tokens_answer
        ).strip()
        llm_calls += 1
        est_tokens += (len(prompt_a) + len(candidate)) // _CHARS_PER_TOKEN

        # Judge step.
        prompt_j = JUDGE_PROMPT.format(
            question=question, gold=gold, candidate=candidate,
        )
        raw = judge_llm.generate(prompt_j, max_tokens=max_tokens_judge)
        llm_calls += 1
        est_tokens += (len(prompt_j) + len(raw)) // _CHARS_PER_TOKEN

        match = _parse_judge_reply(raw)
        if match:
            n_correct += 1

    return QAEvalResult(
        n_questions=n_questions,
        n_correct=n_correct,
        llm_calls=llm_calls,
        est_total_tokens=est_tokens,
    )


def _parse_judge_reply(raw: str) -> bool:
    """Same logic as :func:`judge_answer` but takes the raw reply directly.

    Factored out so :func:`evaluate_qa` doesn't double-call the LLM.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if len(lines) >= 2:
            raw = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    match = parsed.get("match")
    if isinstance(match, bool):
        return match
    if isinstance(match, str) and match.lower() in {"true", "false"}:
        return match.lower() == "true"
    return False
