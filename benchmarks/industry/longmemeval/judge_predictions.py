"""LLM-as-judge post-hoc evaluation for LongMemEval QA predictions.

Reads a per-item jsonl produced by run_qa_compare (columns: question_id,
hypothesis, answer, question_type) and asks a separate LLM "is the
hypothesis a correct answer to the question given the gold?" for each
row. Saves a judged jsonl + a per-type accuracy summary.

Why: token-F1 is known to penalize verbose-but-correct answers and
benchmark-mismatch question types (e.g. single-session-preference
where gold is a paraphrase sentence). An LLM-judge accuracy metric
sidesteps both issues — it just checks semantic equivalence.

Because questions alone don't carry enough context for judging, this
harness also loads the original LongMemEval items so the judge sees
the question + gold answer + candidate answer.

Usage::

    python -m benchmarks.industry.longmemeval.judge_predictions \\
        --pred-jsonl benchmarks/.../qa_compare_soma_hybrid_n500_strict.jsonl \\
        --judge-provider anthropic --judge-model claude-haiku-4-5 \\
        --variant small

Cheap: judge prompt is ~300 tokens, output is ~20 tokens. On haiku
rates ($0.25/$1.25 per Mtok) ~$0.001 per item → $0.5 for 500 items.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from benchmarks.industry.longmemeval.data_loader import load_dataset

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = (
    "You are a strict grader of short-form QA. Given a question, a "
    "gold reference answer, and a candidate answer, reply with EXACTLY "
    "one word: 'yes' if the candidate conveys the same factual answer "
    "as the gold (phrasing/casing differences are OK), or 'no' "
    "otherwise. If the candidate says 'I don't know' or equivalent, "
    "reply 'no'. Never explain. Never output anything else."
)


def _judge_prompt(question: str, gold: str, candidate: str) -> str:
    return (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Candidate answer: {candidate}\n"
        f"Is the candidate correct? Reply only 'yes' or 'no'."
    )


def _call_judge(
    prompt: str,
    provider: str,
    model: str,
    api_base: str = "http://localhost:11434",
) -> str:
    if provider == "ollama":
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": 8, "temperature": 0.0},
        }
        try:
            resp = requests.post(f"{api_base}/api/chat", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip().lower()
        except Exception as exc:
            logger.error("Ollama judge call failed: %s", exc)
            return "error"

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
            client = Anthropic()
            resp = client.messages.create(
                model=model,
                max_tokens=8,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.content[0].text.strip().lower()
        except Exception as exc:
            logger.error("Anthropic judge call failed: %s", exc)
            return "error"

    raise ValueError(f"unknown provider: {provider}")


def _parse_verdict(raw: str) -> int:
    """Map a raw judge response to 1 (yes/correct) or 0 (no/incorrect/error)."""
    s = raw.strip().lower()
    if s.startswith("yes"):
        return 1
    if s.startswith("no"):
        return 0
    # Fallback: empty, error, or unparseable → treat as incorrect
    return 0


def judge_file(
    pred_path: Path,
    out_path: Path,
    question_by_qid: dict[str, str],
    provider: str,
    model: str,
    api_base: str,
) -> dict[str, Any]:
    """Judge every row of a predictions jsonl, with resume support."""
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["question_id"])
        logger.info("Resuming: %d items already judged", len(done))

    verdicts_by_type: dict[str, list[int]] = defaultdict(list)
    t0 = time.perf_counter()
    total = 0

    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["question_id"]
            if qid in done:
                continue
            question = question_by_qid.get(qid)
            if question is None:
                logger.warning("No question text for qid=%s, skipping", qid)
                continue
            hypothesis = row["hypothesis"]
            gold = row["answer"]
            qtype = row["question_type"]
            prompt = _judge_prompt(question, gold, hypothesis)
            raw = _call_judge(prompt, provider, model, api_base)
            verdict = _parse_verdict(raw)
            verdicts_by_type[qtype].append(verdict)

            with open(out_path, "a", encoding="utf-8") as g:
                g.write(json.dumps({
                    "question_id": qid,
                    "question_type": qtype,
                    "hypothesis": hypothesis,
                    "answer": gold,
                    "judge_raw": raw,
                    "judge_correct": verdict,
                }, ensure_ascii=False) + "\n")
            total += 1
            if total % 25 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "[%s] judged %d items | elapsed=%.1fs | last verdict=%d",
                    pred_path.stem, total, elapsed, verdict,
                )

    return {
        "total": sum(len(v) for v in verdicts_by_type.values()),
        "accuracy_by_type": {
            t: sum(v) / len(v) if v else 0.0
            for t, v in verdicts_by_type.items()
        },
        "accuracy_overall": (
            sum(sum(v) for v in verdicts_by_type.values())
            / max(1, sum(len(v) for v in verdicts_by_type.values()))
        ),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--pred-jsonl", required=True, type=Path,
                   help="Input predictions jsonl from run_qa_compare")
    p.add_argument("--variant", default="small",
                   help="LongMemEval variant to load question text from")
    p.add_argument("--judge-provider", default="ollama",
                   choices=["ollama", "anthropic"])
    p.add_argument("--judge-model", default="qwen3.5:4b-q8_0",
                   help="e.g. qwen3.5:4b-q8_0 (ollama) or "
                        "claude-haiku-4-5 (anthropic)")
    p.add_argument("--api-base", default="http://localhost:11434")
    p.add_argument("--out-suffix", default="_judged")
    args = p.parse_args()

    if args.judge_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY env var is empty — set it before "
            "--judge-provider anthropic"
        )

    items = load_dataset(args.variant, limit=None)
    question_by_qid = {it.question_id: it.question for it in items}
    logger.info("Loaded %d questions from variant %s", len(items), args.variant)

    pred_path = args.pred_jsonl
    out_path = pred_path.parent / f"{pred_path.stem}{args.out_suffix}.jsonl"
    summary = judge_file(
        pred_path, out_path, question_by_qid,
        provider=args.judge_provider,
        model=args.judge_model,
        api_base=args.api_base,
    )

    summary_path = pred_path.parent / f"{pred_path.stem}{args.out_suffix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary to %s", summary_path)
    logger.info("Overall accuracy: %.4f (%d items)",
                summary["accuracy_overall"], summary["total"])


if __name__ == "__main__":
    main()
