"""Baseline adapter for LongMemEval benchmark.

No SOMA -- feeds the full conversation history directly into the LLM
context window. When the history exceeds the token budget, older
sessions are truncated from the front.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from benchmarks.industry.longmemeval.data_loader import (
    LongMemEvalItem,
    load_dataset,
)
from benchmarks.industry.longmemeval.metrics import (
    LongMemEvalScores,
    compute_scores,
)

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3.5:4b-q8_0"
CHARS_PER_TOKEN = 4


def _format_history(item: LongMemEvalItem, max_chars: int) -> str:
    """Flatten all sessions into a chronological transcript.

    Pairs sessions with their dates for temporal context, then
    truncates from the front (oldest sessions) if the total exceeds
    *max_chars*.
    """
    blocks: list[str] = []
    for sess_idx, session in enumerate(item.haystack_sessions):
        sess_date = item.haystack_dates[sess_idx] if sess_idx < len(item.haystack_dates) else ""
        lines: list[str] = []
        if sess_date:
            lines.append(f"--- Session ({sess_date}) ---")
        else:
            lines.append(f"--- Session {sess_idx + 1} ---")
        for turn in session:
            lines.append(f"{turn.role}: {turn.content}")
        blocks.append("\n".join(lines))

    full = "\n\n".join(blocks)
    if len(full) <= max_chars:
        return full

    # Truncate oldest sessions until within budget
    while len(blocks) > 1:
        blocks.pop(0)
        full = "\n\n".join(blocks)
        if len(full) <= max_chars:
            break
    return full[-max_chars:]


def answer_question_baseline(
    item: LongMemEvalItem,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = 8000,
    disable_thinking: bool = True,
) -> str:
    """Answer a LongMemEval question using raw context window."""
    max_chars = max_context_tokens * CHARS_PER_TOKEN
    # Reserve chars for system + question
    reserved = 600
    history = _format_history(item, max_chars - reserved)

    system_prompt = (
        "You are an AI assistant that recalls information from past conversations. "
        "Answer the question using ONLY the conversation history provided. "
        "Be concise and direct. If the history does not contain the answer, "
        "say 'I don't know'."
    )
    if disable_thinking:
        system_prompt = "/no_think\n" + system_prompt

    user_msg = (
        f"Conversation history:\n{history}\n\n"
        f"Current date: {item.question_date}\n"
        f"Question: {item.question}\nAnswer:"
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "max_tokens": 256,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content: str = data["choices"][0]["message"]["content"]
        return content.strip()
    except (requests.RequestException, KeyError, IndexError) as exc:
        logger.error("LLM call failed: %s", exc)
        return ""


def run_evaluation(
    variant: str = "oracle",
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = 8000,
    limit: int | None = None,
    output_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> LongMemEvalScores:
    """Run the full LongMemEval evaluation without SOMA (baseline)."""
    items = load_dataset(variant, data_dir=data_dir, limit=limit)
    logger.info("Loaded %d items from LongMemEval %s", len(items), variant)

    predictions: list[dict[str, str]] = []
    references: list[dict[str, str]] = []

    for i, item in enumerate(items):
        t0 = time.monotonic()
        hypothesis = answer_question_baseline(
            item,
            api_base=api_base,
            model=model,
            max_context_tokens=max_context_tokens,
        )
        elapsed = time.monotonic() - t0
        predictions.append(
            {
                "question_id": item.question_id,
                "hypothesis": hypothesis,
            }
        )
        references.append(
            {
                "question_id": item.question_id,
                "answer": item.answer,
                "question_type": item.question_type,
            }
        )
        logger.info(
            "[%d/%d] %s | type=%s | %.1fs | hyp=%s",
            i + 1,
            len(items),
            item.question_id,
            item.question_type,
            elapsed,
            hypothesis[:80],
        )

    scores = compute_scores(predictions, references)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "method": "baseline",
            "model": model,
            "variant": variant,
            "scores": scores.to_dict(),
            "predictions": predictions,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results written to %s", out)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval baseline evaluation")
    parser.add_argument(
        "--variant",
        default="oracle",
        choices=["oracle", "small", "medium"],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--max-context-tokens", type=int, default=8000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scores = run_evaluation(
        variant=args.variant,
        api_base=args.api_base,
        model=args.model,
        max_context_tokens=args.max_context_tokens,
        limit=args.limit,
        output_path=args.output,
        data_dir=args.data_dir,
    )

    print("\n=== LongMemEval Baseline Results ===")
    print(f"  F1:      {scores.f1:.4f}")
    print(f"  EM:      {scores.em:.4f}")
    print(f"  ROUGE-1: {scores.rouge1:.4f}")
    print(f"  ROUGE-2: {scores.rouge2:.4f}")
    print(f"  ROUGE-L: {scores.rougeL:.4f}")
    print(f"  N:       {scores.n_total}")
    print("\nPer-type:")
    for qtype, type_scores in sorted(scores.per_type.items()):
        print(f"  {qtype}: F1={type_scores['f1']:.4f} EM={type_scores['em']:.4f}")


if __name__ == "__main__":
    main()
