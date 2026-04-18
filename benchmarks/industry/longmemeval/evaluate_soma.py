"""SOMA adapter for LongMemEval benchmark.

Feeds each conversation history into a MemoryLayer via store_typed,
then for each question uses retrieve + pack_context to build a prompt
and calls a local LLM to generate an answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
import torch

from benchmarks.industry.longmemeval.data_loader import (
    LongMemEvalItem,
    Turn,
    load_dataset,
)
from benchmarks.industry.longmemeval.metrics import (
    LongMemEvalScores,
    compute_scores,
)
from soma.memory.api import MemoryLayer
from soma.schemas.packing import pack_context

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b-q8_0"
CHARS_PER_TOKEN = 4

# Lazy-loaded sentence-transformer for real semantic embeddings
_SBERT_MODEL = None
_SBERT_DIM = 384


def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _sbert_embed(text: str) -> torch.Tensor:
    model = _get_sbert()
    return model.encode(text, convert_to_tensor=True)


def _build_memory_layer() -> MemoryLayer:
    return MemoryLayer(embed_fn=_sbert_embed, embed_dim=_SBERT_DIM)


def _format_turn_text(turn: Turn, session_id: str, session_date: str) -> str:
    """Format a conversation turn as storable text."""
    return f"[{session_date}] [{session_id}] {turn.role}: {turn.content}"


def ingest_conversations(
    mem: MemoryLayer,
    item: LongMemEvalItem,
) -> None:
    """Store all conversation turns from a LongMemEval item into the MemoryLayer."""
    for sess_idx, session in enumerate(item.haystack_sessions):
        sess_id = (
            item.haystack_session_ids[sess_idx]
            if sess_idx < len(item.haystack_session_ids)
            else f"session_{sess_idx}"
        )
        sess_date = item.haystack_dates[sess_idx] if sess_idx < len(item.haystack_dates) else ""
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for turn_idx, turn in enumerate(session):
            text = _format_turn_text(turn, sess_id, sess_date)
            meta = {
                "session_id": sess_id,
                "session_date": sess_date,
                "turn_idx": turn_idx,
                "role": turn.role,
                "question_id": item.question_id,
            }
            texts.append(text)
            metas.append(meta)
        if texts:
            mem.store_batch(texts, metadatas=metas)


def answer_question(
    mem: MemoryLayer,
    question: str,
    question_date: str,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = 3800,
    disable_thinking: bool = True,
) -> str:
    """Use SOMA memory + LLM to answer a LongMemEval question."""
    context = pack_context(
        mem,
        query=question,
        max_tokens=max_context_tokens,
        mix={
            "recency": 0.10,
            "relevant": 0.80,
            "preferences": 0.10,
        },
    )

    system_prompt = (
        "You are an AI assistant that recalls information from past conversations. "
        "Answer the question using ONLY the context provided. Be concise and direct. "
        "If the context does not contain the answer, say 'I don't know'."
    )

    user_msg = ""
    if context.strip():
        user_msg += f"Relevant conversation history:\n{context}\n\n"
    user_msg += f"Current date: {question_date}\n"
    user_msg += f"Question: {question}\nAnswer:"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": not disable_thinking,
        "options": {"num_predict": 512, "temperature": 0.0},
    }

    try:
        resp = requests.post(
            f"{api_base}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        content: str = data["message"]["content"]
        return content.strip()
    except (requests.RequestException, KeyError, IndexError) as exc:
        logger.error("LLM call failed: %s", exc)
        return ""


def evaluate_item(
    item: LongMemEvalItem,
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = 3800,
) -> dict[str, str]:
    """Evaluate a single LongMemEval item with SOMA."""
    mem = _build_memory_layer()
    ingest_conversations(mem, item)
    hypothesis = answer_question(
        mem,
        item.question,
        item.question_date,
        api_base=api_base,
        model=model,
        max_context_tokens=max_context_tokens,
    )
    return {
        "question_id": item.question_id,
        "hypothesis": hypothesis,
    }


def run_evaluation(
    variant: str = "oracle",
    *,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = 3800,
    limit: int | None = None,
    output_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> LongMemEvalScores:
    """Run the full LongMemEval evaluation with SOMA."""
    items = load_dataset(variant, data_dir=data_dir, limit=limit)
    logger.info("Loaded %d items from LongMemEval %s", len(items), variant)

    predictions: list[dict[str, str]] = []
    references: list[dict[str, str]] = []

    for i, item in enumerate(items):
        t0 = time.monotonic()
        pred = evaluate_item(
            item,
            api_base=api_base,
            model=model,
            max_context_tokens=max_context_tokens,
        )
        elapsed = time.monotonic() - t0
        predictions.append(pred)
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
            pred["hypothesis"][:80],
        )

    scores = compute_scores(predictions, references)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "method": "soma",
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
    parser = argparse.ArgumentParser(description="LongMemEval SOMA evaluation")
    parser.add_argument(
        "--variant",
        default="oracle",
        choices=["oracle", "small", "medium"],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--max-context-tokens", type=int, default=3800)
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

    print("\n=== LongMemEval SOMA Results ===")
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
