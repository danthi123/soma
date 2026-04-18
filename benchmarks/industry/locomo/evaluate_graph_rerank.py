"""LoCoMo graph reranking evaluation.

For each conversation: load all turns into a MemoryLayer, consolidate
through SOMA once, then evaluate all QA pairs at alpha=0.0 (pure cosine)
and alpha=1.0 (pure graph). Reports per-category and overall F1 delta.

Usage::

    python -m benchmarks.industry.locomo.evaluate_graph_rerank
    python -m benchmarks.industry.locomo.evaluate_graph_rerank --limit 2 --qa-limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
import torch

from benchmarks.industry.locomo.data_loader import (
    CATEGORY_NAMES,
    LoCoMoConversation,
    load_dataset,
)
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer
from soma.schemas.packing import pack_context
from soma.system import SOMA

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("research/audit/results")

_SBERT = None


def _get_sbert():
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT


def _sbert_embed(text: str) -> torch.Tensor:
    return _get_sbert().encode(text, convert_to_tensor=True)


def _format_turn(speaker: str, text: str, date_time: str, session_idx: int) -> str:
    return f"[{date_time}] [session_{session_idx}] {speaker}: {text}"


def _call_llm(
    question: str,
    context: str,
    api_base: str,
    model: str,
) -> str:
    system_prompt = (
        "You are an AI assistant that recalls information from past conversations. "
        "Answer the question using ONLY the context provided. Be concise and direct. "
        "If the context does not contain the answer, say 'I don't know'."
    )
    user_msg = ""
    if context.strip():
        user_msg += f"Relevant conversation history:\n{context}\n\n"
    user_msg += f"Question: {question}\nAnswer:"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": 256, "temperature": 0.0},
    }
    try:
        resp = requests.post(
            f"{api_base}/api/chat", json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        logger.error("LLM failed: %s", exc)
        return ""


def evaluate_conversation(
    conv: LoCoMoConversation,
    soma: SOMA,
    tokenizer: Any,
    encoder: TextEncoder,
    *,
    api_base: str,
    model: str,
    qa_limit: int | None = None,
) -> dict:
    """Evaluate one conversation with alpha=0.0 and alpha=1.0."""
    # Build MemoryLayer and ingest all sessions
    mem = MemoryLayer(
        embed_fn=_sbert_embed,
        embed_dim=384,
        graph_rerank_alpha=1.0,
    )

    for session in conv.sessions:
        texts = []
        metas = []
        for turn in session.turns:
            texts.append(_format_turn(
                turn.speaker, turn.text, session.date_time, session.index,
            ))
            metas.append({
                "session_idx": session.index,
                "session_date": session.date_time,
                "dia_id": turn.dia_id,
                "speaker": turn.speaker,
            })
        if texts:
            mem.store_batch(texts, metadatas=metas)

    # Consolidate once
    mem.attach_soma(soma, tokenizer, encoder)
    tc = time.perf_counter()
    mem.consolidate()
    mem.stable_capture()
    consol_time = time.perf_counter() - tc

    # Evaluate QA pairs with both alpha values
    qa_pairs = conv.qa_pairs
    if qa_limit is not None:
        qa_pairs = qa_pairs[:qa_limit]

    qa_results = []
    for qi, qa in enumerate(qa_pairs):
        hypotheses = {}
        for alpha in [0.0, 1.0]:
            mem._graph_rerank_alpha = alpha
            ctx = pack_context(
                mem,
                query=qa.question,
                max_tokens=3800,
                mix={"recency": 0.10, "relevant": 0.80, "preferences": 0.10},
            )
            hyp = _call_llm(qa.question, ctx, api_base, model)
            hypotheses[alpha] = hyp

        f1_0 = token_f1(hypotheses[0.0], str(qa.answer))
        f1_1 = token_f1(hypotheses[1.0], str(qa.answer))
        delta = f1_1 - f1_0

        qa_results.append({
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
            "category_name": qa.category_name,
            "f1_alpha_0": round(f1_0, 4),
            "f1_alpha_1": round(f1_1, 4),
            "delta": round(delta, 4),
            "hyp_alpha_0": hypotheses[0.0],
            "hyp_alpha_1": hypotheses[1.0],
        })

        if (qi + 1) % 10 == 0:
            logger.info(
                "  [%d/%d QAs] avg delta so far: %+.4f",
                qi + 1, len(qa_pairs),
                sum(r["delta"] for r in qa_results) / len(qa_results),
            )

    return {
        "sample_id": conv.sample_id,
        "n_sessions": conv.num_sessions,
        "n_turns": conv.total_turns,
        "n_qa_evaluated": len(qa_results),
        "n_memories": len(mem._ids),
        "consolidation_time_s": round(consol_time, 2),
        "qa_results": qa_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoCoMo graph reranking evaluation",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max conversations to evaluate")
    parser.add_argument("--qa-limit", type=int, default=None,
                        help="Max QA pairs per conversation")
    parser.add_argument("--model", default="qwen3.5:4b-q8_0")
    parser.add_argument("--api-base", default="http://localhost:11434")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else "cpu"
    )
    print(f"Device: {device}", flush=True)

    conversations = load_dataset(limit=args.limit)
    print(f"Loaded {len(conversations)} conversations", flush=True)

    # Build BPE corpus from all conversation text
    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(_format_turn(
                    turn.speaker, turn.text, session.date_time, session.index,
                ))
    print(f"BPE corpus: {len(corpus)} turns", flush=True)

    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=8,
        initial_associator_count=16,
        seed=42,
    )
    tokenizer = train_bpe_tokenizer(corpus[:2000], vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=64, device=device)
    soma = SOMA(config, device=device)

    all_results = []
    t0 = time.perf_counter()

    for ci, conv in enumerate(conversations):
        print(f"\n--- Conversation {ci + 1}/{len(conversations)}: "
              f"{conv.sample_id} ({conv.num_sessions} sessions, "
              f"{conv.total_turns} turns, {conv.num_qa} QAs) ---", flush=True)

        result = evaluate_conversation(
            conv, soma, tokenizer, encoder,
            api_base=args.api_base,
            model=args.model,
            qa_limit=args.qa_limit,
        )
        all_results.append(result)

        # Per-conversation summary
        qa_r = result["qa_results"]
        avg_f1_0 = sum(r["f1_alpha_0"] for r in qa_r) / len(qa_r)
        avg_f1_1 = sum(r["f1_alpha_1"] for r in qa_r) / len(qa_r)
        avg_delta = sum(r["delta"] for r in qa_r) / len(qa_r)
        wins_g = sum(1 for r in qa_r if r["delta"] > 0.01)
        wins_c = sum(1 for r in qa_r if r["delta"] < -0.01)
        ties = len(qa_r) - wins_g - wins_c

        print(f"  F1@0.0={avg_f1_0:.4f}  F1@1.0={avg_f1_1:.4f}  "
              f"delta={avg_delta:+.4f}", flush=True)
        print(f"  Graph wins: {wins_g}, Cosine wins: {wins_c}, Ties: {ties}",
              flush=True)

    total_time = time.perf_counter() - t0

    # Aggregate summary
    print(f"\n{'=' * 70}", flush=True)
    print("LOCOMO GRAPH RERANKING EVALUATION", flush=True)
    print(f"{'=' * 70}", flush=True)

    all_qa = [r for conv_r in all_results for r in conv_r["qa_results"]]
    total_qa = len(all_qa)
    avg_f1_0 = sum(r["f1_alpha_0"] for r in all_qa) / total_qa
    avg_f1_1 = sum(r["f1_alpha_1"] for r in all_qa) / total_qa
    avg_delta = sum(r["delta"] for r in all_qa) / total_qa

    print(f"\nOverall ({total_qa} QAs):")
    print(f"  F1@alpha=0.0 (cosine): {avg_f1_0:.4f}")
    print(f"  F1@alpha=1.0 (graph):  {avg_f1_1:.4f}")
    print(f"  Delta:                 {avg_delta:+.4f} ({avg_delta / max(avg_f1_0, 1e-9) * 100:+.1f}%)")

    wins_g = sum(1 for r in all_qa if r["delta"] > 0.01)
    wins_c = sum(1 for r in all_qa if r["delta"] < -0.01)
    ties = total_qa - wins_g - wins_c
    print(f"  Graph wins: {wins_g}, Cosine wins: {wins_c}, Ties: {ties}")

    # Per-category breakdown
    print(f"\nPer category:")
    cat_groups: dict[str, list] = defaultdict(list)
    for r in all_qa:
        cat_groups[r["category_name"]].append(r)

    print(f"  {'Category':<15s} {'N':>5s} {'F1@0.0':>8s} {'F1@1.0':>8s} "
          f"{'Delta':>8s} {'G-wins':>7s} {'C-wins':>7s}")
    print(f"  {'-' * 60}")
    for cat_name in ["single_hop", "temporal", "multi_hop", "open_domain", "adversarial"]:
        items = cat_groups.get(cat_name, [])
        if not items:
            continue
        n = len(items)
        f0 = sum(r["f1_alpha_0"] for r in items) / n
        f1 = sum(r["f1_alpha_1"] for r in items) / n
        d = f1 - f0
        gw = sum(1 for r in items if r["delta"] > 0.01)
        cw = sum(1 for r in items if r["delta"] < -0.01)
        print(f"  {cat_name:<15s} {n:>5d} {f0:>8.4f} {f1:>8.4f} "
              f"{d:>+8.4f} {gw:>7d} {cw:>7d}")

    print(f"\nTotal time: {total_time:.0f}s ({total_time / total_qa:.1f}s/QA)")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "locomo_graph_rerank.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "total_qa": total_qa,
                "avg_f1_alpha_0": round(avg_f1_0, 4),
                "avg_f1_alpha_1": round(avg_f1_1, 4),
                "delta": round(avg_delta, 4),
                "wins_graph": wins_g,
                "wins_cosine": wins_c,
                "ties": ties,
                "total_time_s": round(total_time, 1),
            },
            "conversations": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
