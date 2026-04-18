"""Phase 1.3: graph_rerank_alpha sweep on LongMemEval.

Tests whether SOMA's graph-based re-ranking improves retrieval quality
over pure cosine similarity.

Consolidates each item ONCE, then sweeps alpha values by changing the
blend weight — avoids repeated O(N) consolidation passes.

Usage::

    python -m research.audit.phase1_graph_rerank --limit 10
    python -m research.audit.phase1_graph_rerank --limit 30
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch

from benchmarks.industry.longmemeval.data_loader import LongMemEvalItem, load_dataset
from benchmarks.industry.longmemeval.metrics import compute_scores
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


def _format_turn(turn, session_id: str, session_date: str) -> str:
    return f"[{session_date}] [{session_id}] {turn.role}: {turn.content}"


def _build_soma_stack(device, corpus):
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=8,
        initial_associator_count=16,
        seed=42,
    )
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=64, device=device)
    soma = SOMA(config, device=device)
    return soma, tokenizer, encoder


def _call_llm(question, context, question_date, api_base, model):
    import requests
    system_prompt = (
        "You are an AI assistant that recalls information from past "
        "conversations. Answer the question using ONLY the context "
        "provided. Be concise and direct. If the context does not "
        "contain the answer, say 'I don't know'."
    )
    user_msg = ""
    if context.strip():
        user_msg += f"Relevant conversation history:\n{context}\n\n"
    user_msg += f"Current date: {question_date}\n"
    user_msg += f"Question: {question}\nAnswer:"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": 512, "temperature": 0.0},
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.3: graph_rerank_alpha sweep"
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="qwen3.5:4b-q8_0")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else "cpu"
    )
    print(f"Device: {device}")

    items = load_dataset("oracle", limit=args.limit)
    print(f"Loaded {len(items)} items")
    print(f"Alpha values: {args.alphas}")

    # Build SOMA stack once from all conversation text
    corpus = []
    for item in items:
        for sess_idx, session in enumerate(item.haystack_sessions):
            sid = item.haystack_session_ids[sess_idx] if sess_idx < len(item.haystack_session_ids) else f"s{sess_idx}"
            sdate = item.haystack_dates[sess_idx] if sess_idx < len(item.haystack_dates) else ""
            for turn in session:
                corpus.append(_format_turn(turn, sid, sdate))
    print(f"BPE corpus: {len(corpus)} turns")
    soma, tokenizer, encoder = _build_soma_stack(device, corpus[:1000])

    # Per-alpha accumulators
    alpha_predictions: dict[float, list] = defaultdict(list)
    alpha_references: dict[float, list] = defaultdict(list)
    alpha_times: dict[float, float] = defaultdict(float)
    consolidation_time = 0.0

    t0 = time.perf_counter()

    for i, item in enumerate(items):
        # Build and populate MemoryLayer (alpha=max so consolidation runs)
        mem = MemoryLayer(
            embed_fn=_sbert_embed,
            embed_dim=384,
            graph_rerank_alpha=max(args.alphas),
        )

        for sess_idx, session in enumerate(item.haystack_sessions):
            sid = item.haystack_session_ids[sess_idx] if sess_idx < len(item.haystack_session_ids) else f"s{sess_idx}"
            sdate = item.haystack_dates[sess_idx] if sess_idx < len(item.haystack_dates) else ""
            texts, metas = [], []
            for turn_idx, turn in enumerate(session):
                texts.append(_format_turn(turn, sid, sdate))
                metas.append({"session_id": sid, "session_date": sdate,
                              "turn_idx": turn_idx, "role": turn.role})
            if texts:
                mem.store_batch(texts, metadatas=metas)

        # Consolidate ONCE (attach SOMA, run growth pass + stable capture)
        mem.attach_soma(soma, tokenizer, encoder)
        tc = time.perf_counter()
        mem.consolidate()
        mem.stable_capture()
        consolidation_time += time.perf_counter() - tc

        # Now sweep alpha values — just change the blend weight
        for alpha in args.alphas:
            mem._graph_rerank_alpha = alpha
            ta = time.perf_counter()

            context = pack_context(
                mem, query=item.question, max_tokens=3800,
                mix={"recency": 0.10, "relevant": 0.80, "preferences": 0.10},
            )
            hypothesis = _call_llm(
                item.question, context, item.question_date,
                args.api_base, args.model,
            )

            alpha_times[alpha] += time.perf_counter() - ta
            alpha_predictions[alpha].append({
                "question_id": item.question_id,
                "hypothesis": hypothesis,
            })
            alpha_references[alpha].append({
                "question_id": item.question_id,
                "answer": item.answer,
                "question_type": item.question_type,
            })

        logger.info(
            "[%d/%d] %s consolidated + %d alphas swept",
            i + 1, len(items), item.question_id, len(args.alphas),
        )

    total = time.perf_counter() - t0

    # Compute scores per alpha
    results = []
    for alpha in args.alphas:
        scores = compute_scores(alpha_predictions[alpha], alpha_references[alpha])
        results.append({
            "alpha": alpha,
            "f1": round(scores.f1, 4),
            "em": round(scores.em, 4),
            "rouge1": round(scores.rouge1, 4),
            "rouge2": round(scores.rouge2, 4),
            "rougeL": round(scores.rougeL, 4),
            "n": scores.n_total,
            "retrieval_time_s": round(alpha_times[alpha], 1),
            "per_type": scores.per_type,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("PHASE 1.3: graph_rerank_alpha SWEEP")
    print(f"{'=' * 60}")
    print(f"{'Alpha':>8s} {'F1':>8s} {'EM':>8s} {'ROUGE-L':>8s} {'Retr(s)':>8s}")
    print("-" * 44)

    for r in results:
        print(f"{r['alpha']:>8.1f} {r['f1']:>8.4f} {r['em']:>8.4f} "
              f"{r['rougeL']:>8.4f} {r['retrieval_time_s']:>7.1f}s")

    print(f"\nConsolidation total: {consolidation_time:.0f}s")
    print(f"Total wall-clock: {total:.0f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase1_3_graph_rerank_alpha.json"
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "consolidation_time_s": round(consolidation_time, 1),
            "total_wall_clock_s": round(total, 1),
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
