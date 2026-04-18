"""Phase 1.3 per-item analysis: WHY does graph reranking help?

Compares alpha=0.0 (pure cosine) vs alpha=1.0 (pure graph) per item
and logs conversation characteristics to identify what makes graph
reranking useful.

Usage::

    python -m research.audit.phase1_3_per_item
    python -m research.audit.phase1_3_per_item --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from benchmarks.industry.longmemeval.data_loader import load_dataset
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


def _format_turn(turn, session_id: str, session_date: str) -> str:
    return f"[{session_date}] [{session_id}] {turn.role}: {turn.content}"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.3 per-item: why does graph reranking help?",
    )
    parser.add_argument("--limit", type=int, default=20)
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

    items = load_dataset("oracle", limit=args.limit)
    print(f"Loaded {len(items)} items", flush=True)

    # Build SOMA stack from all conversation text
    corpus = []
    for item in items:
        for si, sess in enumerate(item.haystack_sessions):
            sid = (
                item.haystack_session_ids[si]
                if si < len(item.haystack_session_ids)
                else f"s{si}"
            )
            sd = (
                item.haystack_dates[si]
                if si < len(item.haystack_dates)
                else ""
            )
            for t in sess:
                corpus.append(_format_turn(t, sid, sd))

    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=8,
        initial_associator_count=16,
        seed=42,
    )
    tokenizer = train_bpe_tokenizer(corpus[:1000], vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=64, device=device)
    soma = SOMA(config, device=device)

    # Header
    print(flush=True)
    print(
        f"{'Item':<20s} {'Type':<12s} {'Sess':>4s} {'Turns':>5s} "
        f"{'F1@0.0':>7s} {'F1@1.0':>7s} {'Delta':>7s} {'Winner':>7s}",
        flush=True,
    )
    print("-" * 78, flush=True)

    wins_graph = 0
    wins_cosine = 0
    ties = 0
    per_item_results = []

    t0 = time.perf_counter()
    for i, item in enumerate(items):
        # Build MemoryLayer and store conversations
        mem = MemoryLayer(
            embed_fn=_sbert_embed,
            embed_dim=384,
            graph_rerank_alpha=1.0,
        )

        total_turns = 0
        for si, sess in enumerate(item.haystack_sessions):
            sid = (
                item.haystack_session_ids[si]
                if si < len(item.haystack_session_ids)
                else f"s{si}"
            )
            sd = (
                item.haystack_dates[si]
                if si < len(item.haystack_dates)
                else ""
            )
            texts, metas = [], []
            for ti, t in enumerate(sess):
                texts.append(_format_turn(t, sid, sd))
                metas.append({
                    "session_id": sid,
                    "turn_idx": ti,
                    "role": t.role,
                })
                total_turns += 1
            if texts:
                mem.store_batch(texts, metadatas=metas)

        # Consolidate once
        mem.attach_soma(soma, tokenizer, encoder)
        tc = time.perf_counter()
        mem.consolidate()
        mem.stable_capture()
        consol_time = time.perf_counter() - tc

        # Compare alpha=0.0 vs alpha=1.0
        hypotheses = {}
        contexts = {}
        for alpha in [0.0, 1.0]:
            mem._graph_rerank_alpha = alpha
            ctx = pack_context(
                mem,
                query=item.question,
                max_tokens=3800,
                mix={"recency": 0.10, "relevant": 0.80, "preferences": 0.10},
            )
            contexts[alpha] = ctx
            hyp = _call_llm(
                item.question, ctx, item.question_date,
                args.api_base, args.model,
            )
            hypotheses[alpha] = hyp

        f1_0 = token_f1(hypotheses[0.0], item.answer)
        f1_1 = token_f1(hypotheses[1.0], item.answer)
        delta = f1_1 - f1_0

        if delta > 0.01:
            winner = "GRAPH"
            wins_graph += 1
        elif delta < -0.01:
            winner = "COSINE"
            wins_cosine += 1
        else:
            winner = "TIE"
            ties += 1

        n_sess = len(item.haystack_sessions)
        q_type = getattr(item, "question_type", "unknown")

        print(
            f"{item.question_id:<20s} {q_type:<12s} {n_sess:>4d} "
            f"{total_turns:>5d} {f1_0:>7.4f} {f1_1:>7.4f} "
            f"{delta:>+7.4f} {winner:>7s}",
            flush=True,
        )

        # Store detailed result
        per_item_results.append({
            "question_id": item.question_id,
            "question_type": q_type,
            "question": item.question,
            "answer": item.answer,
            "n_sessions": n_sess,
            "n_turns": total_turns,
            "n_memories": len(mem._ids),
            "f1_alpha_0": round(f1_0, 4),
            "f1_alpha_1": round(f1_1, 4),
            "delta": round(delta, 4),
            "winner": winner,
            "hyp_alpha_0": hypotheses[0.0],
            "hyp_alpha_1": hypotheses[1.0],
            "context_alpha_0_len": len(contexts[0.0]),
            "context_alpha_1_len": len(contexts[1.0]),
            "consolidation_time_s": round(consol_time, 2),
            "graph_nodes": len(soma.graph.nodes),
            "graph_edges": len(soma.graph.edges),
        })

    total_time = time.perf_counter() - t0

    # Summary
    print(flush=True)
    print(f"Graph wins: {wins_graph}, Cosine wins: {wins_cosine}, Ties: {ties}")
    print(f"Total time: {total_time:.0f}s ({total_time / len(items):.1f}s/item)")

    # Analyze patterns
    graph_items = [r for r in per_item_results if r["winner"] == "GRAPH"]
    cosine_items = [r for r in per_item_results if r["winner"] == "COSINE"]

    if graph_items:
        avg_sess = sum(r["n_sessions"] for r in graph_items) / len(graph_items)
        avg_turns = sum(r["n_turns"] for r in graph_items) / len(graph_items)
        print(f"\nGraph-wins avg: {avg_sess:.1f} sessions, {avg_turns:.0f} turns")

    if cosine_items:
        avg_sess = sum(r["n_sessions"] for r in cosine_items) / len(cosine_items)
        avg_turns = sum(r["n_turns"] for r in cosine_items) / len(cosine_items)
        print(f"Cosine-wins avg: {avg_sess:.1f} sessions, {avg_turns:.0f} turns")

    # Per-type breakdown
    from collections import defaultdict
    type_wins: dict[str, dict[str, int]] = defaultdict(lambda: {"GRAPH": 0, "COSINE": 0, "TIE": 0})
    for r in per_item_results:
        type_wins[r["question_type"]][r["winner"]] += 1
    print("\nPer question type:")
    for qtype, counts in sorted(type_wins.items()):
        print(f"  {qtype}: graph={counts['GRAPH']}, cosine={counts['COSINE']}, tie={counts['TIE']}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase1_3_per_item_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "wins_graph": wins_graph,
                "wins_cosine": wins_cosine,
                "ties": ties,
                "total_time_s": round(total_time, 1),
            },
            "per_item": per_item_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
