"""Feed ALL LoCoMo conversations through one SOMA instance.

Tests whether development compounds across 10 conversations (5882 turns).
Evaluates cross-conversation association — can SOMA recall information
from early conversations after developing through later ones?

Usage::

    python -m research.developmental.locomo_full_development
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
import torch

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop
from soma.developmental.verbalize import verbalize_state

RESULTS_DIR = Path("research/developmental/results")


def _call_llm(question: str, state: str, api_base: str, model: str) -> str:
    system_prompt = (
        "You are the voice of a developing mind. Use ONLY the recalled "
        "memories and internal state below to answer the question. "
        "Be concise. If you don't have enough information, say so."
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{state}\n\nQuestion: {question}\nAnswer:"},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": 128, "temperature": 0.0},
    }
    try:
        resp = requests.post(
            f"{api_base}/api/chat", json=payload, timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Full LoCoMo developmental test",
    )
    parser.add_argument("--qa-per-conv", type=int, default=10)
    parser.add_argument("--model", default="qwen3.5:4b-q8_0")
    parser.add_argument("--api-base", default="http://localhost:11434")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else "cpu"
    )
    print(f"Device: {device}", flush=True)

    conversations = load_dataset()
    total_turns = sum(c.total_turns for c in conversations)
    print(f"Loaded {len(conversations)} conversations, {total_turns} total turns",
          flush=True)

    # Build tokenizer from all text
    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(f"[{session.date_time}] {turn.speaker}: {turn.text}")

    config = SOMAConfig.developmental()
    loop = InteractionLoop(
        config=config,
        llm_model=args.model,
        llm_api_base=args.api_base,
        device=device,
    )
    loop.train_tokenizer(corpus[:500])

    # Phase 1: Development — feed ALL conversations through SOMA
    print(f"\n--- Phase 1: Development ({total_turns} turns) ---", flush=True)
    t0 = time.perf_counter()

    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                loop.process_input(text, call_llm=False)

        soma = loop.predictive_soma.soma
        n_nodes = len(soma.graph.nodes)
        n_edges = len(soma.graph.edges)
        n_mem = len(loop.predictive_soma.text_store)
        recent_err = list(loop.predictive_soma.error_history)[-50:]
        avg_err = sum(recent_err) / len(recent_err) if recent_err else 0

        print(f"  Conv {ci + 1}/10 ({conv.sample_id}): "
              f"nodes={n_nodes}, edges={n_edges}, memories={n_mem}, "
              f"avg_pred_err={avg_err:.6f}", flush=True)

    dev_time = time.perf_counter() - t0
    print(f"  Development complete in {dev_time:.1f}s", flush=True)

    # Phase 2: QA evaluation — test each conversation's questions
    print(f"\n--- Phase 2: QA Evaluation ---", flush=True)

    blank_state = (
        "[SOMA Internal State]\n"
        "Developmental Stage: blank-slate (step 0)\n"
        "No associations, no memories, no development.\n"
    )

    all_results = []
    for ci, conv in enumerate(conversations):
        qa_pairs = conv.qa_pairs[:args.qa_per_conv]
        if not qa_pairs:
            continue

        conv_results = []
        for qi, qa in enumerate(qa_pairs):
            # Graph-driven retrieval per question
            qvec = loop.encode_text(qa.question)
            recalled = loop.predictive_soma.retrieve_by_graph(qvec, top_k=5)
            dev_state = verbalize_state(
                loop.predictive_soma.soma, recalled=recalled,
            )

            hyp_dev = _call_llm(qa.question, dev_state, args.api_base, args.model)
            hyp_blank = _call_llm(qa.question, blank_state, args.api_base, args.model)

            f1_dev = token_f1(hyp_dev, str(qa.answer))
            f1_blank = token_f1(hyp_blank, str(qa.answer))

            conv_results.append({
                "question": qa.question,
                "answer": str(qa.answer),
                "category": qa.category_name,
                "f1_dev": round(f1_dev, 4),
                "f1_blank": round(f1_blank, 4),
                "delta": round(f1_dev - f1_blank, 4),
            })

        avg_dev = sum(r["f1_dev"] for r in conv_results) / len(conv_results)
        avg_blank = sum(r["f1_blank"] for r in conv_results) / len(conv_results)
        dev_wins = sum(1 for r in conv_results if r["delta"] > 0.01)
        blank_wins = sum(1 for r in conv_results if r["delta"] < -0.01)

        print(f"  Conv {ci + 1} ({conv.sample_id}): "
              f"dev={avg_dev:.3f} blank={avg_blank:.3f} "
              f"delta={avg_dev - avg_blank:+.3f} "
              f"wins={dev_wins}:{blank_wins}", flush=True)

        all_results.extend(conv_results)

    # Summary
    total_time = time.perf_counter() - t0
    total_qa = len(all_results)
    avg_dev = sum(r["f1_dev"] for r in all_results) / total_qa
    avg_blank = sum(r["f1_blank"] for r in all_results) / total_qa
    dev_wins = sum(1 for r in all_results if r["delta"] > 0.01)
    blank_wins = sum(1 for r in all_results if r["delta"] < -0.01)

    print(f"\n{'=' * 60}", flush=True)
    print("FULL LOCOMO DEVELOPMENTAL RESULTS", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Conversations: {len(conversations)}")
    print(f"Development turns: {total_turns}")
    print(f"QAs evaluated: {total_qa}")
    print(f"Developed F1: {avg_dev:.4f}")
    print(f"Blank F1:     {avg_blank:.4f}")
    print(f"Delta:        {avg_dev - avg_blank:+.4f}")
    print(f"Dev wins: {dev_wins}, Blank wins: {blank_wins}, "
          f"Ties: {total_qa - dev_wins - blank_wins}")

    soma = loop.predictive_soma.soma
    print(f"\nFinal graph: {len(soma.graph.nodes)} nodes, "
          f"{len(soma.graph.edges)} edges")
    print(f"Total memories: {len(loop.predictive_soma.text_store)}")
    print(f"Total time: {total_time:.0f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "locomo_full_development.json"
    with open(out_path, "w") as f:
        json.dump({
            "total_turns": total_turns,
            "total_qa": total_qa,
            "avg_f1_dev": round(avg_dev, 4),
            "avg_f1_blank": round(avg_blank, 4),
            "dev_wins": dev_wins,
            "blank_wins": blank_wins,
            "dev_time_s": round(dev_time, 1),
            "total_time_s": round(total_time, 1),
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
