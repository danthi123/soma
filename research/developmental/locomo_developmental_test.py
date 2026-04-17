"""Test developmental SOMA with LoCoMo structured conversations.

Feeds a real multi-session conversation through PredictiveSOMA
(one turn at a time, in order), then tests whether the developed
state enables better QA responses than an undeveloped state.

This is a developmental version of the graph reranking test — instead
of testing retrieval, we test whether SOMA's structural development
through conversation exposure improves its ability to support QA.

Usage::

    python -m research.developmental.locomo_developmental_test
    python -m research.developmental.locomo_developmental_test --qa-limit 10
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


def _call_llm(
    question: str,
    soma_state: str,
    api_base: str,
    model: str,
) -> str:
    """Ask the LLM a question with SOMA's state as context."""
    system_prompt = (
        "You are the voice of a developing mind. You have been exposed "
        "to conversations and have developed internal associations and "
        "memories. Use your internal state to answer the question. "
        "Be concise. If you don't have enough information, say so."
    )
    user_msg = f"{soma_state}\n\nQuestion: {question}\nAnswer:"
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
        return f"ERROR: {exc}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LoCoMo developmental test",
    )
    parser.add_argument("--conv-idx", type=int, default=0,
                        help="Which conversation to use (0-9)")
    parser.add_argument("--qa-limit", type=int, default=20,
                        help="Max QA pairs to evaluate")
    parser.add_argument("--model", default="qwen3:0.6b",
                        help="LLM model (small = better test)")
    parser.add_argument("--api-base", default="http://localhost:11434")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else "cpu"
    )
    print(f"Device: {device}", flush=True)

    # Load conversation
    conversations = load_dataset()
    conv = conversations[args.conv_idx]
    print(
        f"Conversation: {conv.sample_id} "
        f"({conv.num_sessions} sessions, {conv.total_turns} turns, "
        f"{conv.num_qa} QAs)",
        flush=True,
    )

    # Build corpus for tokenizer
    corpus = []
    for session in conv.sessions:
        for turn in session.turns:
            corpus.append(
                f"[{session.date_time}] {turn.speaker}: {turn.text}"
            )

    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        synaptogenesis_interval=20,
        neurogenesis_interval=50,
        pruning_interval=100,
        consolidation_interval=50,
        consolidation_replay_steps=20,
        seed=42,
    )

    loop = InteractionLoop(
        config=config,
        llm_model=args.model,
        llm_api_base=args.api_base,
        device=device,
    )
    loop.train_tokenizer(corpus)

    # Phase 1: Feed all conversation turns through SOMA (development)
    print(f"\n--- Phase 1: Development ({conv.total_turns} turns) ---",
          flush=True)
    t0 = time.perf_counter()

    for si, session in enumerate(conv.sessions):
        for turn in session.turns:
            text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
            loop.process_input(text, call_llm=False)

        soma = loop.predictive_soma.soma
        if (si + 1) % 5 == 0 or si == len(conv.sessions) - 1:
            n_nodes = len(soma.graph.nodes)
            n_edges = len(soma.graph.edges)
            ep = int(soma.episodic_memory.valid.sum().item())
            print(
                f"  Session {si + 1}/{conv.num_sessions}: "
                f"nodes={n_nodes}, edges={n_edges}, episodes={ep}",
                flush=True,
            )

    dev_time = time.perf_counter() - t0
    print(f"  Development complete in {dev_time:.1f}s", flush=True)

    # Get developed state
    developed_state = verbalize_state(loop.predictive_soma.soma)
    print(f"\n--- Developed SOMA State ---", flush=True)
    print(developed_state, flush=True)

    # Phase 2: QA evaluation — developed vs blank state
    qa_pairs = conv.qa_pairs[:args.qa_limit]
    print(
        f"\n--- Phase 2: QA Evaluation ({len(qa_pairs)} questions) ---",
        flush=True,
    )

    blank_state = (
        "[SOMA Internal State]\n"
        "Developmental Stage: blank-slate (step 0)\n"
        "Novelty Score: 0.000\n"
        "No associations, no memories, no development.\n"
    )

    results = []
    for qi, qa in enumerate(qa_pairs):
        # With developed SOMA
        hyp_dev = _call_llm(
            qa.question, developed_state, args.api_base, args.model,
        )
        # With blank SOMA
        hyp_blank = _call_llm(
            qa.question, blank_state, args.api_base, args.model,
        )

        f1_dev = token_f1(hyp_dev, str(qa.answer))
        f1_blank = token_f1(hyp_blank, str(qa.answer))
        delta = f1_dev - f1_blank

        results.append({
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category_name,
            "f1_developed": round(f1_dev, 4),
            "f1_blank": round(f1_blank, 4),
            "delta": round(delta, 4),
            "hyp_developed": hyp_dev,
            "hyp_blank": hyp_blank,
        })

        winner = "DEV" if delta > 0.01 else "BLANK" if delta < -0.01 else "TIE"
        print(
            f"  [{qi + 1:>2d}/{len(qa_pairs)}] "
            f"{qa.category_name:<12s} "
            f"dev={f1_dev:.3f} blank={f1_blank:.3f} "
            f"delta={delta:+.3f} {winner}",
            flush=True,
        )

    # Summary
    total_time = time.perf_counter() - t0
    dev_wins = sum(1 for r in results if r["delta"] > 0.01)
    blank_wins = sum(1 for r in results if r["delta"] < -0.01)
    ties = len(results) - dev_wins - blank_wins
    avg_f1_dev = sum(r["f1_developed"] for r in results) / len(results)
    avg_f1_blank = sum(r["f1_blank"] for r in results) / len(results)

    print(f"\n{'=' * 50}", flush=True)
    print("LOCOMO DEVELOPMENTAL TEST RESULTS", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"Developed F1: {avg_f1_dev:.4f}")
    print(f"Blank F1:     {avg_f1_blank:.4f}")
    print(f"Delta:        {avg_f1_dev - avg_f1_blank:+.4f}")
    print(f"Dev wins: {dev_wins}, Blank wins: {blank_wins}, Ties: {ties}")
    print(f"Total time: {total_time:.0f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "locomo_developmental_test.json"
    with open(out_path, "w") as f:
        json.dump({
            "conversation": conv.sample_id,
            "n_sessions": conv.num_sessions,
            "n_turns": conv.total_turns,
            "n_qa": len(qa_pairs),
            "development_time_s": round(dev_time, 1),
            "total_time_s": round(total_time, 1),
            "avg_f1_developed": round(avg_f1_dev, 4),
            "avg_f1_blank": round(avg_f1_blank, 4),
            "dev_wins": dev_wins,
            "blank_wins": blank_wins,
            "ties": ties,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
