"""Longitudinal evaluation — show development improves QA over time.

Feeds text through PredictiveSOMA and evaluates QA ability at regular
intervals (every N steps). The "development curve" shows whether longer
interaction leads to better answers.

Uses a 4B LLM for QA evaluation (weak enough to need SOMA's state,
strong enough to extract answers from recalled memories).

Usage::

    python -m research.developmental.longitudinal_eval
    python -m research.developmental.longitudinal_eval --eval-every 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
import torch

from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop
from soma.developmental.verbalize import verbalize_state

RESULTS_DIR = Path("research/developmental/results")

# Development corpus — diverse personal facts
DEVELOPMENT_CORPUS = [
    "I love cooking Italian food, especially pasta and risotto.",
    "My grandmother taught me her secret tomato sauce recipe.",
    "Rome was amazing, the food there was incredible.",
    "I want to visit Japan next spring for the cherry blossoms.",
    "Learning guitar has been my hobby for about six months.",
    "Jazz music is incredibly complex but beautiful.",
    "The James Webb telescope images are breathtaking.",
    "I started running three times a week this month.",
    "My cat Nebula loves sitting by the window all day.",
    "I read a great book about neuroscience yesterday.",
    "Cooking relaxes me after a long day at work.",
    "Italian food is all about fresh ingredients and simplicity.",
    "My grandmother made the best tomato sauce I've ever tasted.",
    "I practiced guitar scales this morning before work.",
    "The concert last weekend was one of the best I've been to.",
    "Quantum computing could revolutionize cryptography.",
    "Running in the morning gives me energy for the whole day.",
    "Nebula knocked over my coffee again this morning.",
    "I want to try making risotto with saffron this weekend.",
    "Travel teaches you so much about yourself and the world.",
    "I'm thinking about adopting another cat to keep Nebula company.",
    "My favorite pasta shape to make by hand is fettuccine.",
    "Barcelona's architecture blew my mind, especially Gaudi.",
    "I've been learning music theory and it changed how I listen.",
    "The ocean floor is less explored than the surface of Mars.",
    "Yoga has really helped with my back pain from sitting all day.",
    "My grandmother always said cooking is an act of love.",
    "I bought new trail running shoes for the half marathon.",
    "Nebula has a funny habit of chirping at birds through the glass.",
    "I find evolutionary biology absolutely fascinating.",
]

# Evaluation questions — things the LLM cannot know without SOMA
EVAL_QUESTIONS = [
    ("What do I like to cook?", "Italian food, pasta, risotto, fettuccine"),
    ("What is my cat's name?", "Nebula"),
    ("Tell me about my grandmother.", "She taught me secret recipes, made the best tomato sauce, said cooking is an act of love"),
    ("What musical instrument am I learning?", "guitar"),
    ("Where do I want to travel?", "Japan for cherry blossoms, Rome, Barcelona"),
    ("What exercise do I do?", "running three times a week, yoga, training for half marathon"),
    ("What science topics interest me?", "quantum computing, neuroscience, James Webb telescope, evolutionary biology, ocean exploration"),
    ("What does my cat do?", "sits by the window, knocks over coffee, chirps at birds"),
]


def _call_llm(
    question: str,
    state: str,
    api_base: str,
    model: str,
) -> str:
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


def evaluate_qa(
    loop: InteractionLoop,
    questions: list[tuple[str, str]],
    api_base: str,
    model: str,
) -> dict[str, Any]:
    """Evaluate QA ability at current developmental state."""
    from benchmarks.industry.longmemeval.metrics import token_f1

    text_store = loop.predictive_soma.text_store
    soma = loop.predictive_soma.soma

    results = []
    for question, reference in questions:
        state = verbalize_state(
            soma, query=question, text_store=text_store,
        )
        hypothesis = _call_llm(question, state, api_base, model)
        f1 = token_f1(hypothesis, reference)
        results.append({
            "question": question,
            "reference": reference,
            "hypothesis": hypothesis,
            "f1": round(f1, 4),
        })

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    return {
        "avg_f1": round(avg_f1, 4),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Longitudinal developmental evaluation",
    )
    parser.add_argument("--eval-every", type=int, default=30,
                        help="Evaluate QA every N development steps")
    parser.add_argument("--cycles", type=int, default=10,
                        help="Number of corpus repetitions")
    parser.add_argument("--model", default="qwen3.5:4b-q8_0")
    parser.add_argument("--api-base", default="http://localhost:11434")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else "cpu"
    )
    print(f"Device: {device}", flush=True)

    config = SOMAConfig.developmental()
    loop = InteractionLoop(
        config=config,
        llm_model=args.model,
        llm_api_base=args.api_base,
        device=device,
    )
    loop.train_tokenizer(DEVELOPMENT_CORPUS)

    total_inputs = len(DEVELOPMENT_CORPUS) * args.cycles
    print(f"Development corpus: {len(DEVELOPMENT_CORPUS)} texts x "
          f"{args.cycles} cycles = {total_inputs} inputs", flush=True)
    print(f"Evaluate every {args.eval_every} steps", flush=True)
    print(f"LLM: {args.model}", flush=True)

    # Evaluate at step 0 (blank state)
    print(f"\n--- Eval at step 0 (blank) ---", flush=True)
    eval_0 = evaluate_qa(loop, EVAL_QUESTIONS, args.api_base, args.model)
    print(f"  Avg F1: {eval_0['avg_f1']:.4f}", flush=True)

    checkpoints: list[dict[str, Any]] = [{
        "step": 0,
        "avg_f1": eval_0["avg_f1"],
        "num_nodes": len(loop.predictive_soma.soma.graph.nodes),
        "num_edges": len(loop.predictive_soma.soma.graph.edges),
        "avg_pred_error": 0.0,
        "results": eval_0["results"],
    }]

    t0 = time.perf_counter()
    step = 0

    for cycle in range(args.cycles):
        for text in DEVELOPMENT_CORPUS:
            loop.process_input(text, call_llm=False)
            step += 1

            if step % args.eval_every == 0:
                soma = loop.predictive_soma.soma
                n_nodes = len(soma.graph.nodes)
                n_edges = len(soma.graph.edges)

                recent_errors = list(loop.predictive_soma.error_history)[-args.eval_every:]
                avg_pe = sum(recent_errors) / len(recent_errors) if recent_errors else 0

                print(f"\n--- Eval at step {step} "
                      f"(nodes={n_nodes}, edges={n_edges}, "
                      f"pred_err={avg_pe:.6f}) ---", flush=True)

                eval_result = evaluate_qa(
                    loop, EVAL_QUESTIONS, args.api_base, args.model,
                )
                print(f"  Avg F1: {eval_result['avg_f1']:.4f}", flush=True)

                # Show per-question scores
                for r in eval_result["results"]:
                    marker = "+" if r["f1"] > 0 else " "
                    print(f"  {marker} {r['question'][:40]:<40s} F1={r['f1']:.3f}",
                          flush=True)

                checkpoints.append({
                    "step": step,
                    "avg_f1": eval_result["avg_f1"],
                    "num_nodes": n_nodes,
                    "num_edges": n_edges,
                    "avg_pred_error": round(avg_pe, 6),
                    "results": eval_result["results"],
                })

    total_time = time.perf_counter() - t0

    # Summary: development curve
    print(f"\n{'=' * 60}", flush=True)
    print("DEVELOPMENT CURVE", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"{'Step':>6s} {'F1':>8s} {'Nodes':>6s} {'Edges':>6s} {'PredErr':>10s}",
          flush=True)
    print("-" * 40, flush=True)
    for cp in checkpoints:
        print(f"{cp['step']:>6d} {cp['avg_f1']:>8.4f} "
              f"{cp['num_nodes']:>6d} {cp['num_edges']:>6d} "
              f"{cp['avg_pred_error']:>10.6f}", flush=True)

    print(f"\nTotal time: {total_time:.0f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "longitudinal_eval.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "eval_every": args.eval_every,
                "cycles": args.cycles,
                "model": args.model,
                "total_inputs": total_inputs,
            },
            "checkpoints": checkpoints,
            "total_time_s": round(total_time, 1),
        }, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
