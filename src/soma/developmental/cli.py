"""Interactive CLI for talking to the developing SOMA.

Usage::

    python -m soma.developmental.cli
    python -m soma.developmental.cli --model qwen2:1.5b-instruct-q8_0
"""
from __future__ import annotations

import argparse

import torch

from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop

_SEED_CORPUS = [
    "hello how are you",
    "what is the meaning of life",
    "tell me something interesting",
    "I feel happy today",
    "the weather is nice outside",
    "can you help me understand",
    "memory learning growth development",
    "curiosity exploration novelty prediction",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the developing SOMA mind.",
    )
    parser.add_argument(
        "--model",
        default="qwen2:1.5b-instruct-q8_0",
        help="Ollama model name (default: qwen2:1.5b-instruct-q8_0)",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Print SOMA internal state after each response",
    )
    parser.add_argument(
        "--consolidate-every",
        type=int,
        default=0,
        help="Run consolidation every N steps (0 = never)",
    )
    parser.add_argument(
        "--load",
        default=None,
        help="Load a saved developmental state from this directory",
    )
    parser.add_argument(
        "--save-dir",
        default="soma_dev_state",
        help="Directory to save state on /save or exit (default: soma_dev_state)",
    )
    parser.add_argument(
        "--no-train-encoder",
        action="store_true",
        help="Disable encoder training (embeddings stay fixed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the interactive SOMA CLI."""
    args = _parse_args(argv)
    device = torch.device(args.device)

    config = SOMAConfig.developmental()
    loop = InteractionLoop(
        config=config,
        llm_model=args.model,
        llm_api_base=args.api_base,
        device=device,
        train_encoder=not args.no_train_encoder,
    )

    # Seed tokenizer
    loop.train_tokenizer(_SEED_CORPUS)

    # Load saved state if requested
    if args.load:
        print(f"Loading state from {args.load}...")
        loop.load(args.load)
        print(f"  Loaded: step {loop.predictive_soma.soma.global_step}, "
              f"{len(loop.predictive_soma.text_store)} memories")

    print("SOMA Developmental CLI")
    print(f"Model: {args.model} | Device: {device}")
    print("Commands: /state /stats /report /recall <q> /sleep /save /quit\n")

    step = 0
    try:
        while True:
            try:
                user_input = input("You> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # ---- Commands ------------------------------------------------
            if user_input.lower() == "/quit":
                break

            if user_input.lower() == "/state":
                # Process a no-op just to get current state text
                from soma.developmental.verbalize import verbalize_state

                print(verbalize_state(loop.predictive_soma.soma))
                continue

            if user_input.lower() == "/stats":
                summary = loop.tracker.summary()
                for metric, stats in summary.items():
                    print(
                        f"  {metric}: mean={stats['mean']:.4f}  "
                        f"last={stats['last']:.4f}  "
                        f"(n={int(stats['count'])})"
                    )
                continue

            if user_input.lower() == "/save":
                loop.save(args.save_dir)
                print(f"State saved to {args.save_dir}/")
                continue

            if user_input.lower() == "/report":
                soma = loop.predictive_soma.soma
                ps = loop.predictive_soma
                n_nodes = len(soma.graph.nodes)
                n_edges = len(soma.graph.edges)
                n_mem = len(ps.text_store)
                stage = "blank-slate" if soma.global_step < 100 else \
                        "early-plasticity" if soma.global_step < 500 else \
                        "pattern-recognition" if soma.global_step < 2000 else \
                        "association-formation" if soma.global_step < 5000 else \
                        "mature"
                errors = list(ps.error_history)
                avg_err = sum(errors[-50:]) / max(len(errors[-50:]), 1)
                print("\n  Development Report")
                print(f"  Stage: {stage} (step {soma.global_step})")
                print(f"  Graph: {n_nodes} nodes, {n_edges} edges")
                print(f"  Memories: {n_mem} stored")
                print(f"  Prediction error (recent): {avg_err:.6f}")
                if errors:
                    first = sum(errors[:10]) / min(len(errors), 10)
                    last = sum(errors[-10:]) / min(len(errors), 10)
                    if first > 0:
                        print(f"  Learning: {(1 - last/first) * 100:.0f}% error reduction")
                summary = loop.tracker.summary()
                if "num_edges" in summary:
                    print(f"  Edge growth: {summary['num_edges']['min']:.0f}"
                          f" -> {summary['num_edges']['last']:.0f}")
                print()
                continue

            if user_input.lower() == "/sleep":
                soma = loop.predictive_soma.soma
                print("  Consolidating (sleep cycle)...", flush=True)
                n_before = len(soma.graph.edges)
                soma._maybe_consolidate(rng=None)
                n_after = len(soma.graph.edges)
                delta_e = n_after - n_before
                print(f"  Sleep complete. Edges: {n_before} -> {n_after} ({delta_e:+d})")
                continue

            if user_input.lower().startswith("/recall "):
                query = user_input[8:].strip()
                if query:
                    qvec = loop.encode_text(query)
                    recalled = loop.predictive_soma.retrieve_by_graph(qvec, top_k=5)
                    print(f"\n  Memories recalled for \"{query}\":")
                    if recalled:
                        for s, text, sim in recalled:
                            print(f"    [{s}] (sim={sim:.3f}) {text[:100]}")
                    else:
                        print("    (none)")
                    print()
                continue

            # ---- Normal interaction --------------------------------------
            result = loop.process_input(user_input, call_llm=True)
            step += 1

            soma = loop.predictive_soma.soma
            n = len(soma.graph.nodes)
            e = len(soma.graph.edges)
            m = len(loop.predictive_soma.text_store)

            if result["response"]:
                print(f"\nSOMA> {result['response']}\n")
            else:
                print("\nSOMA> (no response)\n")

            # Dev metrics — compact status line
            print(
                f"  [step {result['global_step']}  "
                f"graph:{n}n/{e}e  mem:{m}  "
                f"pred_err={result['prediction_error']:.4f}  "
                f"novelty={result['novelty']:.4f}]"
            )

            if args.show_state:
                print(result["soma_state"])

            # Optional consolidation
            if args.consolidate_every > 0 and step % args.consolidate_every == 0:
                soma = loop.predictive_soma.soma
                if hasattr(soma, "consolidation_cycle"):
                    soma.consolidation_cycle(result["global_step"])
                    print("  [consolidation cycle complete]")

    except KeyboardInterrupt:
        print("\n")

    # Save tracker on exit
    log_path = "soma_development_log.json"
    loop.tracker.save(log_path)
    print(f"Development log saved to {log_path}")


if __name__ == "__main__":
    main()
