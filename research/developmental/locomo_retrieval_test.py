"""Test retrieval quality at LoCoMo scale WITHOUT LLM.

Develops on all 10 LoCoMo conversations, then for each QA pair,
checks if the retrieved memories contain text relevant to the answer.
Uses token overlap to score relevance (no LLM needed).

This gives fast feedback on whether architectural changes improve
retrieval at scale.
"""
from __future__ import annotations

import time

import torch

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conversations = load_dataset()
    total_turns = sum(c.total_turns for c in conversations)
    print(f"Loaded {len(conversations)} conversations, {total_turns} total turns")

    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(f"[{session.date_time}] {turn.speaker}: {turn.text}")

    config = SOMAConfig.developmental()
    loop = InteractionLoop(
        config=config,
        llm_model="unused",
        device=device,
        train_encoder=not args.no_train,
    )
    loop.train_tokenizer(corpus[:500])

    # Phase 1: Development
    print(f"\n--- Development ({total_turns} turns, "
          f"encoder_train={'OFF' if args.no_train else 'ON'}) ---")
    t0 = time.perf_counter()

    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                loop.process_input(text, call_llm=False)

        ps = loop.predictive_soma
        soma = ps.soma
        recent_err = list(ps.error_history)[-50:]
        avg_err = sum(recent_err) / len(recent_err) if recent_err else 0
        print(f"  Conv {ci + 1}/10: nodes={len(soma.graph.nodes)}, "
              f"edges={len(soma.graph.edges)}, mem={len(ps.text_store)}, "
              f"pred_err={avg_err:.6f}", flush=True)

    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s")

    # Phase 2: Retrieval evaluation
    print("\n--- Retrieval Evaluation ---")
    all_scores = []

    for ci, conv in enumerate(conversations):
        qa_pairs = conv.qa_pairs[:args.qa_per_conv]
        if not qa_pairs:
            continue

        conv_scores = []
        for qa in qa_pairs:
            # Use token-overlap retrieval (outperforms graph fingerprint
            # with random BPE embeddings — see hybrid_retrieval_test.py)
            q_tokens = set(loop._encoder.tokenize(qa.question))
            recalled = loop.predictive_soma.retrieve_by_tokens(
                q_tokens, top_k=5,
            )

            # Score: best token F1 between any recalled memory and the answer
            best_f1 = 0.0
            answer = str(qa.answer)
            for _, text, _sim in recalled:
                f1 = token_f1(text, answer)
                best_f1 = max(best_f1, f1)

            conv_scores.append(best_f1)

        avg = sum(conv_scores) / len(conv_scores) if conv_scores else 0
        hits = sum(1 for s in conv_scores if s > 0.05)
        print(f"  Conv {ci + 1} ({conv.sample_id}): "
              f"avg_f1={avg:.3f}, hits={hits}/{len(conv_scores)}", flush=True)
        all_scores.extend(conv_scores)

    total_qa = len(all_scores)
    avg_f1 = sum(all_scores) / total_qa if total_qa else 0
    total_hits = sum(1 for s in all_scores if s > 0.05)
    total_time = time.perf_counter() - t0

    print(f"\n{'=' * 50}")
    print(f"RETRIEVAL RESULTS (encoder_train={'OFF' if args.no_train else 'ON'})")
    print(f"{'=' * 50}")
    print(f"Total QA: {total_qa}")
    print(f"Avg recall F1: {avg_f1:.4f}")
    print(f"Recall hits (F1>0.05): {total_hits}/{total_qa}")
    print(f"Time: {total_time:.0f}s")

    ps = loop.predictive_soma
    print(f"Final graph: {len(ps.soma.graph.nodes)} nodes, "
          f"{len(ps.soma.graph.edges)} edges")
    print(f"Total memories: {len(ps.text_store)}")


if __name__ == "__main__":
    main()
