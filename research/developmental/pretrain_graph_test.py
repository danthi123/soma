"""Test whether pre-training SOMA's graph on a large diverse corpus
improves retrieval on LoCoMo.

Hypothesis: SOMA's graph needs diverse, high-volume data to develop
genuine node specialization via competitive learning and anti-Hebbian
dynamics. 5882 turns from 10 narrow conversations isn't enough --
200K turns from 19K diverse sessions should let the graph learn
general activation patterns that transfer.

The BPE encoder stays untrained (random embeddings) -- this is honest.
Only the graph structure (node weights, edge weights, synaptic
connections) learns from the pre-training corpus.

Comparison:
  A. Baseline: develop on LoCoMo only (5882 turns) -> test
  B. Pre-trained: develop on LongMemEval-S (200K turns) -> clear
     memory stores -> develop on LoCoMo -> test

The graph's learned structure (node weights, edge strengths, growth
patterns) should transfer. Memory stores (fingerprints, text, tokens)
are cleared between phases -- only the graph itself carries over.
"""
from __future__ import annotations

import json
import time

import torch

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop


def load_longmemeval_sessions(max_sessions: int | None = None) -> list[list[dict]]:
    """Extract unique sessions from LongMemEval-S."""
    path = "benchmarks/industry/longmemeval/data/longmemeval_s_cleaned.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    sessions_by_id: dict[str, dict] = {}
    for item in data:
        for sid, session, date in zip(
            item["haystack_session_ids"],
            item["haystack_sessions"],
            item["haystack_dates"],
        ):
            if sid not in sessions_by_id:
                sessions_by_id[sid] = {"date": date, "turns": session}

    sessions = list(sessions_by_id.values())
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    return sessions


def run_retrieval(
    loop: InteractionLoop,
    conversations,
    qa_per_conv: int,
    label: str,
) -> dict:
    """Run all three retrieval strategies and print results."""
    ps = loop.predictive_soma
    results = {}

    # Graph fingerprint
    t0 = time.perf_counter()
    scores = []
    for conv in conversations:
        for qa in conv.qa_pairs[:qa_per_conv]:
            qvec = loop.encode_text(qa.question, keep_grad=False)
            hits = ps.retrieve_by_graph(qvec, top_k=5)
            best = max(
                (token_f1(t, str(qa.answer)) for _, t, _ in hits),
                default=0.0,
            )
            scores.append(best)
    elapsed = time.perf_counter() - t0
    n_hits = sum(1 for s in scores if s > 0.05)
    avg = sum(scores) / len(scores) if scores else 0
    print(f"  [{label}] graph_fingerprint : F1={avg:.4f}  hits={n_hits}/100  ({elapsed:.1f}s)")
    results["graph_fp"] = {"f1": avg, "hits": n_hits}

    # Topology
    t0 = time.perf_counter()
    scores = []
    for conv in conversations:
        for qa in conv.qa_pairs[:qa_per_conv]:
            qvec = loop.encode_text(qa.question, keep_grad=False)
            hits = ps.retrieve_by_topology(qvec, top_k=5)
            best = max(
                (token_f1(t, str(qa.answer)) for _, t, _ in hits),
                default=0.0,
            )
            scores.append(best)
    elapsed = time.perf_counter() - t0
    n_hits = sum(1 for s in scores if s > 0.05)
    avg = sum(scores) / len(scores) if scores else 0
    print(f"  [{label}] graph_topology    : F1={avg:.4f}  hits={n_hits}/100  ({elapsed:.1f}s)")
    results["graph_topo"] = {"f1": avg, "hits": n_hits}

    # Token overlap (baseline)
    t0 = time.perf_counter()
    scores = []
    for conv in conversations:
        for qa in conv.qa_pairs[:qa_per_conv]:
            q_tokens = set(loop._encoder.tokenize(qa.question))
            hits = ps.retrieve_by_tokens(q_tokens, top_k=5)
            best = max(
                (token_f1(t, str(qa.answer)) for _, t, _ in hits),
                default=0.0,
            )
            scores.append(best)
    elapsed = time.perf_counter() - t0
    n_hits = sum(1 for s in scores if s > 0.05)
    avg = sum(scores) / len(scores) if scores else 0
    print(f"  [{label}] token_overlap     : F1={avg:.4f}  hits={n_hits}/100  ({elapsed:.1f}s)")
    results["token_overlap"] = {"f1": avg, "hits": n_hits}

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    parser.add_argument("--pretrain-sessions", type=int, default=2000,
                        help="Number of LongMemEval sessions for pre-training "
                             "(default 2000, about 20K turns)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the no-pretrain baseline (saves about 6 min)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load LoCoMo for assessment
    conversations = load_dataset()
    total_turns = sum(c.total_turns for c in conversations)
    print(f"LoCoMo: {len(conversations)} convs, {total_turns} turns")

    # Build BPE tokenizer from LoCoMo corpus
    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(f"[{session.date_time}] {turn.speaker}: {turn.text}")

    # ================================================================
    # A. BASELINE: develop on LoCoMo only
    # ================================================================
    if not args.skip_baseline:
        print("\n=== A. BASELINE (LoCoMo only) ===")
        config_a = SOMAConfig.developmental()
        loop_a = InteractionLoop(config=config_a, llm_model="unused", device=device)
        loop_a.train_tokenizer(corpus[:500])

        t0 = time.perf_counter()
        for ci, conv in enumerate(conversations):
            for session in conv.sessions:
                for turn in session.turns:
                    text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                    loop_a.process_input(text, call_llm=False)
            ps = loop_a.predictive_soma
            print(f"  Conv {ci+1}/10: nodes={len(ps.soma.graph.nodes)}, "
                  f"mem={len(ps.text_store)}", flush=True)
        baseline_time = time.perf_counter() - t0
        print(f"  Development: {baseline_time:.1f}s")

        run_retrieval(loop_a, conversations, args.qa_per_conv, "baseline")
        del loop_a
        torch.cuda.empty_cache()

    # ================================================================
    # B. PRE-TRAINED: LongMemEval-S then LoCoMo
    # ================================================================
    print(f"\n=== B. PRE-TRAINED ({args.pretrain_sessions} sessions) ===")

    # Load pre-training data
    lme_sessions = load_longmemeval_sessions(max_sessions=args.pretrain_sessions)
    pretrain_turns = sum(len(s["turns"]) for s in lme_sessions)
    print(f"Pre-training corpus: {len(lme_sessions)} sessions, {pretrain_turns} turns")

    config_b = SOMAConfig.developmental()
    loop_b = InteractionLoop(config=config_b, llm_model="unused", device=device)
    loop_b.train_tokenizer(corpus[:500])

    # Phase 1: Pre-train graph on diverse corpus
    print("\n--- Phase 1: Pre-train graph ---", flush=True)
    t0 = time.perf_counter()
    step_count = 0
    for si, session in enumerate(lme_sessions):
        for turn in session["turns"]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            text = f"[{session['date']}] {role}: {content}"
            loop_b.process_input(text, call_llm=False)
            step_count += 1

        if (si + 1) % 200 == 0:
            ps = loop_b.predictive_soma
            print(f"  Session {si+1}/{len(lme_sessions)}: "
                  f"nodes={len(ps.soma.graph.nodes)}, "
                  f"steps={step_count}, "
                  f"pred_err={ps.prediction_error:.6f}",
                  flush=True)

    pretrain_time = time.perf_counter() - t0
    ps = loop_b.predictive_soma
    print(f"  Pre-training done: {pretrain_time:.1f}s, "
          f"{step_count} steps, "
          f"nodes={len(ps.soma.graph.nodes)}, "
          f"edges={len(ps.soma.graph.edges)}")

    # Phase 2: Clear memory stores but KEEP graph structure
    # The graph's learned weights, edge strengths, and node
    # specializations carry over. Only the retrieval index is
    # reset so LoCoMo memories are stored fresh.
    print("\n--- Phase 2: Clear memory stores, keep graph ---", flush=True)
    ps.text_store.clear()
    ps._activation_store.clear()
    ps._token_cache.clear()
    ps._node_memory_index.clear()
    # Reset prediction head state (it trained on LME patterns)
    ps._last_summary = None
    ps._last_prediction = None
    ps.error_history.clear()
    print(f"  Cleared. Graph preserved: nodes={len(ps.soma.graph.nodes)}, "
          f"edges={len(ps.soma.graph.edges)}")

    # Phase 3: Develop on LoCoMo (same as baseline)
    print("\n--- Phase 3: Develop on LoCoMo ---", flush=True)
    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                loop_b.process_input(text, call_llm=False)
        print(f"  Conv {ci+1}/10: nodes={len(ps.soma.graph.nodes)}, "
              f"mem={len(ps.text_store)}", flush=True)
    locomo_time = time.perf_counter() - t0
    print(f"  LoCoMo development: {locomo_time:.1f}s")

    # Phase 4: Assess
    print("\n--- Phase 4: Assessment ---", flush=True)
    run_retrieval(loop_b, conversations, args.qa_per_conv, "pretrained")


if __name__ == "__main__":
    main()
