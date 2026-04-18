"""Test graph-topology retrieval vs fingerprint vs token overlap.

Topology retrieval uses SOMA's learned graph structure — which nodes
co-activate for which memories — to find associations. This is SOMA's
genuine contribution: cross-domain associations that neither token
overlap nor fingerprint comparison can provide.
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conversations = load_dataset()
    total_turns = sum(c.total_turns for c in conversations)
    print(f"Loaded {len(conversations)} convs, {total_turns} turns")

    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(f"[{session.date_time}] {turn.speaker}: {turn.text}")

    config = SOMAConfig.developmental()
    loop = InteractionLoop(
        config=config, llm_model="unused", device=device,
    )
    loop.train_tokenizer(corpus[:500])

    # Development
    print(f"\n--- Development ({total_turns} turns) ---", flush=True)
    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                loop.process_input(text, call_llm=False)

        ps = loop.predictive_soma
        n_idx = sum(len(v) for v in ps._node_memory_index.values())
        print(f"  Conv {ci+1}/10: nodes={len(ps.soma.graph.nodes)}, "
              f"mem={len(ps.text_store)}, "
              f"node_index_entries={n_idx}", flush=True)

    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s", flush=True)

    ps = loop.predictive_soma

    # Strategy 1: Graph fingerprint (current default)
    print("\n--- Evaluation ---", flush=True)
    t1 = time.perf_counter()
    graph_scores = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qvec = loop.encode_text(qa.question, keep_grad=False)
            results = ps.retrieve_by_graph(qvec, top_k=5)
            best = max(
                (token_f1(t, str(qa.answer)) for _, t, _ in results),
                default=0.0,
            )
            graph_scores.append(best)
    graph_time = time.perf_counter() - t1
    graph_hits = sum(1 for s in graph_scores if s > 0.05)
    graph_f1 = sum(graph_scores) / len(graph_scores) if graph_scores else 0
    print(f"  graph_fingerprint : F1={graph_f1:.4f}  "
          f"hits={graph_hits}/{len(graph_scores)}  ({graph_time:.1f}s)",
          flush=True)

    # Strategy 2: Topology (shared active nodes)
    t2 = time.perf_counter()
    topo_scores = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qvec = loop.encode_text(qa.question, keep_grad=False)
            results = ps.retrieve_by_topology(qvec, top_k=5)
            best = max(
                (token_f1(t, str(qa.answer)) for _, t, _ in results),
                default=0.0,
            )
            topo_scores.append(best)
    topo_time = time.perf_counter() - t2
    topo_hits = sum(1 for s in topo_scores if s > 0.05)
    topo_f1 = sum(topo_scores) / len(topo_scores) if topo_scores else 0
    print(f"  graph_topology    : F1={topo_f1:.4f}  "
          f"hits={topo_hits}/{len(topo_scores)}  ({topo_time:.1f}s)",
          flush=True)

    # Strategy 3: Token overlap (baseline to beat)
    t3 = time.perf_counter()
    token_scores = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            q_tokens = set(loop._encoder.tokenize(qa.question))
            results = ps.retrieve_by_tokens(q_tokens, top_k=5)
            best = max(
                (token_f1(t, str(qa.answer)) for _, t, _ in results),
                default=0.0,
            )
            token_scores.append(best)
    token_time = time.perf_counter() - t3
    token_hits = sum(1 for s in token_scores if s > 0.05)
    token_f1_avg = sum(token_scores) / len(token_scores) if token_scores else 0
    print(f"  token_overlap     : F1={token_f1_avg:.4f}  "
          f"hits={token_hits}/{len(token_scores)}  ({token_time:.1f}s)",
          flush=True)

    # Per-query comparison: where does topology win?
    print("\n--- Where Topology Wins ---", flush=True)
    wins = 0
    for i, (conv_idx, qa) in enumerate(
        (ci, qa)
        for ci, conv in enumerate(conversations)
        for qa in conv.qa_pairs[:args.qa_per_conv]
    ):
        if topo_scores[i] > graph_scores[i] + 0.01:
            wins += 1
            if wins <= 5:
                print(f"  Q: {qa.question[:80]}")
                print(f"    topo={topo_scores[i]:.3f} graph={graph_scores[i]:.3f} "
                      f"token={token_scores[i]:.3f}")
    print(f"  Topology wins: {wins}/{len(topo_scores)}")


if __name__ == "__main__":
    main()
