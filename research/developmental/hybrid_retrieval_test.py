"""Test hybrid retrieval: graph fingerprint + BPE token overlap.

The graph fingerprint captures structural similarity (SOMA's contribution).
BPE token overlap captures character-level similarity (no SOMA needed).
Combining both should outperform either alone.

This is NOT cheating — the BPE tokenizer is part of SOMA's input pipeline
and doesn't use pretrained embeddings. Token overlap is what a simple
hash-based memory system would give; SOMA's contribution is the graph
fingerprint ON TOP of that.
"""
from __future__ import annotations

import time

import torch

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop


def retrieval_graph_only(
    loop: InteractionLoop, question: str, top_k: int = 5,
) -> list[tuple[int, str, float]]:
    """Standard graph-based retrieval."""
    qvec = loop.encode_text(question, keep_grad=False)
    return loop.predictive_soma.retrieve_by_graph(qvec, top_k=top_k)


def retrieval_token_overlap(
    loop: InteractionLoop, question: str, top_k: int = 5,
) -> list[tuple[int, str, float]]:
    """BPE token overlap retrieval (no SOMA graph)."""
    if loop._encoder is None:
        return []
    ps = loop.predictive_soma

    # Tokenize the query
    q_ids = set(loop._encoder.tokenize(question))

    # Score each stored text by token overlap
    scored: list[tuple[float, int]] = []
    for step, text in ps.text_store.items():
        t_ids = set(loop._encoder.tokenize(text))
        if not q_ids or not t_ids:
            scored.append((0.0, step))
            continue
        overlap = len(q_ids & t_ids)
        union = len(q_ids | t_ids)
        jaccard = overlap / union if union > 0 else 0.0
        scored.append((jaccard, step))

    scored.sort(key=lambda t: -t[0])
    results = []
    for sim, step in scored[:top_k]:
        text = ps.text_store.get(step, "")
        if text:
            results.append((step, text, sim))
    return results


def retrieval_hybrid(
    loop: InteractionLoop,
    question: str,
    top_k: int = 5,
    graph_weight: float = 0.5,
) -> list[tuple[int, str, float]]:
    """Hybrid: weighted combination of graph sim and token overlap."""
    if loop._encoder is None:
        return []
    ps = loop.predictive_soma

    # Graph fingerprint scores
    qvec = loop.encode_text(question, keep_grad=False)
    modality = ps.config.input_modalities[0]
    ps.soma.step({modality: qvec}, eval_mode=True)
    ps._diversify_activations(qvec)
    ps._apply_lateral_inhibition()
    query_fp = ps._get_node_fingerprint()

    steps = list(ps._activation_store.keys())
    stored_matrix = torch.stack([ps._activation_store[s] for s in steps])
    graph_sims = torch.nn.functional.cosine_similarity(
        query_fp.unsqueeze(0), stored_matrix, dim=1,
    )

    # Token overlap scores
    q_ids = set(loop._encoder.tokenize(question))
    token_scores = torch.zeros(len(steps))
    for i, step in enumerate(steps):
        text = ps.text_store.get(step, "")
        if text:
            t_ids = set(loop._encoder.tokenize(text))
            overlap = len(q_ids & t_ids)
            union = len(q_ids | t_ids)
            token_scores[i] = overlap / union if union > 0 else 0.0

    # Hybrid score
    combined = graph_weight * graph_sims.cpu() + (1 - graph_weight) * token_scores

    k = min(top_k, len(steps))
    top_scores, top_indices = torch.topk(combined, k)

    results = []
    for i in range(k):
        step = steps[top_indices[i].item()]
        text = ps.text_store.get(step, "")
        if text:
            results.append((step, text, float(top_scores[i].item())))
    return results


STRATEGIES = {
    "graph_only": retrieval_graph_only,
    "token_overlap": retrieval_token_overlap,
    "hybrid_0.3": lambda l, q: retrieval_hybrid(l, q, graph_weight=0.3),
    "hybrid_0.5": lambda l, q: retrieval_hybrid(l, q, graph_weight=0.5),
    "hybrid_0.7": lambda l, q: retrieval_hybrid(l, q, graph_weight=0.7),
}


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
        print(f"  Conv {ci+1}/10", flush=True)

    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s", flush=True)

    # Evaluate each strategy
    print(f"\n--- Retrieval Strategies ({args.qa_per_conv} QA/conv) ---",
          flush=True)

    for name, fn in STRATEGIES.items():
        all_scores = []
        t1 = time.perf_counter()

        for conv in conversations:
            qa_pairs = conv.qa_pairs[:args.qa_per_conv]
            for qa in qa_pairs:
                results = fn(loop, qa.question)
                best_f1 = 0.0
                answer = str(qa.answer)
                for _step, text, _sim in results:
                    f1 = token_f1(text, answer)
                    best_f1 = max(best_f1, f1)
                all_scores.append(best_f1)

        elapsed = time.perf_counter() - t1
        avg_f1 = sum(all_scores) / len(all_scores) if all_scores else 0
        hits = sum(1 for s in all_scores if s > 0.05)
        print(f"  {name:20s}: F1={avg_f1:.4f}  "
              f"hits={hits}/{len(all_scores)}  ({elapsed:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
