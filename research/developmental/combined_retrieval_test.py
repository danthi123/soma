"""Test combined retrieval: fingerprint + topology scores merged.

Fingerprint captures activation magnitude patterns.
Topology captures which nodes co-activated (structural associations).
Combining both should outperform either alone.
"""
from __future__ import annotations

import time

import torch

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.core.node import NodeType
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
        print(f"  Conv {ci+1}/10", flush=True)

    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s", flush=True)

    ps = loop.predictive_soma

    def combined_retrieve(question, fp_weight=0.5):
        """Combine fingerprint cosine sim + topology node overlap."""
        qvec = loop.encode_text(question, keep_grad=False)
        modality = ps.config.input_modalities[0]
        ps.soma.step({modality: qvec}, eval_mode=True)
        ps._diversify_activations(qvec)
        ps._apply_lateral_inhibition()

        # Fingerprint scores
        query_fp = ps._get_node_fingerprint()
        steps = list(ps._activation_store.keys())
        if not steps:
            return []
        stored_matrix = torch.stack([ps._activation_store[s] for s in steps])
        fp_sims = torch.nn.functional.cosine_similarity(
            query_fp.unsqueeze(0), stored_matrix, dim=1,
        ).cpu()

        # Topology scores
        active_nodes = set()
        for node in ps.soma.graph.all_nodes():
            if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                continue
            if node.last_activation is not None and node.last_activation.norm().item() > 1e-8:
                active_nodes.add(node.id)

        topo_scores = torch.zeros(len(steps))
        if active_nodes:
            n_active = len(active_nodes)
            for i, step in enumerate(steps):
                count = 0
                for nid in active_nodes:
                    if nid in ps._node_memory_index and step in ps._node_memory_index[nid]:
                        count += 1
                topo_scores[i] = count / n_active

        # Normalize both to [0, 1] range
        fp_min, fp_max = fp_sims.min(), fp_sims.max()
        if fp_max > fp_min:
            fp_norm = (fp_sims - fp_min) / (fp_max - fp_min)
        else:
            fp_norm = fp_sims

        topo_min, topo_max = topo_scores.min(), topo_scores.max()
        if topo_max > topo_min:
            topo_norm = (topo_scores - topo_min) / (topo_max - topo_min)
        else:
            topo_norm = topo_scores

        combined = fp_weight * fp_norm + (1 - fp_weight) * topo_norm

        k = min(5, len(steps))
        top_scores, top_indices = torch.topk(combined, k)

        results = []
        for i in range(k):
            step = steps[top_indices[i].item()]
            text = ps.text_store.get(step, "")
            if text:
                results.append((step, text, float(top_scores[i].item())))
        return results

    # Evaluate strategies
    print("\n--- Evaluation ---", flush=True)

    for name, weight in [
        ("fingerprint_only", 1.0),
        ("combined_0.7", 0.7),
        ("combined_0.5", 0.5),
        ("combined_0.3", 0.3),
        ("topology_only", 0.0),
    ]:
        t1 = time.perf_counter()
        scores = []
        for conv in conversations:
            for qa in conv.qa_pairs[:args.qa_per_conv]:
                results = combined_retrieve(qa.question, fp_weight=weight)
                best = max(
                    (token_f1(t, str(qa.answer)) for _, t, _ in results),
                    default=0.0,
                )
                scores.append(best)
        elapsed = time.perf_counter() - t1
        hits = sum(1 for s in scores if s > 0.05)
        avg_f1 = sum(scores) / len(scores) if scores else 0
        print(f"  {name:20s}: F1={avg_f1:.4f}  hits={hits}/100  ({elapsed:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
