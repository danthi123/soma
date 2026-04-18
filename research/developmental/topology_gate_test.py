"""Phase 13: Topology-based signal in the gated-hybrid framework.

The fingerprint signal is weak (~0.8% absolute over random, slice-
sensitive). Earlier work (retrieve_by_topology) showed that node
co-activation topology is a DIFFERENT aspect of the graph — it wins
20/100 LoCoMo queries where fingerprint fails. Previously we tried
linearly combining fingerprint + topology scores and the signals were
too correlated, giving ~44 hits. Here we swap the fingerprint signal
for topology directly in the gated-hybrid framework, testing on
three slices.

If topology as gate signal outperforms fingerprint as gate signal
on held-out, we have a better graph-derived retrieval signal. If it
doesn't, we've ruled out graph-derived retrieval signal as a
shippable angle.
"""
from __future__ import annotations

import time
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.developmental.prediction import PredictiveSOMA


def build_step_to_nodes(ps: PredictiveSOMA) -> dict[int, set[str]]:
    """Invert node_memory_index into {step -> set of active node ids}."""
    step_to_nodes: dict[int, set[str]] = defaultdict(set)
    for node_id, steps in ps._node_memory_index.items():
        for s in steps:
            step_to_nodes[s].add(node_id)
    return step_to_nodes


def get_query_active_nodes(ps: PredictiveSOMA) -> set[str]:
    """Nodes active after the current query step."""
    active: set[str] = set()
    for node in ps.soma.graph.all_nodes():
        if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
            continue
        if (
            node.last_activation is not None
            and node.last_activation.norm().item() > 1e-8
        ):
            active.add(node.id)
    return active


def evaluate_slice(
    qa_pairs: list[tuple[str, str]],
    ps: PredictiveSOMA,
    corpus: list[str],
    corpus_emb: torch.Tensor,
    cidx_to_step: dict[int, int],
    step_to_nodes: dict[int, set[str]],
    st_model: SentenceTransformer,
    device: torch.device,
    signal: str = "topology",
) -> dict:
    """Evaluate with given signal type for the gate."""
    hits = 0
    vecdb_hits = 0
    graph_used = 0
    wins = 0
    losses = 0

    for question, answer in qa_pairs:
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_emb, dim=1,
        )
        top_k_sims, top_k_indices = torch.topk(sims, 20)

        vecdb_top5 = [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]
        vecdb_score = max(
            (token_f1(corpus[idx], answer) for idx in vecdb_top5),
            default=0.0,
        )
        if vecdb_score > 0.05:
            vecdb_hits += 1

        modality = ps.config.input_modalities[0]
        ps.soma.step({modality: q_emb}, eval_mode=True)
        ps._diversify_activations(q_emb)
        ps._apply_lateral_inhibition()

        if signal == "topology":
            query_active = get_query_active_nodes(ps)
            n_q = len(query_active)
            graph_sims = []
            for i in range(len(top_k_indices)):
                c = top_k_indices[i].item()
                step_num = cidx_to_step.get(c)
                sim = 0.0
                if step_num is not None and n_q > 0:
                    mem_active = step_to_nodes.get(step_num, set())
                    if mem_active:
                        overlap = len(query_active & mem_active)
                        n_m = len(mem_active)
                        sim = overlap / max(n_q, n_m)
                graph_sims.append(sim)
        else:  # fingerprint (reference)
            query_fp = ps._get_node_fingerprint()
            graph_sims = []
            for i in range(len(top_k_indices)):
                c = top_k_indices[i].item()
                step_num = cidx_to_step.get(c)
                sim = 0.0
                if step_num is not None and step_num in ps._activation_store:
                    sim = torch.nn.functional.cosine_similarity(
                        query_fp.unsqueeze(0),
                        ps._activation_store[step_num].unsqueeze(0),
                    ).item()
                graph_sims.append(sim)

        g_sorted = sorted(graph_sims, reverse=True)
        confidence = g_sorted[0] - (g_sorted[1] if len(g_sorted) > 1 else 0)

        if confidence >= 0.05:
            graph_used += 1
            rerank = [
                (0.8 * top_k_sims[i].item() + 0.2 * graph_sims[i],
                 top_k_indices[i].item())
                for i in range(len(top_k_indices))
            ]
            rerank.sort(key=lambda x: -x[0])
            top5 = [idx for _, idx in rerank[:5]]
        else:
            top5 = vecdb_top5

        score = max(
            (token_f1(corpus[idx], answer) for idx in top5),
            default=0.0,
        )
        if score > 0.05:
            hits += 1
        if score > vecdb_score + 0.01:
            wins += 1
        elif score < vecdb_score - 0.01:
            losses += 1

    return {
        "signal": signal,
        "n": len(qa_pairs),
        "hits": hits,
        "vecdb_hits": vecdb_hits,
        "graph_used": graph_used,
        "wins": wins,
        "losses": losses,
        "delta": hits - vecdb_hits,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conversations = load_dataset()
    corpus: list[str] = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(
                    f"[{session.date_time}] {turn.speaker}: {turn.text}"
                )

    slices = {
        "A_[0:10]":  (0, 10),
        "B_[10:30]": (10, 30),
        "C_[30:50]": (30, 50),
    }
    slice_qa: dict[str, list[tuple[str, str]]] = {}
    for label, (lo, hi) in slices.items():
        pairs = []
        for conv in conversations:
            for qa in conv.qa_pairs[lo:hi]:
                pairs.append((qa.question, str(qa.answer)))
        slice_qa[label] = pairs

    print("Loading model...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    config = SOMAConfig.developmental(
        sensor_output_dim=st_dim, text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2, associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2, integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2, integrator_output_dim=st_dim,
    )
    ps = PredictiveSOMA(config, device=device)
    step_to_cidx: dict[int, int] = {}
    cidx = 0

    print("\nDeveloping (n_init=8)...", flush=True)
    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                with torch.no_grad():
                    emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                result = ps.process_input(emb, source_text=text)
                step_to_cidx[result["global_step"]] = cidx
                cidx += 1
        print(f"    Conv {ci + 1}/10", flush=True)
    print(f"  Development: {time.perf_counter() - t0:.1f}s")

    print("\nEncoding corpus...", flush=True)
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    cidx_to_step = {v: k for k, v in step_to_cidx.items()}
    step_to_nodes = build_step_to_nodes(ps)
    print(f"  step_to_nodes: {len(step_to_nodes)} steps indexed")

    print("\nEvaluating (fingerprint vs topology)...")
    print(f"  {'Slice':<14}{'Signal':<14}{'Hits':<6}{'VecDB':<8}"
          f"{'Delta':<8}{'Used':<6}{'W':<4}{'L':<4}")
    for label, pairs in slice_qa.items():
        for signal in ["fingerprint", "topology"]:
            r = evaluate_slice(
                pairs, ps, corpus, corpus_emb, cidx_to_step,
                step_to_nodes, st_model, device, signal=signal,
            )
            print(
                f"  {label:<14}{signal:<14}{r['hits']:<6}{r['vecdb_hits']:<8}"
                f"{r['delta']:<+8d}{r['graph_used']:<6}{r['wins']:<4}{r['losses']:<4}"
            )


if __name__ == "__main__":
    main()
