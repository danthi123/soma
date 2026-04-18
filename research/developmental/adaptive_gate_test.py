"""Phase 6: Adaptive gating — scale rerank weight by fingerprint quality.

Problem: In Phase 3's confidence-gated retrieval, the gate sometimes
fires confidently but reranking HURTS (6 queries lose). Three have
conf>0.18 with delta -0.11 to -0.15.

Root cause: confidence measures the GAP between top-1 and top-2
fingerprint sims, not the ABSOLUTE quality. A high gap between noise
values (e.g., 0.15 vs 0.02) shouldn't trigger strong reranking.

Fix: Scale rerank_weight by the absolute best fingerprint similarity:
    effective_w = rerank_weight * clamp(fp_sorted[0], 0, 1)

This attenuates the graph's influence when fingerprints are low-quality
noise, while preserving full influence when the graph has strong matches.

Also tests a 2nd strategy: combined confidence = gap * absolute_sim,
which is stricter — requires BOTH high gap AND high absolute match.
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def evaluate_strategies(
    ps: PredictiveSOMA,
    st_model: SentenceTransformer,
    corpus: list[str],
    corpus_embeddings: torch.Tensor,
    step_to_cidx: dict[int, int],
    qa_pairs: list[tuple[str, str]],
    device: torch.device,
) -> None:
    """Run evaluation with multiple gating strategies and compare."""
    cidx_to_step = {v: k for k, v in step_to_cidx.items()}

    # Collect per-query data
    query_data: list[dict] = []

    for qi, (question, answer) in enumerate(qa_pairs):
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_embeddings, dim=1,
        )
        top_k_sims, top_k_indices = torch.topk(sims, 20)

        modality = ps.config.input_modalities[0]
        ps.soma.step({modality: q_emb}, eval_mode=True)
        ps._diversify_activations(q_emb)
        ps._apply_lateral_inhibition()
        query_fp = ps._get_node_fingerprint()

        fp_sims: list[float] = []
        for i in range(len(top_k_indices)):
            cidx = top_k_indices[i].item()
            step_num = cidx_to_step.get(cidx)
            fp_sim = 0.0
            if step_num is not None and step_num in ps._activation_store:
                fp_sim = torch.nn.functional.cosine_similarity(
                    query_fp.unsqueeze(0),
                    ps._activation_store[step_num].unsqueeze(0),
                ).item()
            fp_sims.append(fp_sim)

        fp_sorted = sorted(fp_sims, reverse=True)
        confidence = fp_sorted[0] - (fp_sorted[1] if len(fp_sorted) > 1 else 0)

        # VecDB baseline (pure embedding, top-5)
        vecdb_top5 = [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]
        vecdb_score = max(
            (token_f1(corpus[idx], answer) for idx in vecdb_top5),
            default=0.0,
        )

        query_data.append({
            "question": question[:60],
            "answer": answer[:60],
            "confidence": confidence,
            "fp_best": fp_sorted[0],
            "fp_second": fp_sorted[1] if len(fp_sorted) > 1 else 0,
            "vecdb_score": vecdb_score,
            "top_k_sims": [top_k_sims[i].item() for i in range(len(top_k_sims))],
            "fp_sims": fp_sims,
            "top_k_indices": [top_k_indices[i].item() for i in range(len(top_k_indices))],
        })

    # Now replay with different strategies
    strategies = {
        "vecdb_only": {"gate": 999.0, "weight": 0.0},
        "fixed_gate_0.05_w0.2": {"gate": 0.05, "weight": 0.2, "adaptive": False},
        "adaptive_gate_0.05_w0.2": {"gate": 0.05, "weight": 0.2, "adaptive": True},
        "combined_conf_0.05_w0.2": {"gate": 0.05, "weight": 0.2, "combined": True},
        "fixed_gate_0.10_w0.2": {"gate": 0.10, "weight": 0.2, "adaptive": False},
        "adaptive_gate_0.10_w0.2": {"gate": 0.10, "weight": 0.2, "adaptive": True},
        "fixed_gate_0.05_w0.3": {"gate": 0.05, "weight": 0.3, "adaptive": False},
        "adaptive_gate_0.05_w0.3": {"gate": 0.05, "weight": 0.3, "adaptive": True},
    }

    results: dict[str, dict] = {}
    for name, params in strategies.items():
        hits = 0
        graph_used = 0
        wins = 0
        losses = 0

        for qd in query_data:
            gate_threshold = params["gate"]
            rerank_weight = params["weight"]

            # Determine effective confidence
            if params.get("combined"):
                effective_conf = qd["confidence"] * qd["fp_best"]
            else:
                effective_conf = qd["confidence"]

            # Check gate
            if effective_conf >= gate_threshold:
                graph_used += 1
                w = rerank_weight
                if params.get("adaptive"):
                    w = rerank_weight * max(0.0, min(qd["fp_best"], 1.0))

                rerank = [
                    ((1 - w) * qd["top_k_sims"][i] + w * qd["fp_sims"][i],
                     qd["top_k_indices"][i])
                    for i in range(len(qd["top_k_indices"]))
                ]
                rerank.sort(key=lambda x: -x[0])
                top5 = [idx for _, idx in rerank[:5]]
            else:
                top5 = [qd["top_k_indices"][i] for i in range(min(5, len(qd["top_k_indices"])))]

            score = max(
                (token_f1(corpus[idx], qd["answer"]) for idx in top5),
                default=0.0,
            )
            if score > 0.05:
                hits += 1

            # Compare to vecdb
            if score > qd["vecdb_score"] + 0.01:
                wins += 1
            elif score < qd["vecdb_score"] - 0.01:
                losses += 1

        results[name] = {
            "hits": hits,
            "graph_used": graph_used,
            "wins": wins,
            "losses": losses,
        }

    # Print results
    print(f"\n{'='*70}")
    print("ADAPTIVE GATING COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'Hits':>5} {'Used':>5} {'Wins':>5} {'Loss':>5} {'Net':>5}")
    print("-" * 70)
    for name, r in results.items():
        net = r["wins"] - r["losses"]
        print(f"  {name:<33} {r['hits']:>5} {r['graph_used']:>5} {r['wins']:>5} {r['losses']:>5} {net:>+5}")

    # Detailed analysis: which queries does adaptive save?
    print(f"\n{'='*70}")
    print("ADAPTIVE vs FIXED DETAILED (gate=0.05, w=0.2)")
    print(f"{'='*70}")

    for qd in query_data:
        if qd["confidence"] < 0.05:
            continue

        # Fixed reranking
        w_fixed = 0.2
        fixed_rerank = [
            ((1 - w_fixed) * qd["top_k_sims"][i] + w_fixed * qd["fp_sims"][i],
             qd["top_k_indices"][i])
            for i in range(len(qd["top_k_indices"]))
        ]
        fixed_rerank.sort(key=lambda x: -x[0])
        fixed_top5 = [idx for _, idx in fixed_rerank[:5]]
        fixed_score = max(
            (token_f1(corpus[idx], qd["answer"]) for idx in fixed_top5),
            default=0.0,
        )

        # Adaptive reranking
        w_adapt = 0.2 * max(0.0, min(qd["fp_best"], 1.0))
        adapt_rerank = [
            ((1 - w_adapt) * qd["top_k_sims"][i] + w_adapt * qd["fp_sims"][i],
             qd["top_k_indices"][i])
            for i in range(len(qd["top_k_indices"]))
        ]
        adapt_rerank.sort(key=lambda x: -x[0])
        adapt_top5 = [idx for _, idx in adapt_rerank[:5]]
        adapt_score = max(
            (token_f1(corpus[idx], qd["answer"]) for idx in adapt_top5),
            default=0.0,
        )

        delta_fixed = fixed_score - qd["vecdb_score"]
        delta_adapt = adapt_score - qd["vecdb_score"]

        if abs(delta_fixed - delta_adapt) > 0.005:
            marker = ""
            if delta_adapt > delta_fixed + 0.01:
                marker = " << ADAPTIVE BETTER"
            elif delta_fixed > delta_adapt + 0.01:
                marker = " << FIXED BETTER"
            print(
                f"  conf={qd['confidence']:.3f} fp_best={qd['fp_best']:.3f} "
                f"w_eff={w_adapt:.3f} "
                f"fixed={delta_fixed:+.3f} adapt={delta_adapt:+.3f}{marker}"
            )
            print(f"    Q: {qd['question']}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    args = parser.parse_args()

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

    qa_pairs: list[tuple[str, str]] = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qa_pairs.append((qa.question, str(qa.answer)))

    print("Loading model...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    print("Encoding corpus...", flush=True)
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    config = SOMAConfig.developmental(
        sensor_output_dim=st_dim, text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2, associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2, integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2, integrator_output_dim=st_dim,
    )
    ps = PredictiveSOMA(config, device=device)
    step_to_cidx: dict[int, int] = {}
    cidx = 0

    print("Developing graph...", flush=True)
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
        print(f"  Conv {ci + 1}/10", flush=True)
    print(f"  Dev: {time.perf_counter() - t0:.1f}s")

    evaluate_strategies(
        ps, st_model, corpus, corpus_emb, step_to_cidx, qa_pairs, device,
    )


if __name__ == "__main__":
    main()
