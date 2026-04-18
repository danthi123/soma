"""Phase 8: Learnable diversification projections.

The frozen random projections in _diversify_activations() create a
static input->winner mapping. Competitive learning on node weights
can't change this mapping. This experiment tests whether making the
projections learn (Hebbian for winners, anti-Hebbian for suppressed)
improves hybrid retrieval.

The current code already has learnable projections enabled. This script
compares against a frozen-projection baseline by zeroing out the
projection learning rate.
"""
from __future__ import annotations

import copy
import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def run_config(
    label: str,
    proj_lr_scale: float,
    conversations: list,
    corpus: list[str],
    qa_pairs: list[tuple[str, str]],
    st_model: SentenceTransformer,
    st_dim: int,
    device: torch.device,
) -> dict:
    """Run full development + hybrid retrieval with given proj_lr_scale."""
    config = SOMAConfig.developmental(
        sensor_output_dim=st_dim, text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2, associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2, integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2, integrator_output_dim=st_dim,
    )
    ps = PredictiveSOMA(config, device=device)
    step_to_cidx: dict[int, int] = {}
    cidx = 0

    # Patch competitive learning to control projection lr
    original_cl = ps._competitive_learning

    def patched_cl(
        input_tensor: torch.Tensor,
        suppressed_ids: list[str] | None = None,
        lr: float = 0.001,
        anti_lr: float = 0.0003,
    ) -> None:
        # Save current projections if we want to freeze them
        if proj_lr_scale == 0.0:
            saved = {k: v.clone() for k, v in ps._input_projections.items()}
        original_cl(input_tensor, suppressed_ids, lr, anti_lr)
        if proj_lr_scale == 0.0:
            # Restore projections (undo any updates)
            for k, v in saved.items():
                ps._input_projections[k] = v

    ps._competitive_learning = patched_cl

    # Develop
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
    dev_time = time.perf_counter() - t0

    # Encode corpus
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    # Measure hybrid retrieval
    cidx_to_step = {v: k for k, v in step_to_cidx.items()}
    hits = 0
    graph_used = 0
    wins = 0
    losses = 0
    vecdb_hits = 0

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
        query_fp = ps._get_node_fingerprint()

        fp_sims = []
        for i in range(len(top_k_indices)):
            c = top_k_indices[i].item()
            step_num = cidx_to_step.get(c)
            fp_sim = 0.0
            if step_num is not None and step_num in ps._activation_store:
                fp_sim = torch.nn.functional.cosine_similarity(
                    query_fp.unsqueeze(0),
                    ps._activation_store[step_num].unsqueeze(0),
                ).item()
            fp_sims.append(fp_sim)

        fp_sorted = sorted(fp_sims, reverse=True)
        confidence = fp_sorted[0] - (fp_sorted[1] if len(fp_sorted) > 1 else 0)

        if confidence >= 0.05:
            graph_used += 1
            rerank = [
                (0.8 * top_k_sims[i].item() + 0.2 * fp_sims[i],
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
        "label": label,
        "hits": hits,
        "vecdb_hits": vecdb_hits,
        "graph_used": graph_used,
        "wins": wins,
        "losses": losses,
        "dev_time": dev_time,
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

    qa_pairs: list[tuple[str, str]] = []
    for conv in conversations:
        for qa in conv.qa_pairs[:10]:
            qa_pairs.append((qa.question, str(qa.answer)))

    print("Loading model...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    configs = [
        ("frozen_proj", 0.0),
        ("learnable_proj", 1.0),
    ]

    results = []
    for label, proj_lr_scale in configs:
        print(f"\n=== {label} ===", flush=True)
        r = run_config(
            label, proj_lr_scale, conversations, corpus, qa_pairs,
            st_model, st_dim, device,
        )
        net = r["wins"] - r["losses"]
        print(f"  Hits: {r['hits']}/100  VecDB: {r['vecdb_hits']}  "
              f"Used: {r['graph_used']}  Wins: {r['wins']}  Loss: {r['losses']}  "
              f"Net: {net:+d}  ({r['dev_time']:.1f}s)")
        results.append(r)

    print(f"\n{'='*60}")
    print("LEARNABLE PROJECTION SUMMARY")
    print(f"{'='*60}")
    for r in results:
        net = r["wins"] - r["losses"]
        print(f"  {r['label']:<20} Hits: {r['hits']:>3}  Used: {r['graph_used']:>3}  "
              f"Net: {net:>+3}  ({r['dev_time']:.0f}s)")


if __name__ == "__main__":
    main()
