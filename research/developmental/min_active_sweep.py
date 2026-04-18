"""Phase 7: min_active sweep in hybrid retrieval context.

Previously min_active=4 hurt pure graph retrieval (-6 hits). But in
the hybrid context, we only need fingerprints for RERANKING within
embedding-recalled candidates. More discriminative fingerprints (more
active nodes) might improve gate coverage and accuracy.

Tests min_active={2,3,4,5,6} with fixed gate=0.05, w=0.2.
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def run_hybrid_assessment(
    min_active: int,
    conversations: list,
    corpus: list[str],
    qa_pairs: list[tuple[str, str]],
    st_model: SentenceTransformer,
    st_dim: int,
    device: torch.device,
    gate_threshold: float = 0.05,
    rerank_weight: float = 0.2,
) -> dict:
    """Build graph with given min_active and measure hybrid retrieval."""
    config = SOMAConfig.developmental(
        sensor_output_dim=st_dim, text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2, associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2, integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2, integrator_output_dim=st_dim,
    )
    ps = PredictiveSOMA(config, device=device)
    step_to_cidx: dict[int, int] = {}
    cidx = 0

    # Patch lateral inhibition to use our min_active
    original_inhibit = ps._apply_lateral_inhibition

    def patched_inhibit(**kwargs: object) -> list[str]:
        kwargs["min_active"] = min_active
        return original_inhibit(**kwargs)

    ps._apply_lateral_inhibition = patched_inhibit

    # Develop
    for conv in conversations:
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

    # Encode corpus
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    # Measure
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

        # VecDB baseline
        vecdb_top5 = [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]
        vecdb_score = max(
            (token_f1(corpus[idx], answer) for idx in vecdb_top5),
            default=0.0,
        )
        if vecdb_score > 0.05:
            vecdb_hits += 1

        # Graph fingerprint
        modality = ps.config.input_modalities[0]
        ps.soma.step({modality: q_emb}, eval_mode=True)
        ps._diversify_activations(q_emb)
        ps._apply_lateral_inhibition(min_active=min_active)
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

        if confidence >= gate_threshold:
            graph_used += 1
            rerank = [
                ((1 - rerank_weight) * top_k_sims[i].item()
                 + rerank_weight * fp_sims[i],
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
        "min_active": min_active,
        "hits": hits,
        "vecdb_hits": vecdb_hits,
        "graph_used": graph_used,
        "wins": wins,
        "losses": losses,
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

    results = []
    for min_active in [2, 3, 4, 5, 6]:
        print(f"\n=== min_active={min_active} ===", flush=True)
        t0 = time.perf_counter()
        r = run_hybrid_assessment(
            min_active, conversations, corpus, qa_pairs,
            st_model, st_dim, device,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Hits: {r['hits']}/100  VecDB: {r['vecdb_hits']}  "
              f"Used: {r['graph_used']}  Wins: {r['wins']}  Loss: {r['losses']}  "
              f"Net: {r['wins'] - r['losses']:+d}  ({elapsed:.1f}s)")
        results.append(r)

    print(f"\n{'='*60}")
    print("MIN_ACTIVE SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'min_active':>10} {'Hits':>5} {'VecDB':>5} {'Used':>5} "
          f"{'Wins':>5} {'Loss':>5} {'Net':>5}")
    print("-" * 60)
    for r in results:
        net = r["wins"] - r["losses"]
        print(f"  {r['min_active']:>8} {r['hits']:>5} {r['vecdb_hits']:>5} "
              f"{r['graph_used']:>5} {r['wins']:>5} {r['losses']:>5} {net:>+5}")


if __name__ == "__main__":
    main()
