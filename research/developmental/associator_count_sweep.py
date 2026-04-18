"""Phase 9: Initial associator count sweep.

Hypothesis: hybrid retrieval hits plateau near ~66/100 because fingerprint
vocabulary is bounded by C(N, min_active) where N is the graph's active
node pool. With ~14-20 nodes and min_active=3, only ~360-1140 distinct
fingerprint patterns are available for 5882 turns — ~5-16 turns per
pattern, not discriminative enough to lift more queries.

Scaling initial_associator_count should expand the fingerprint vocabulary:
  8 associators -> ~14 total nodes -> C(14,3) = 364 patterns (current baseline)
 32 associators -> ~38 total nodes -> C(38,3) = 8,436 patterns (~23x)
 64 associators -> ~70 total nodes -> C(70,3) = 54,834 patterns (~150x)

Keeps everything else fixed (learnable projections, min_active=3,
gate=0.05, weight=0.2, same encoder). `max_nodes` lifted to 128 so
neurogenesis has room on the 64-node run.
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def run_config(
    associator_count: int,
    conversations: list,
    corpus: list[str],
    qa_pairs: list[tuple[str, str]],
    st_model: SentenceTransformer,
    st_dim: int,
    device: torch.device,
) -> dict:
    """Run full development + hybrid retrieval with given associator count."""
    config = SOMAConfig.developmental(
        sensor_output_dim=st_dim, text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2, associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2, integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2, integrator_output_dim=st_dim,
        initial_associator_count=associator_count,
        max_nodes=128,
    )
    ps = PredictiveSOMA(config, device=device)
    step_to_cidx: dict[int, int] = {}
    cidx = 0

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

    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

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

    final_nodes = len(ps.soma.graph.nodes)
    return {
        "associator_count": associator_count,
        "final_nodes": final_nodes,
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

    sweep = [8, 32, 64]
    results = []
    for n in sweep:
        print(f"\n=== associator_count={n} ===", flush=True)
        r = run_config(
            n, conversations, corpus, qa_pairs, st_model, st_dim, device,
        )
        net = r["wins"] - r["losses"]
        print(f"  Final nodes: {r['final_nodes']}  Hits: {r['hits']}/100  "
              f"VecDB: {r['vecdb_hits']}  Used: {r['graph_used']}  "
              f"Wins: {r['wins']}  Loss: {r['losses']}  Net: {net:+d}  "
              f"({r['dev_time']:.1f}s)")
        results.append(r)

    print(f"\n{'='*70}")
    print("ASSOCIATOR COUNT SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'n_init':<8}{'n_final':<10}{'Hits':<8}{'VecDB':<8}"
          f"{'Used':<8}{'Wins':<8}{'Loss':<8}{'Net':<6}{'Time':<8}")
    for r in results:
        net = r["wins"] - r["losses"]
        print(f"  {r['associator_count']:<8}{r['final_nodes']:<10}"
              f"{r['hits']:<8}{r['vecdb_hits']:<8}{r['graph_used']:<8}"
              f"{r['wins']:<8}{r['losses']:<8}{net:<+6}"
              f"{r['dev_time']:.0f}s")


if __name__ == "__main__":
    main()
