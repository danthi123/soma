"""Phase 10: Multi-slice held-out validation.

Phase 3 established gated hybrid beats VecDB by +3 hits (66 vs 63) on
LoCoMo's first 10 QA pairs per conversation (100 queries). Every
subsequent phase tuned against the same 100 queries. This script tests
whether the +3 win is real or an artifact of tuning to that specific
slice.

Design: one development run (n=8 Pareto-optimal config), measure both
VecDB-only and gated-hybrid hits on three disjoint QA slices:

  slice_A: [0:10]   = 100 queries (historical tuning set)
  slice_B: [10:30]  = 200 queries (unseen #1)
  slice_C: [30:50]  = 200 queries (unseen #2)

All 10 LoCoMo conversations have >=105 QA pairs, so slices [30:50] fit
all convs. If gated-hybrid - VecDB holds >= +3 on slice_B and slice_C,
the finding generalizes. If it collapses, we were overfitting to
slice_A.
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def evaluate_slice(
    label: str,
    qa_pairs: list[tuple[str, str]],
    ps: PredictiveSOMA,
    corpus: list[str],
    corpus_emb: torch.Tensor,
    cidx_to_step: dict[int, int],
    st_model: SentenceTransformer,
    device: torch.device,
) -> dict:
    """Measure VecDB and gated-hybrid hits on a QA slice."""
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
        "slice": label,
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
        "A_[0:10]_tuned":  [(0, 10)],
        "B_[10:30]_heldout1": [(10, 30)],
        "C_[30:50]_heldout2": [(30, 50)],
    }
    slice_qa = {}
    for label, ranges in slices.items():
        pairs = []
        for conv in conversations:
            for lo, hi in ranges:
                for qa in conv.qa_pairs[lo:hi]:
                    pairs.append((qa.question, str(qa.answer)))
        slice_qa[label] = pairs
        print(f"  {label}: {len(pairs)} queries")

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
    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s, final nodes: {len(ps.soma.graph.nodes)}")

    print("\nEncoding corpus...", flush=True)
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    cidx_to_step = {v: k for k, v in step_to_cidx.items()}

    print("\nEvaluating slices...")
    results = []
    for label, pairs in slice_qa.items():
        t0 = time.perf_counter()
        r = evaluate_slice(
            label, pairs, ps, corpus, corpus_emb, cidx_to_step,
            st_model, device,
        )
        r["eval_time"] = time.perf_counter() - t0
        results.append(r)
        print(f"  {label}: Hits {r['hits']}/{r['n']} VecDB {r['vecdb_hits']} "
              f"Delta {r['delta']:+d} Used {r['graph_used']} "
              f"W/L {r['wins']}/{r['losses']} ({r['eval_time']:.1f}s)")

    print(f"\n{'='*78}")
    print("MULTI-SLICE VALIDATION SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Slice':<22}{'N':<6}{'Hits':<8}{'VecDB':<8}"
          f"{'Delta':<8}{'Rate':<8}{'VdRate':<8}{'W':<4}{'L':<4}")
    for r in results:
        rate = r['hits'] / r['n'] * 100
        vd_rate = r['vecdb_hits'] / r['n'] * 100
        print(f"  {r['slice']:<22}{r['n']:<6}{r['hits']:<8}"
              f"{r['vecdb_hits']:<8}{r['delta']:<+8d}{rate:<7.1f}%"
              f"{vd_rate:<7.1f}%{r['wins']:<4}{r['losses']:<4}")

    total_hits = sum(r['hits'] for r in results)
    total_vd = sum(r['vecdb_hits'] for r in results)
    total_n = sum(r['n'] for r in results)
    print(f"\n  Combined: {total_hits}/{total_n} ({total_hits/total_n*100:.1f}%) "
          f"vs VecDB {total_vd}/{total_n} ({total_vd/total_n*100:.1f}%) "
          f"delta {total_hits - total_vd:+d}")


if __name__ == "__main__":
    main()
