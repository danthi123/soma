"""Phase 14: LongMemEval gated-hybrid validation.

Every prior phase tested on LoCoMo. The 04-17 Phase 1.3 audit found
that graph reranking (alpha=1.0) beat pure-cosine (alpha=0.0) on
LongMemEval by +15% F1 (0.163 vs 0.142) — a different framework than
our gated hybrid. This script tests the current gated hybrid
(confidence-gated fingerprint reranking, w=0.2, gate=0.05) on
LongMemEval to check whether the plateau is LoCoMo-specific or
cross-benchmark.

Uses the oracle variant (500 temporal-reasoning items). Merges all
haystack sessions into one corpus (same methodology as LoCoMo tests).
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.longmemeval.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use a cap for first pass. Scale up if results are interesting.
    LIMIT = 100
    items = load_dataset("oracle", limit=LIMIT)
    print(f"Loaded {len(items)} LongMemEval items (limit={LIMIT})")

    # Build a merged corpus from all haystack sessions across items.
    # Each corpus entry is tagged with its originating item index for
    # later analysis if needed.
    corpus: list[str] = []
    for item in items:
        for sess_idx, session in enumerate(item.haystack_sessions):
            for turn in session:
                corpus.append(f"{turn.role}: {turn.content}")
    print(f"Corpus size: {len(corpus)} turns")

    qa_pairs: list[tuple[str, str, str]] = [
        (item.question, str(item.answer), item.question_type) for item in items
    ]
    print(f"Query set: {len(qa_pairs)} questions")
    print(f"Question types: {set(qt for _, _, qt in qa_pairs)}")

    print("\nLoading encoder...", flush=True)
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

    print(f"\nDeveloping on {len(corpus)} turns...", flush=True)
    t0 = time.perf_counter()
    report_every = max(1, len(corpus) // 20)
    for turn_idx, text in enumerate(corpus):
        with torch.no_grad():
            emb = st_model.encode(
                text, convert_to_tensor=True, device=str(device),
            )
        result = ps.process_input(emb, source_text=text)
        step_to_cidx[result["global_step"]] = cidx
        cidx += 1
        if (turn_idx + 1) % report_every == 0:
            print(f"    {turn_idx + 1}/{len(corpus)} turns", flush=True)
    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s, final nodes: {len(ps.soma.graph.nodes)}")

    print("\nEncoding corpus for retrieval...", flush=True)
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    cidx_to_step = {v: k for k, v in step_to_cidx.items()}

    print("\nEvaluating gated hybrid vs VecDB...")
    hits = 0
    vecdb_hits = 0
    graph_used = 0
    wins = 0
    losses = 0
    by_type: dict[str, dict[str, int]] = {}

    t0 = time.perf_counter()
    for i, (question, answer, qtype) in enumerate(qa_pairs):
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
        for j in range(len(top_k_indices)):
            c = top_k_indices[j].item()
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
                (0.8 * top_k_sims[j].item() + 0.2 * fp_sims[j],
                 top_k_indices[j].item())
                for j in range(len(top_k_indices))
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

        bt = by_type.setdefault(qtype, {
            "n": 0, "hits": 0, "vdb_hits": 0, "wins": 0, "losses": 0,
        })
        bt["n"] += 1
        if score > 0.05:
            bt["hits"] += 1
        if vecdb_score > 0.05:
            bt["vdb_hits"] += 1
        if score > vecdb_score + 0.01:
            bt["wins"] += 1
        elif score < vecdb_score - 0.01:
            bt["losses"] += 1

    eval_time = time.perf_counter() - t0
    print(f"  Evaluation: {eval_time:.1f}s")

    print(f"\n{'='*78}")
    print("LONGMEMEVAL GATED-HYBRID SUMMARY")
    print(f"{'='*78}")
    print(f"  N:        {len(qa_pairs)}")
    print(f"  VecDB:    {vecdb_hits} hits ({vecdb_hits / len(qa_pairs) * 100:.1f}%)")
    print(f"  Hybrid:   {hits} hits ({hits / len(qa_pairs) * 100:.1f}%)")
    print(f"  Delta:    {hits - vecdb_hits:+d}")
    print(f"  Used:     {graph_used}/{len(qa_pairs)} "
          f"({graph_used / len(qa_pairs) * 100:.1f}%)")
    print(f"  Wins:     {wins}")
    print(f"  Losses:   {losses}")
    print(f"  Net:      {wins - losses:+d}")

    print(f"\n{'='*78}")
    print("BY QUESTION TYPE")
    print(f"{'='*78}")
    print(f"  {'Type':<30}{'N':<6}{'Hits':<8}{'VecDB':<8}{'Delta':<8}{'W/L'}")
    for qtype, bt in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        delta = bt["hits"] - bt["vdb_hits"]
        print(
            f"  {qtype:<30}{bt['n']:<6}{bt['hits']:<8}{bt['vdb_hits']:<8}"
            f"{delta:<+8d}{bt['wins']}/{bt['losses']}"
        )


if __name__ == "__main__":
    main()
