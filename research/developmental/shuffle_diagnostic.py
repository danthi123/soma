"""Phase 11: Shuffle diagnostic.

Multi-slice validation showed the gated-hybrid delta vs VecDB is +2/500
queries (32 wins, 34 losses) — essentially coin-flip, with wide slice
variance (-2, +1, +3). This suggests the fingerprint mechanism isn't
adding retrievable signal.

This script is the definitive test. One dev run (n=8). Evaluate on all
three slices with:
  (a) REAL mapping: memory idx -> its own stored fingerprint
  (b) SHUFFLED mapping x 5 seeds: memory idx -> random other memory's fingerprint

If shuffle deltas are indistinguishable from real deltas, the graph
contributes no signal. If real consistently beats shuffle, the graph
is contributing something (weak but real).
"""
from __future__ import annotations

import random
import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def evaluate_slice(
    qa_pairs: list[tuple[str, str]],
    ps: PredictiveSOMA,
    corpus: list[str],
    corpus_emb: torch.Tensor,
    cidx_to_fp_step: dict[int, int],
    st_model: SentenceTransformer,
    device: torch.device,
) -> dict:
    """Measure gated-hybrid hits with a given cidx->fp_step mapping.

    `cidx_to_fp_step` says, "when ranking corpus index c, use the
    fingerprint stored at step cidx_to_fp_step[c]." For the real
    condition this matches cidx_to_step (identity over steps). For
    shuffled, memory indices get randomly reassigned fingerprints.
    """
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
        query_fp = ps._get_node_fingerprint()

        fp_sims = []
        for i in range(len(top_k_indices)):
            c = top_k_indices[i].item()
            fp_step = cidx_to_fp_step.get(c)
            fp_sim = 0.0
            if fp_step is not None and fp_step in ps._activation_store:
                fp_sim = torch.nn.functional.cosine_similarity(
                    query_fp.unsqueeze(0),
                    ps._activation_store[fp_step].unsqueeze(0),
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
        "n": len(qa_pairs),
        "hits": hits,
        "vecdb_hits": vecdb_hits,
        "graph_used": graph_used,
        "wins": wins,
        "losses": losses,
        "delta": hits - vecdb_hits,
    }


def shuffle_mapping(cidx_to_step: dict[int, int], seed: int) -> dict[int, int]:
    """Randomly permute fingerprint-step assignment over memory indices."""
    rng = random.Random(seed)
    cidxs = list(cidx_to_step.keys())
    steps = list(cidx_to_step.values())
    rng.shuffle(steps)
    return dict(zip(cidxs, steps, strict=True))


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
    print(f"  Development: {time.perf_counter() - t0:.1f}s, "
          f"final nodes: {len(ps.soma.graph.nodes)}")

    print("\nEncoding corpus...", flush=True)
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    real_mapping = {v: k for k, v in step_to_cidx.items()}
    shuffle_seeds = [1, 2, 3, 4, 5]
    shuffled_mappings = [shuffle_mapping(real_mapping, s) for s in shuffle_seeds]

    all_results: dict[str, dict] = {}
    print("\nEvaluating...")
    for label, pairs in slice_qa.items():
        real_r = evaluate_slice(
            pairs, ps, corpus, corpus_emb, real_mapping, st_model, device,
        )
        shuf_rs = [
            evaluate_slice(
                pairs, ps, corpus, corpus_emb, sm, st_model, device,
            )
            for sm in shuffled_mappings
        ]
        shuf_deltas = [r["delta"] for r in shuf_rs]
        shuf_mean = sum(shuf_deltas) / len(shuf_deltas)
        shuf_min = min(shuf_deltas)
        shuf_max = max(shuf_deltas)

        all_results[label] = {
            "real": real_r,
            "shuffled_deltas": shuf_deltas,
            "shuf_mean": shuf_mean,
            "shuf_min": shuf_min,
            "shuf_max": shuf_max,
        }
        print(
            f"  {label}: real delta {real_r['delta']:+d}  "
            f"shuffled deltas {shuf_deltas}  "
            f"mean {shuf_mean:+.1f} range [{shuf_min:+d}, {shuf_max:+d}]",
            flush=True,
        )

    print(f"\n{'='*78}")
    print("SHUFFLE DIAGNOSTIC SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Slice':<14}{'N':<5}{'VecDB':<8}{'Real':<8}{'RealD':<8}"
          f"{'ShufD mean':<12}{'ShufD range':<16}")
    for label, r in all_results.items():
        real = r["real"]
        sm = r["shuf_mean"]
        smin, smax = r["shuf_min"], r["shuf_max"]
        print(
            f"  {label:<14}{real['n']:<5}{real['vecdb_hits']:<8}"
            f"{real['hits']:<8}{real['delta']:<+8d}{sm:<+12.1f}"
            f"[{smin:+d}, {smax:+d}]"
        )

    total_n = sum(r["real"]["n"] for r in all_results.values())
    total_real = sum(r["real"]["delta"] for r in all_results.values())
    total_shuf_per_seed = [
        sum(all_results[lbl]["shuffled_deltas"][i] for lbl in all_results)
        for i in range(len(shuffle_seeds))
    ]
    total_shuf_mean = sum(total_shuf_per_seed) / len(total_shuf_per_seed)
    print(
        f"\n  COMBINED: n={total_n}  real delta {total_real:+d}  "
        f"shuffled {total_shuf_per_seed}  mean {total_shuf_mean:+.1f}  "
        f"range [{min(total_shuf_per_seed):+d}, {max(total_shuf_per_seed):+d}]"
    )


if __name__ == "__main__":
    main()
