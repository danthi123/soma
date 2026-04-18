"""Phase 12: Per-query attribution.

Phase 10/11 established the graph contributes weak, slice-sensitive
signal (~0.8% absolute over random). This script investigates WHERE
the graph consistently helps or hurts, by classifying each query into
a simple category and reporting per-category win/loss rates.

Category rules (substring match, lowercased):
  cross_entity : contains " and " in a noun context ("John and Maria")
  temporal     : starts with when / what time / what date / how long
  counting     : contains how many / count / number of
  factual_self : contains my / your / our
  causal       : starts with why / what led / what caused / what helped
  preference   : contains like to / prefer / favorite / love / enjoy
  other        : no match

If one or more categories show >60% win rate with the graph (and
the category is big enough, n>=20), tighter gating by category could
lift net hits.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def classify(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["how many", "count", "number of"]):
        return "counting"
    if any(q.startswith(w) for w in ["when ", "what time", "what date", "how long"]):
        return "temporal"
    if any(q.startswith(w) for w in ["why ", "what led", "what caused", "what helped"]):
        return "causal"
    if any(w in q for w in [" like to", " prefer", " favorite", " love ", " enjoy"]):
        return "preference"
    if " and " in q and any(w in q for w in ["who ", "what ", "which "]):
        return "cross_entity"
    if any(w in q for w in [" my ", " your ", " our "]):
        return "factual_self"
    return "other"


def evaluate_with_dump(
    qa_pairs: list[tuple[str, str]],
    ps: PredictiveSOMA,
    corpus: list[str],
    corpus_emb: torch.Tensor,
    cidx_to_step: dict[int, int],
    st_model: SentenceTransformer,
    device: torch.device,
    slice_label: str,
) -> list[dict]:
    """Evaluate and return per-query records."""
    records = []
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

        graph_fired = confidence >= 0.05
        if graph_fired:
            rerank = [
                (0.8 * top_k_sims[i].item() + 0.2 * fp_sims[i],
                 top_k_indices[i].item())
                for i in range(len(top_k_indices))
            ]
            rerank.sort(key=lambda x: -x[0])
            top5 = [idx for _, idx in rerank[:5]]
        else:
            top5 = vecdb_top5

        hybrid_score = max(
            (token_f1(corpus[idx], answer) for idx in top5),
            default=0.0,
        )

        delta = hybrid_score - vecdb_score
        if delta > 0.01:
            outcome = "win"
        elif delta < -0.01:
            outcome = "loss"
        else:
            outcome = "tie"

        records.append({
            "slice": slice_label,
            "question": question,
            "answer": answer[:80],
            "category": classify(question),
            "vecdb_score": round(vecdb_score, 3),
            "hybrid_score": round(hybrid_score, 3),
            "delta": round(delta, 3),
            "outcome": outcome,
            "graph_fired": graph_fired,
            "confidence": round(confidence, 3),
            "vecdb_hit": vecdb_score > 0.05,
            "hybrid_hit": hybrid_score > 0.05,
        })
    return records


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

    all_records: list[dict] = []
    print("\nEvaluating with per-query dump...")
    for label, pairs in slice_qa.items():
        t0 = time.perf_counter()
        recs = evaluate_with_dump(
            pairs, ps, corpus, corpus_emb, cidx_to_step, st_model, device, label,
        )
        all_records.extend(recs)
        print(f"  {label}: {len(recs)} records ({time.perf_counter() - t0:.1f}s)")

    dump_path = "research/developmental/results/per_query_attribution.json"
    with open(dump_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nDumped {len(all_records)} records to {dump_path}")

    # Attribution analysis
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "win": 0, "loss": 0, "tie": 0,
                 "fired_win": 0, "fired_loss": 0, "fired_tie": 0,
                 "vdb_hits": 0, "hyb_hits": 0}
    )
    for r in all_records:
        c = by_cat[r["category"]]
        c["n"] += 1
        c[r["outcome"]] += 1
        if r["graph_fired"]:
            c[f"fired_{r['outcome']}"] += 1
        if r["vecdb_hit"]:
            c["vdb_hits"] += 1
        if r["hybrid_hit"]:
            c["hyb_hits"] += 1

    print(f"\n{'='*90}")
    print("PER-QUERY ATTRIBUTION BY CATEGORY")
    print(f"{'='*90}")
    print(f"  {'Category':<15}{'N':<5}{'Win':<6}{'Loss':<6}{'Tie':<6}"
          f"{'FiredW':<8}{'FiredL':<8}{'FiredT':<8}"
          f"{'VdbHit':<8}{'HybHit':<8}{'Delta':<7}")
    categories = sorted(by_cat.keys(), key=lambda k: -by_cat[k]["n"])
    for cat in categories:
        c = by_cat[cat]
        net = c["hyb_hits"] - c["vdb_hits"]
        print(
            f"  {cat:<15}{c['n']:<5}{c['win']:<6}{c['loss']:<6}{c['tie']:<6}"
            f"{c['fired_win']:<8}{c['fired_loss']:<8}{c['fired_tie']:<8}"
            f"{c['vdb_hits']:<8}{c['hyb_hits']:<8}{net:<+7d}"
        )

    print("\nWin-dominant categories (>=2x more wins than losses, n>=20):")
    for cat in categories:
        c = by_cat[cat]
        if c["n"] >= 20 and c["loss"] > 0 and c["win"] / max(c["loss"], 1) >= 2:
            print(f"  {cat}: {c['win']}W / {c['loss']}L / {c['tie']}T (n={c['n']})")

    print("\nLoss-dominant categories (>=2x more losses than wins, n>=20):")
    for cat in categories:
        c = by_cat[cat]
        if c["n"] >= 20 and c["win"] > 0 and c["loss"] / max(c["win"], 1) >= 2:
            print(f"  {cat}: {c['win']}W / {c['loss']}L / {c['tie']}T (n={c['n']})")

    # Graph-fired-only analysis
    fired_records = [r for r in all_records if r["graph_fired"]]
    print(f"\nGraph fired on {len(fired_records)}/{len(all_records)} queries "
          f"({len(fired_records)/len(all_records)*100:.1f}%)")
    if fired_records:
        fw = sum(1 for r in fired_records if r["outcome"] == "win")
        fl = sum(1 for r in fired_records if r["outcome"] == "loss")
        ft = sum(1 for r in fired_records if r["outcome"] == "tie")
        print(f"  Fired W/L/T: {fw}/{fl}/{ft} (net when firing: {fw - fl:+d})")


if __name__ == "__main__":
    main()
