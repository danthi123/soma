"""LoCoMo retrieval with LLM-distilled projections — Direction 4a Phase 3.

Tests the primary success criterion:
  ``soma-distilled > chroma-mxbai`` on LoCoMo R@5 by > 0.02.

Architecture (different from run_locomo_locality):
This runner uses PredictiveSOMA **directly** (not via MemoryLayer +
attach_soma) because the distillation loss lives inside
``PredictiveSOMA.process_input``. MemoryLayer's ``attach_soma`` attaches
a plain SOMA that doesn't exercise the distillation path.

LoCoMo protocol: each conversation is evaluated independently. For
each sample, we build a fresh memory, process that sample's turns,
then answer that sample's queries. Evidence is expected to be within
the same conversation (cross-sample matches are false positives).

Systems (4):
- ``chroma-mxbai``       Chroma with mxbai-embed-large embeddings.
- ``soma-random``        PredictiveSOMA, frozen random projections,
                         retrieve_hybrid(alpha=0.3).
- ``soma-distilled``     PredictiveSOMA, learnable projections +
                         ``llm_embedding`` distillation (Direction 4a),
                         retrieve_hybrid(alpha=0.3).
- ``soma-spatial``       PredictiveSOMA, learnable projections +
                         ``llm_spatial`` distillation + learnable
                         positions + position-coupling loss
                         (Direction 4b), retrieve_hybrid(alpha=0.3).

Usage:
    python -u -m benchmarks.run_locomo_distill \\
        --max-samples 2 \\
        --out benchmarks/reports/locomo_distill_subset.md
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from benchmarks.datasets.locomo import (
    CATEGORY_NAMES,
    LoCoMoQuery,
    LoCoMoTurn,
    load_locomo,
    turns_for_sample,
)
from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.llm.embedders import CachedEmbedder, OllamaEmbedder


@dataclass
class DistillResult:
    system: str
    n_turns: int
    n_queries: int
    recall_at_k: dict[int, float]  # k -> R@k
    recall_by_category: dict[str, dict[int, float]]
    store_total_s: float
    retrieve_avg_ms: float


K_VALUES = [1, 5, 10]


def embed_corpus_with_mxbai(
    teacher: CachedEmbedder,
    texts: list[str],
    target_dim: int,
) -> torch.Tensor:
    """Embed, truncate to target_dim, L2-normalize rows."""
    out = []
    for t in texts:
        e = teacher.embed(t)
        if e.shape[0] >= target_dim:
            e = e[:target_dim]
        else:
            pad = torch.zeros(target_dim)
            pad[: e.shape[0]] = e
            e = pad
        e = e / (e.norm() + 1e-8)
        out.append(e)
    return torch.stack(out)


def _embed_query(teacher: CachedEmbedder, text: str, target_dim: int) -> torch.Tensor:
    e = teacher.embed(text)
    if e.shape[0] >= target_dim:
        e = e[:target_dim]
    else:
        pad = torch.zeros(target_dim)
        pad[: e.shape[0]] = e
        e = pad
    return e / (e.norm() + 1e-8)


def _aggregate_recall(
    per_sample: list[tuple[dict[int, int], dict[int, int], dict[str, dict[int, int]], dict[str, int]]],
) -> tuple[dict[int, float], dict[str, dict[int, float]]]:
    """Merge per-sample (hit_counts, total_counts, cat_hits, cat_totals)
    into dataset-wide R@k and R@k by category.
    """
    total_hits: dict[int, int] = {k: 0 for k in K_VALUES}
    total_queries: dict[int, int] = {k: 0 for k in K_VALUES}
    cat_hits: dict[str, dict[int, int]] = defaultdict(lambda: {k: 0 for k in K_VALUES})
    cat_totals: dict[str, int] = defaultdict(int)

    for hit_counts, total_counts, ch, ct in per_sample:
        for k in K_VALUES:
            total_hits[k] += hit_counts[k]
            total_queries[k] += total_counts[k]
        for name, kmap in ch.items():
            for k in K_VALUES:
                cat_hits[name][k] += kmap[k]
        for name, n in ct.items():
            cat_totals[name] += n

    recall = {k: total_hits[k] / max(1, total_queries[k]) for k in K_VALUES}
    recall_by_cat: dict[str, dict[int, float]] = {
        name: {k: cat_hits[name][k] / max(1, cat_totals[name]) for k in K_VALUES}
        for name in cat_totals
    }
    return recall, recall_by_cat


def _score_sample_queries(
    queries: list[LoCoMoQuery],
    hits_list: list[list[str]],   # retrieved dia_ids per query, in rank order
) -> tuple[dict[int, int], dict[int, int], dict[str, dict[int, int]], dict[str, int]]:
    """Compute per-query R@k hits within a single sample."""
    hit_counts: dict[int, int] = {k: 0 for k in K_VALUES}
    total_counts: dict[int, int] = {k: 0 for k in K_VALUES}
    cat_hits: dict[str, dict[int, int]] = {
        name: {k: 0 for k in K_VALUES} for name in CATEGORY_NAMES.values()
    }
    cat_totals: dict[str, int] = {name: 0 for name in CATEGORY_NAMES.values()}

    for q, retrieved in zip(queries, hits_list, strict=False):
        cat_name = CATEGORY_NAMES.get(q.category, "unknown")
        if cat_name in cat_totals:
            cat_totals[cat_name] += 1
        for k in K_VALUES:
            total_counts[k] += 1
            if any(e in retrieved[:k] for e in q.evidence):
                hit_counts[k] += 1
                if cat_name in cat_hits:
                    cat_hits[cat_name][k] += 1
    return hit_counts, total_counts, cat_hits, cat_totals


def _run_chroma_on_sample(
    sample_id: str,
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
    teacher: CachedEmbedder,
    target_dim: int,
) -> tuple[tuple, float, float]:
    import chromadb

    client = chromadb.EphemeralClient()
    # Unique collection name per sample; safe IDs = dia_ids (unique within sample)
    col = client.get_or_create_collection(
        name=f"locomo_{sample_id}", metadata={"hnsw:space": "cosine"},
    )
    texts = [t.text for t in turns]
    ids = [t.dia_id for t in turns]
    t0 = time.perf_counter()
    embeddings = embed_corpus_with_mxbai(teacher, texts, target_dim)
    col.add(ids=ids, documents=texts, embeddings=embeddings.tolist())
    store_s = time.perf_counter() - t0

    retrieved_per_q: list[list[str]] = []
    rt_start = time.perf_counter()
    for q in queries:
        q_emb = _embed_query(teacher, q.question, target_dim)
        result = col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=max(K_VALUES),
        )
        retrieved_per_q.append(result.get("ids", [[]])[0])
    retrieve_s = time.perf_counter() - rt_start

    score_tuple = _score_sample_queries(queries, retrieved_per_q)
    return score_tuple, store_s, retrieve_s


def _run_soma_on_sample(
    sample_id: str,
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
    teacher: CachedEmbedder,
    target_dim: int,
    *,
    projection_mode: str,
    distillation_target: str,
    distillation_weight: float,
    synap_locality: float,
    device: torch.device,
    seed: int,
    rerank_weight: float,
    gate_threshold: float,
    position_mode: str = "frozen_random",
    position_coupling_weight: float = 1.0,
    projection_distillation_winners: int = 3,
) -> tuple[tuple, float, float]:
    config = SOMAConfig.developmental(
        sensor_output_dim=target_dim,
        text_embed_dim=target_dim,
        associator_input_dim=target_dim,
        associator_hidden_dim=target_dim * 2,
        associator_output_dim=target_dim,
        integrator_input_dim=target_dim,
        integrator_hidden_dim=target_dim * 2,
        integrator_output_dim=target_dim,
        projection_mode=projection_mode,
        projection_distillation_target=distillation_target,
        projection_distillation_weight=distillation_weight,
        projection_distillation_winners=projection_distillation_winners,
        position_mode=position_mode,
        position_coupling_weight=position_coupling_weight,
        synaptogenesis_max_distance=synap_locality,
        neurogenesis_interval=0,
        seed=seed,
    )
    pred = PredictiveSOMA(config=config, device=device)
    if distillation_target in ("llm_embedding", "llm_spatial"):
        pred.attach_teacher(teacher)

    # Storage: embed turns and feed through SOMA
    texts = [t.text for t in turns]
    corpus_embeddings = embed_corpus_with_mxbai(teacher, texts, target_dim).to(device)
    dia_ids = [t.dia_id for t in turns]
    step_to_dia_id: dict[int, str] = {}
    step_to_corpus_idx: dict[int, int] = {}

    t0 = time.perf_counter()
    for i, t in enumerate(turns):
        result = pred.process_input(
            corpus_embeddings[i], source_text=t.text,
        )
        step = result.get("global_step", i)
        step_to_dia_id[step] = t.dia_id
        step_to_corpus_idx[step] = i
    store_s = time.perf_counter() - t0

    retrieved_per_q: list[list[str]] = []
    rt_start = time.perf_counter()
    for q in queries:
        q_emb = _embed_query(teacher, q.question, target_dim).to(device)
        hits = pred.retrieve_hybrid(
            q_emb,
            corpus_embeddings=corpus_embeddings,
            corpus_step_map=step_to_corpus_idx,
            recall_k=20,
            top_k=max(K_VALUES),
            gate_threshold=gate_threshold,
            rerank_weight=rerank_weight,
        )
        # hits are (step, text, score); map step → dia_id
        retrieved_dia_ids = [
            step_to_dia_id.get(h[0], "")
            for h in hits
        ]
        retrieved_per_q.append(retrieved_dia_ids)
    retrieve_s = time.perf_counter() - rt_start

    score_tuple = _score_sample_queries(queries, retrieved_per_q)
    return score_tuple, store_s, retrieve_s


def run_chroma_mxbai(
    samples: list[str],
    turns_by_sample: dict[str, list[LoCoMoTurn]],
    queries_by_sample: dict[str, list[LoCoMoQuery]],
    teacher: CachedEmbedder,
    target_dim: int,
) -> DistillResult:
    per_sample = []
    total_store = 0.0
    total_retrieve = 0.0
    n_turns = 0
    n_queries = 0
    for sid in samples:
        sample_turns = turns_by_sample[sid]
        sample_queries = queries_by_sample.get(sid, [])
        if not sample_queries:
            continue
        print(f"  chroma-mxbai sample={sid} turns={len(sample_turns)} queries={len(sample_queries)}")
        score_tuple, ss, rs = _run_chroma_on_sample(
            sid, sample_turns, sample_queries, teacher, target_dim,
        )
        per_sample.append(score_tuple)
        total_store += ss
        total_retrieve += rs
        n_turns += len(sample_turns)
        n_queries += len(sample_queries)

    recall, recall_by_cat = _aggregate_recall(per_sample)
    return DistillResult(
        system="chroma-mxbai",
        n_turns=n_turns,
        n_queries=n_queries,
        recall_at_k=recall,
        recall_by_category=recall_by_cat,
        store_total_s=total_store,
        retrieve_avg_ms=total_retrieve * 1000 / max(1, n_queries),
    )


def run_soma_predictive(
    system_name: str,
    samples: list[str],
    turns_by_sample: dict[str, list[LoCoMoTurn]],
    queries_by_sample: dict[str, list[LoCoMoQuery]],
    teacher: CachedEmbedder,
    target_dim: int,
    *,
    projection_mode: str,
    distillation_target: str,
    distillation_weight: float,
    synap_locality: float,
    device: torch.device,
    seed: int = 0,
    rerank_weight: float = 0.0,
    gate_threshold: float = 0.05,
    position_mode: str = "frozen_random",
    position_coupling_weight: float = 1.0,
    projection_distillation_winners: int = 3,
) -> DistillResult:
    per_sample = []
    total_store = 0.0
    total_retrieve = 0.0
    n_turns = 0
    n_queries = 0
    for sid in samples:
        sample_turns = turns_by_sample[sid]
        sample_queries = queries_by_sample.get(sid, [])
        if not sample_queries:
            continue
        print(f"  {system_name} sample={sid} turns={len(sample_turns)} queries={len(sample_queries)}")
        score_tuple, ss, rs = _run_soma_on_sample(
            sid, sample_turns, sample_queries, teacher, target_dim,
            projection_mode=projection_mode,
            distillation_target=distillation_target,
            distillation_weight=distillation_weight,
            synap_locality=synap_locality,
            device=device,
            seed=seed,
            rerank_weight=rerank_weight,
            gate_threshold=gate_threshold,
            position_mode=position_mode,
            position_coupling_weight=position_coupling_weight,
            projection_distillation_winners=projection_distillation_winners,
        )
        per_sample.append(score_tuple)
        total_store += ss
        total_retrieve += rs
        n_turns += len(sample_turns)
        n_queries += len(sample_queries)

    recall, recall_by_cat = _aggregate_recall(per_sample)
    return DistillResult(
        system=system_name,
        n_turns=n_turns,
        n_queries=n_queries,
        recall_at_k=recall,
        recall_by_category=recall_by_cat,
        store_total_s=total_store,
        retrieve_avg_ms=total_retrieve * 1000 / max(1, n_queries),
    )


def format_markdown(results: list[DistillResult]) -> str:
    lines = [
        "# LoCoMo distillation ablation",
        "",
        "Direction 4a Phase 3: tests whether LLM-distilled projections",
        "give SOMA's graph rerank retrieval advantage over pure embedding",
        "retrieval using the same embedding model.",
        "",
        "## Overall Recall@k",
        "",
        "| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |",
        "| --- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.system} | "
            f"{r.recall_at_k.get(1, 0):.3f} | "
            f"{r.recall_at_k.get(5, 0):.3f} | "
            f"{r.recall_at_k.get(10, 0):.3f} | "
            f"{r.store_total_s:.1f}s | "
            f"{r.retrieve_avg_ms:.1f}ms |"
        )

    lines += [
        "",
        "## Per-category R@5",
        "",
        "| System | " + " | ".join(
            f"{c} R@5" for c in CATEGORY_NAMES.values()
        ) + " |",
        "| --- | " + " | ".join(":---:" for _ in CATEGORY_NAMES) + " |",
    ]
    for r in results:
        row = [r.system]
        for cat in CATEGORY_NAMES.values():
            val = r.recall_by_category.get(cat, {}).get(5, 0.0)
            row.append(f"{val:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    sys_by_name = {r.system: r for r in results}
    # Primary comparison finds any system starting with "soma-distilled"
    # (Direction 4a) or "soma-spatial" (Direction 4b) vs chroma-mxbai so
    # suffixed variant sweeps still get a summary row.
    primary_candidates = [
        name
        for name in sys_by_name
        if name.startswith("soma-distilled") or name.startswith("soma-spatial")
    ]
    if "chroma-mxbai" in sys_by_name and primary_candidates:
        baseline_r5 = sys_by_name["chroma-mxbai"].recall_at_k.get(5, 0)
        lines += [
            "",
            "## Primary comparison (soma-distilled / soma-spatial variants vs chroma-mxbai on R@5)",
            "",
            "| Variant | R@5 | Delta | Verdict |",
            "| --- | :---: | :---: | :---: |",
        ]
        for name in primary_candidates:
            r5 = sys_by_name[name].recall_at_k.get(5, 0)
            delta = r5 - baseline_r5
            status = "SHIP" if delta > 0.02 else ("WEAK" if delta > 0.005 else "NULL")
            lines.append(
                f"| {name} | {r5:.3f} | {delta:+.3f} | {status} |"
            )

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/locomo_distill.md"),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="limit number of LoCoMo conversations (quick subset)",
    )
    p.add_argument("--target-dim", type=int, default=128)
    p.add_argument("--teacher-model", default="mxbai-embed-large")
    p.add_argument(
        "--skip-chroma",
        action="store_true",
        help="skip the chroma-mxbai baseline",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="distillation weight for soma-distilled variant",
    )
    p.add_argument(
        "--locality",
        type=float,
        default=0.0,
        help="synap max_distance for SOMA variants (0 = disabled)",
    )
    p.add_argument(
        "--skip-random",
        action="store_true",
        help="skip the soma-random comparison",
    )
    p.add_argument(
        "--variant-suffix",
        default="",
        help="append to soma variant names for distinguishing sweeps",
    )
    p.add_argument(
        "--skip-spatial",
        action="store_true",
        help="skip the soma-spatial (Direction 4b) comparison",
    )
    p.add_argument(
        "--spatial-beta",
        type=float,
        default=1.0,
        help="position_coupling_weight for soma-spatial",
    )
    p.add_argument(
        "--spatial-winners",
        type=int,
        default=3,
        help="projection_distillation_winners for soma-spatial",
    )
    p.add_argument(
        "--rerank-weight",
        type=float,
        default=0.0,
        help=(
            "SOMA retrieve_hybrid rerank_weight applied to ALL soma-* variants "
            "(soma-random / soma-distilled / soma-spatial). Default 0.0 = pure "
            "embedding cosine. LoCoMo sweep across {0.0, 0.05, 0.1, 0.2, 0.3} "
            "showed monotonic degradation past w≈0.1; pure cosine matches chroma "
            "R@5. Set >0 to opt into rerank (e.g. for non-retrieval tasks where "
            "the graph fingerprint may add signal)."
        ),
    )
    p.add_argument(
        "--gate-threshold",
        type=float,
        default=0.05,
        help=(
            "SOMA retrieve_hybrid confidence gate. Higher = stricter "
            "(rerank fires on fewer queries). Set very high (e.g. 10.0) "
            "to effectively disable rerank without zeroing the weight."
        ),
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading LoCoMo dataset...")
    all_turns, all_queries = load_locomo()
    samples = sorted({t.sample_id for t in all_turns})
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    samples_set = set(samples)
    turns_by_sample: dict[str, list[LoCoMoTurn]] = defaultdict(list)
    queries_by_sample: dict[str, list[LoCoMoQuery]] = defaultdict(list)
    for t in all_turns:
        if t.sample_id in samples_set:
            turns_by_sample[t.sample_id].append(t)
    for q in all_queries:
        if q.sample_id in samples_set:
            queries_by_sample[q.sample_id].append(q)
    n_turns = sum(len(v) for v in turns_by_sample.values())
    n_queries = sum(len(v) for v in queries_by_sample.values())
    print(f"  {len(samples)} conversations, {n_turns} turns, {n_queries} queries.")

    cache_dir = Path("benchmarks/.teacher_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = CachedEmbedder(
        teacher=OllamaEmbedder(model=args.teacher_model),
        cache_dir=str(cache_dir),
    )

    results: list[DistillResult] = []
    if not args.skip_chroma:
        print("\n=== chroma-mxbai ===")
        try:
            r = run_chroma_mxbai(samples, turns_by_sample, queries_by_sample, teacher, args.target_dim)
            results.append(r)
            print(f"  R@1={r.recall_at_k[1]:.3f} R@5={r.recall_at_k[5]:.3f} R@10={r.recall_at_k[10]:.3f} retrieve={r.retrieve_avg_ms:.1f}ms")
        except ImportError as e:
            print(f"  SKIP: {e}")

    if not args.skip_random:
        print(
            f"\n=== soma-random (frozen projections, rerank_w={args.rerank_weight}, "
            f"gate={args.gate_threshold}) ==="
        )
        r = run_soma_predictive(
            f"soma-random{args.variant_suffix}",
            samples, turns_by_sample, queries_by_sample, teacher, args.target_dim,
            projection_mode="frozen_random",
            distillation_target="none",
            distillation_weight=0.0,
            synap_locality=args.locality,
            device=device,
            rerank_weight=args.rerank_weight,
            gate_threshold=args.gate_threshold,
        )
        results.append(r)
        print(f"  R@1={r.recall_at_k[1]:.3f} R@5={r.recall_at_k[5]:.3f} R@10={r.recall_at_k[10]:.3f} retrieve={r.retrieve_avg_ms:.1f}ms")

    print(
        f"\n=== soma-distilled (learnable + mxbai distill, alpha={args.alpha}, "
        f"locality={args.locality}, rerank_w={args.rerank_weight}, dim={args.target_dim}) ==="
    )
    r = run_soma_predictive(
        f"soma-distilled{args.variant_suffix}",
        samples, turns_by_sample, queries_by_sample, teacher, args.target_dim,
        projection_mode="learnable",
        distillation_target="llm_embedding",
        distillation_weight=args.alpha,
        synap_locality=args.locality,
        device=device,
        rerank_weight=args.rerank_weight,
        gate_threshold=args.gate_threshold,
    )
    results.append(r)
    print(f"  R@1={r.recall_at_k[1]:.3f} R@5={r.recall_at_k[5]:.3f} R@10={r.recall_at_k[10]:.3f} retrieve={r.retrieve_avg_ms:.1f}ms")

    if not args.skip_spatial:
        print(
            f"\n=== soma-spatial (learnable + llm_spatial + positions "
            f"beta={args.spatial_beta} K={args.spatial_winners}) ==="
        )
        r = run_soma_predictive(
            f"soma-spatial{args.variant_suffix}",
            samples, turns_by_sample, queries_by_sample, teacher, args.target_dim,
            projection_mode="learnable",
            distillation_target="llm_spatial",
            distillation_weight=args.alpha,
            synap_locality=args.locality,
            device=device,
            position_mode="learnable",
            position_coupling_weight=args.spatial_beta,
            projection_distillation_winners=args.spatial_winners,
            rerank_weight=args.rerank_weight,
            gate_threshold=args.gate_threshold,
        )
        results.append(r)
        print(f"  R@1={r.recall_at_k[1]:.3f} R@5={r.recall_at_k[5]:.3f} R@10={r.recall_at_k[10]:.3f} retrieve={r.retrieve_avg_ms:.1f}ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = format_markdown(results)
    args.out.write_text(md, encoding="utf-8")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "n_samples": len(samples),
                "n_turns": n_turns,
                "n_queries": n_queries,
                "results": [
                    {
                        "system": r.system,
                        "recall_at_k": r.recall_at_k,
                        "recall_by_category": r.recall_by_category,
                        "store_total_s": r.store_total_s,
                        "retrieve_avg_ms": r.retrieve_avg_ms,
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nReport: {args.out}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
