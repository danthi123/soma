"""LoCoMo retrieval with LLM-distilled projections — Direction 4a Phase 3.

Tests the primary success criterion:
  ``soma-distilled > chroma-mxbai`` on LoCoMo R@5 by > 0.02.

Architecture (different from run_locomo_locality):
This runner uses PredictiveSOMA **directly** (not via MemoryLayer +
attach_soma) because the distillation loss lives inside
``PredictiveSOMA.process_input``. MemoryLayer's ``attach_soma`` attaches
a plain SOMA that doesn't exercise the distillation path, so its
graph rerank can't benefit from distilled projections.

Systems (3 minimal, expandable to 5):
- ``chroma-mxbai``       Chroma with mxbai-embed-large embeddings.
- ``soma-random``        PredictiveSOMA, frozen random projections,
                         retrieve_hybrid(alpha=0.3).
- ``soma-distilled``     PredictiveSOMA, learnable projections +
                         distillation, retrieve_hybrid(alpha=0.3).

All use the same mxbai-embed-large teacher/encoder for the
**embedding** step so the difference is purely in graph re-ranking.

Usage:
    python -m benchmarks.run_locomo_distill \\
        --max-samples 2 \\  # quick subset first
        --out benchmarks/reports/locomo_distill_subset.md

    python -m benchmarks.run_locomo_distill \\
        --out benchmarks/reports/locomo_distill_full.md
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
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
    facts_stored: int


K_VALUES = [1, 5, 10]


def embed_corpus_with_mxbai(
    teacher: CachedEmbedder,
    texts: list[str],
    target_dim: int,
) -> torch.Tensor:
    """Embed a list of texts, truncate to target_dim, L2-normalize rows."""
    batches = []
    for t in texts:
        e = teacher.embed(t)
        if e.shape[0] >= target_dim:
            e = e[:target_dim]
        else:
            pad = torch.zeros(target_dim)
            pad[: e.shape[0]] = e
            e = pad
        # L2 normalize for cosine similarity
        e = e / (e.norm() + 1e-8)
        batches.append(e)
    return torch.stack(batches)


def _score_queries(
    hits_list: list[list[tuple[int, str, float]]],
    queries: list[LoCoMoQuery],
    turn_id_to_step: dict[str, int],
) -> tuple[dict[int, float], dict[str, dict[int, float]]]:
    """For each query, compute R@k based on evidence turn IDs.

    A "hit" is credited if any of the top-k results corresponds to an
    evidence turn of the query.
    """
    recall_hits: dict[int, int] = {k: 0 for k in K_VALUES}
    by_cat: dict[str, dict[int, int]] = {
        name: {k: 0 for k in K_VALUES} for name in CATEGORY_NAMES.values()
    }
    by_cat_total: dict[str, int] = {name: 0 for name in CATEGORY_NAMES.values()}

    for query, hits in zip(queries, hits_list, strict=False):
        cat_name = CATEGORY_NAMES.get(query.category, "unknown")
        if cat_name in by_cat_total:
            by_cat_total[cat_name] += 1

        evidence_steps = {
            turn_id_to_step[tid] for tid in query.evidence
            if tid in turn_id_to_step
        }
        if not evidence_steps:
            continue

        # Hits is list of (step, text, score) or (step, doc_id, score)
        for k in K_VALUES:
            if any(
                h[0] in evidence_steps for h in hits[:k]
            ):
                recall_hits[k] += 1
                if cat_name in by_cat[cat_name]:
                    by_cat[cat_name][k] += 1

    n = len(queries)
    recall = {k: recall_hits[k] / max(1, n) for k in K_VALUES}
    recall_by_cat = {
        name: {k: by_cat[name][k] / max(1, by_cat_total[name]) for k in K_VALUES}
        for name in by_cat
    }
    return recall, recall_by_cat


def run_chroma_mxbai(
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
    teacher: CachedEmbedder,
    target_dim: int,
) -> DistillResult:
    """Chroma with pre-computed mxbai embeddings."""
    import chromadb

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        name="locomo_mxbai", metadata={"hnsw:space": "cosine"},
    )

    # Each turn becomes a doc whose ID is its dia_id so we can look up
    # the turn index from the evidence list later.
    t0 = time.perf_counter()
    texts = [t.text for t in turns]
    ids = [t.dia_id for t in turns]
    print(f"  Embedding {len(texts)} turns with mxbai...")
    embeddings = embed_corpus_with_mxbai(teacher, texts, target_dim)
    # Chroma accepts embeddings as list[list[float]]
    col.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
    )
    store_total = time.perf_counter() - t0

    # Map dia_id -> "step" (just its index for consistent scoring with SOMA)
    turn_id_to_step = {t.dia_id: i for i, t in enumerate(turns)}

    hits_list: list[list[tuple[int, str, float]]] = []
    retrieve_times = []
    print(f"  Running {len(queries)} queries...")
    for q in queries:
        q_emb = teacher.embed(q.question)
        if q_emb.shape[0] >= target_dim:
            q_emb = q_emb[:target_dim]
        else:
            pad = torch.zeros(target_dim)
            pad[: q_emb.shape[0]] = q_emb
            q_emb = pad
        q_emb = q_emb / (q_emb.norm() + 1e-8)

        t0 = time.perf_counter()
        result = col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=max(K_VALUES),
        )
        retrieve_times.append(time.perf_counter() - t0)
        result_ids = result.get("ids", [[]])[0]
        result_docs = result.get("documents", [[]])[0]
        result_dists = result.get("distances", [[]])[0]
        hits = [
            (turn_id_to_step.get(nid, -1), doc, 1.0 - float(d))
            for nid, doc, d in zip(result_ids, result_docs, result_dists, strict=False)
        ]
        hits_list.append(hits)

    recall, recall_by_cat = _score_queries(hits_list, queries, turn_id_to_step)

    return DistillResult(
        system="chroma-mxbai",
        n_turns=len(turns),
        n_queries=len(queries),
        recall_at_k=recall,
        recall_by_category=recall_by_cat,
        store_total_s=store_total,
        retrieve_avg_ms=sum(retrieve_times) * 1000 / max(1, len(retrieve_times)),
        facts_stored=len(turns),
    )


def run_soma_predictive(
    system_name: str,
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
    seed: int = 0,
    rerank_weight: float = 0.3,
    gate_threshold: float = 0.05,
) -> DistillResult:
    """PredictiveSOMA-based retrieval using mxbai for embedding +
    graph fingerprint for rerank.
    """
    # Build a developmental PredictiveSOMA sized to the target embedding dim
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
        synaptogenesis_max_distance=synap_locality,
        neurogenesis_interval=0,  # synap-only per plan
        seed=seed,
    )
    pred = PredictiveSOMA(config=config, device=device)
    if distillation_target == "llm_embedding":
        pred.attach_teacher(teacher)

    # Storage phase: embed all turns (mxbai) + process through SOMA
    print(f"  Storing {len(turns)} turns...")
    t0 = time.perf_counter()
    texts = [t.text for t in turns]
    corpus_embeddings = embed_corpus_with_mxbai(teacher, texts, target_dim).to(device)
    turn_id_to_step: dict[str, int] = {}
    step_to_corpus_idx: dict[int, int] = {}
    for i, t in enumerate(turns):
        result = pred.process_input(
            corpus_embeddings[i], source_text=t.text,
        )
        step = result.get("global_step", i)
        turn_id_to_step[t.dia_id] = step
        step_to_corpus_idx[step] = i
    store_total = time.perf_counter() - t0
    print(f"    store_total={store_total:.1f}s")

    # Retrieval phase: embed query (mxbai) + hybrid retrieval
    hits_list: list[list[tuple[int, str, float]]] = []
    retrieve_times = []
    print(f"  Running {len(queries)} queries...")
    for q in queries:
        q_emb = teacher.embed(q.question)
        if q_emb.shape[0] >= target_dim:
            q_emb = q_emb[:target_dim]
        else:
            pad = torch.zeros(target_dim)
            pad[: q_emb.shape[0]] = q_emb
            q_emb = pad
        q_emb = (q_emb / (q_emb.norm() + 1e-8)).to(device)

        t0 = time.perf_counter()
        hits = pred.retrieve_hybrid(
            q_emb,
            corpus_embeddings=corpus_embeddings,
            corpus_step_map=step_to_corpus_idx,
            recall_k=20,
            top_k=max(K_VALUES),
            gate_threshold=gate_threshold,
            rerank_weight=rerank_weight,
        )
        retrieve_times.append(time.perf_counter() - t0)
        hits_list.append(hits)

    recall, recall_by_cat = _score_queries(hits_list, queries, turn_id_to_step)

    return DistillResult(
        system=system_name,
        n_turns=len(turns),
        n_queries=len(queries),
        recall_at_k=recall,
        recall_by_category=recall_by_cat,
        store_total_s=store_total,
        retrieve_avg_ms=sum(retrieve_times) * 1000 / max(1, len(retrieve_times)),
        facts_stored=len(turns),
    )


def format_markdown(results: list[DistillResult]) -> str:
    """Generate the markdown report table."""
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

    # Primary comparison
    sys_by_name = {r.system: r for r in results}
    if "chroma-mxbai" in sys_by_name and "soma-distilled" in sys_by_name:
        baseline_r5 = sys_by_name["chroma-mxbai"].recall_at_k.get(5, 0)
        distilled_r5 = sys_by_name["soma-distilled"].recall_at_k.get(5, 0)
        delta = distilled_r5 - baseline_r5
        status = "SHIP" if delta > 0.02 else ("WEAK" if delta > 0.005 else "NULL")
        lines += [
            "",
            "## Primary comparison",
            "",
            f"**Delta (soma-distilled - chroma-mxbai) on R@5: {delta:+.3f}** → {status}",
        ]

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
        help="skip the chroma-mxbai baseline (for when chromadb isn't installed)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading LoCoMo dataset...")
    all_turns, all_queries = load_locomo()
    samples = sorted({t.sample_id for t in all_turns})
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    turns = [t for t in all_turns if t.sample_id in set(samples)]
    queries = [q for q in all_queries if q.sample_id in set(samples)]
    print(
        f"  {len(samples)} conversations, {len(turns)} turns, "
        f"{len(queries)} queries."
    )

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
            r = run_chroma_mxbai(turns, queries, teacher, args.target_dim)
            results.append(r)
            print(f"  R@5={r.recall_at_k[5]:.3f} retrieve={r.retrieve_avg_ms:.1f}ms")
        except ImportError as e:
            print(f"  SKIP: {e}")

    print("\n=== soma-random (frozen random projections) ===")
    r = run_soma_predictive(
        "soma-random", turns, queries, teacher, args.target_dim,
        projection_mode="frozen_random",
        distillation_target="none",
        distillation_weight=0.0,
        synap_locality=0.0,
        device=device,
    )
    results.append(r)
    print(f"  R@5={r.recall_at_k[5]:.3f} retrieve={r.retrieve_avg_ms:.1f}ms")

    print("\n=== soma-distilled (learnable + mxbai distillation) ===")
    r = run_soma_predictive(
        "soma-distilled", turns, queries, teacher, args.target_dim,
        projection_mode="learnable",
        distillation_target="llm_embedding",
        distillation_weight=0.5,
        synap_locality=0.0,
        device=device,
    )
    results.append(r)
    print(f"  R@5={r.recall_at_k[5]:.3f} retrieve={r.retrieve_avg_ms:.1f}ms")

    # Write out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = format_markdown(results)
    args.out.write_text(md, encoding="utf-8")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "n_samples": len(samples),
                "n_turns": len(turns),
                "n_queries": len(queries),
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
