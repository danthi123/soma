"""Recall-boost benchmark on LoCoMo using a STRONG embedder (mxbai).

The existing run_recall_boost uses sbert all-MiniLM-L6-v2 which is a
weak embedder — its cosine baseline lands at R@5=0.238. On that weak
baseline, hybrid BM25 + cross-encoder rerank lifts R@5 to 0.450 (+89%
relative). Open question: does the lift still hold when the cosine
baseline is already strong?

This script runs the same 5 SOMA configs (baseline, hybrid α=0.3,
hybrid α=0.5, rerank top-20, hybrid+rerank) AND includes a fair
Chroma-with-same-reranker comparison so we can cite:

  chroma-mxbai (baseline)        vs
  chroma-mxbai + rerank (fair)   vs
  soma-mxbai (baseline)          vs
  soma-mxbai + hybrid+rerank (kitchen sink)

Same mxbai-embed-large teacher on all, same cross-encoder reranker,
same LoCoMo corpus. The target is: show whether SOMA's opt-in BM25 +
rerank produces a real lift that chroma+same-reranker can't match.

Run::

    python -m benchmarks.run_recall_boost_mxbai

Report lands in ``benchmarks/reports/recall_boost_locomo_mxbai.md``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.datasets.locomo import LoCoMoQuery, LoCoMoTurn, load_locomo
from soma.llm.embedders import CachedEmbedder, OllamaEmbedder
from soma.memory import MemoryLayer
from soma.memory.rerank import CrossEncoderReranker

REPORT = Path("benchmarks/reports/recall_boost_locomo_mxbai.md")


@dataclass
class ConfigResult:
    name: str
    r_at_1: float
    r_at_5: float
    r_at_10: float
    avg_retrieve_ms: float


def _build_mxbai_embed_fn(
    target_dim: int,
) -> tuple[Any, int, CachedEmbedder]:
    """Wrap the cached ollama mxbai embedder as an EmbedFn that truncates
    to ``target_dim`` and L2-normalizes. Returns (embed_fn, dim, teacher)
    so the teacher can be reused for the Chroma + rerank leg below."""
    cache_dir = Path("benchmarks/.teacher_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = CachedEmbedder(
        teacher=OllamaEmbedder(model="mxbai-embed-large"),
        cache_dir=str(cache_dir),
    )

    def _embed(text: str) -> torch.Tensor:
        e = teacher.embed(text)
        if e.shape[0] >= target_dim:
            e = e[:target_dim]
        else:
            pad = torch.zeros(target_dim)
            pad[: e.shape[0]] = e
            e = pad
        return e / (e.norm() + 1e-8)

    return _embed, target_dim, teacher


def _ingest_soma(mem: MemoryLayer, turns: list[LoCoMoTurn]) -> None:
    for t in turns:
        mem.store(
            t.text,
            metadata={"sample_id": t.sample_id, "dia_id": t.dia_id},
        )


def _recall_soma(hits: list[Any], query: LoCoMoQuery) -> tuple[bool, bool, bool]:
    """(R@1, R@5, R@10) for a single query.

    Cross-sample: only count a hit if it's from the same sample_id as
    the query. Evidence is a list of dia_ids (strings); a hit matches
    if its metadata's dia_id is in that list.
    """
    gold = set(query.evidence or [])
    top10 = [h for h in hits[:10] if h.metadata.get("sample_id") == query.sample_id]
    hit_ranks = [
        i for i, h in enumerate(top10) if h.metadata.get("dia_id") in gold
    ]
    if not hit_ranks:
        return False, False, False
    first = hit_ranks[0]
    return first < 1, first < 5, first < 10


def _score_soma(
    mem: MemoryLayer,
    queries: list[LoCoMoQuery],
    *,
    label: str,
    **retrieve_kwargs: Any,
) -> ConfigResult:
    r1 = r5 = r10 = 0
    t_total = 0.0
    for q in queries:
        t0 = time.perf_counter()
        hits = mem.retrieve(q.question, k=10, **retrieve_kwargs)
        t_total += time.perf_counter() - t0
        h1, h5, h10 = _recall_soma(hits, q)
        r1 += h1
        r5 += h5
        r10 += h10
    n = max(1, len(queries))
    return ConfigResult(
        name=label,
        r_at_1=r1 / n,
        r_at_5=r5 / n,
        r_at_10=r10 / n,
        avg_retrieve_ms=1000 * t_total / n,
    )


def _score_chroma(
    teacher: CachedEmbedder,
    target_dim: int,
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
    reranker: CrossEncoderReranker | None,
    label: str,
) -> ConfigResult:
    """LoCoMo protocol: one chroma collection per sample. Optional
    cross-encoder reranker applied to top-20 candidates before returning."""
    import chromadb
    from collections import defaultdict

    turns_by_sample: dict[str, list[LoCoMoTurn]] = defaultdict(list)
    for t in turns:
        turns_by_sample[t.sample_id].append(t)

    queries_by_sample: dict[str, list[LoCoMoQuery]] = defaultdict(list)
    for q in queries:
        queries_by_sample[q.sample_id].append(q)

    r1 = r5 = r10 = 0
    n = 0
    t_total = 0.0
    for sid, sample_turns in turns_by_sample.items():
        sample_queries = queries_by_sample.get(sid, [])
        if not sample_queries:
            continue

        client = chromadb.EphemeralClient()
        col = client.get_or_create_collection(
            name=f"locomo_{sid}", metadata={"hnsw:space": "cosine"},
        )
        texts = [t.text for t in sample_turns]
        ids = [t.dia_id for t in sample_turns]
        id_to_text = {t.dia_id: t.text for t in sample_turns}

        # Embed + L2-normalize each corpus text to target_dim.
        embeds = []
        for tx in texts:
            e = teacher.embed(tx)
            if e.shape[0] >= target_dim:
                e = e[:target_dim]
            else:
                pad = torch.zeros(target_dim)
                pad[: e.shape[0]] = e
                e = pad
            e = e / (e.norm() + 1e-8)
            embeds.append(e.tolist())
        col.add(ids=ids, documents=texts, embeddings=embeds)

        for q in sample_queries:
            t0 = time.perf_counter()
            q_emb = teacher.embed(q.question)
            if q_emb.shape[0] >= target_dim:
                q_emb = q_emb[:target_dim]
            else:
                pad = torch.zeros(target_dim)
                pad[: q_emb.shape[0]] = q_emb
                q_emb = pad
            q_emb = q_emb / (q_emb.norm() + 1e-8)

            # Retrieve candidate pool. If a reranker is attached, over-
            # fetch 20 and rerank; else return top-10 directly.
            pool_k = 20 if reranker is not None else 10
            result = col.query(
                query_embeddings=[q_emb.tolist()],
                n_results=pool_k,
            )
            candidate_ids: list[str] = result.get("ids", [[]])[0]

            if reranker is not None and candidate_ids:
                # Cross-encoder scores (query, doc) independently per doc,
                # sorted descending.
                docs = [id_to_text[cid] for cid in candidate_ids]
                scores = reranker.score(q.question, docs)
                order = sorted(range(len(docs)), key=lambda i: -scores[i])
                top10_ids = [candidate_ids[i] for i in order[:10]]
            else:
                top10_ids = candidate_ids[:10]

            t_total += time.perf_counter() - t0

            gold = set(q.evidence or [])
            hit_ranks = [i for i, cid in enumerate(top10_ids) if cid in gold]
            n += 1
            if hit_ranks:
                first = hit_ranks[0]
                r1 += int(first < 1)
                r5 += int(first < 5)
                r10 += int(first < 10)

    return ConfigResult(
        name=label,
        r_at_1=r1 / max(1, n),
        r_at_5=r5 / max(1, n),
        r_at_10=r10 / max(1, n),
        avg_retrieve_ms=1000 * t_total / max(1, n),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-dim", type=int, default=128)
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="limit LoCoMo conversations (quick subset for smoke)",
    )
    p.add_argument(
        "--skip-chroma", action="store_true",
        help="skip the chroma-fair comparisons",
    )
    args = p.parse_args()

    print("=== Recall-boost benchmark on LoCoMo (mxbai-embed-large) ===\n")
    print("Loading LoCoMo...")
    turns, queries = load_locomo()
    if args.max_samples is not None:
        kept = sorted({t.sample_id for t in turns})[: args.max_samples]
        kept_set = set(kept)
        turns = [t for t in turns if t.sample_id in kept_set]
        queries = [q for q in queries if q.sample_id in kept_set]
    print(f"  {len(turns)} turns, {len(queries)} queries")

    print(f"\nBuilding SOMA MemoryLayer (mxbai, target_dim={args.target_dim})...")
    embed_fn, dim, teacher = _build_mxbai_embed_fn(args.target_dim)
    mem = MemoryLayer(embed_fn=embed_fn, embed_dim=dim)
    t0 = time.perf_counter()
    _ingest_soma(mem, turns)
    print(f"  stored {len(turns)} in {time.perf_counter() - t0:.1f}s")

    print("\nAttaching CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    reranker_shared = CrossEncoderReranker()
    mem.attach_reranker(reranker_shared)
    # Warm-up: one rerank pass triggers model load outside the timer.
    _ = mem.retrieve(queries[0].question, k=1, rerank_top_n=5)

    # SOMA configs — same as run_recall_boost.py but with mxbai.
    soma_configs: list[tuple[str, dict[str, Any]]] = [
        ("soma-mxbai baseline (cosine)", {}),
        ("soma-mxbai hybrid (alpha=0.3)", {"hybrid_alpha": 0.3}),
        ("soma-mxbai hybrid (alpha=0.5)", {"hybrid_alpha": 0.5}),
        ("soma-mxbai rerank (top-20)", {"rerank_top_n": 20}),
        (
            "soma-mxbai hybrid+rerank (a=0.3, top-20)",
            {"hybrid_alpha": 0.3, "rerank_top_n": 20},
        ),
    ]
    soma_results: list[ConfigResult] = []
    for label, cfg in soma_configs:
        print(f"\nScoring: {label}")
        r = _score_soma(mem, queries, label=label, **cfg)
        soma_results.append(r)
        print(
            f"  R@1={r.r_at_1:.3f}  R@5={r.r_at_5:.3f}  R@10={r.r_at_10:.3f}  "
            f"retrieve={r.avg_retrieve_ms:.1f} ms"
        )

    chroma_results: list[ConfigResult] = []
    if not args.skip_chroma:
        print("\nScoring: chroma-mxbai baseline (cosine only)")
        r = _score_chroma(teacher, args.target_dim, turns, queries, None,
                          "chroma-mxbai baseline")
        chroma_results.append(r)
        print(
            f"  R@1={r.r_at_1:.3f}  R@5={r.r_at_5:.3f}  R@10={r.r_at_10:.3f}  "
            f"retrieve={r.avg_retrieve_ms:.1f} ms"
        )

        print("\nScoring: chroma-mxbai + cross-encoder rerank (top-20)")
        r = _score_chroma(teacher, args.target_dim, turns, queries,
                          reranker_shared,
                          "chroma-mxbai + rerank (top-20)")
        chroma_results.append(r)
        print(
            f"  R@1={r.r_at_1:.3f}  R@5={r.r_at_5:.3f}  R@10={r.r_at_10:.3f}  "
            f"retrieve={r.avg_retrieve_ms:.1f} ms"
        )

    # Summary
    base = soma_results[0]
    all_results = soma_results + chroma_results
    print("\n=== Summary ===")
    print(
        f"{'config':<45}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  "
        f"{'ms':>6}  {'Lift R@5':>9}"
    )
    for r in all_results:
        delta5 = (r.r_at_5 - base.r_at_5) * 100
        print(
            f"{r.name:<45}  {r.r_at_1:>6.3f}  {r.r_at_5:>6.3f}  "
            f"{r.r_at_10:>6.3f}  {r.avg_retrieve_ms:>6.1f}  "
            f"{delta5:>+7.2f} pp"
        )

    lines = [
        "# Recall-Boost Benchmark — LoCoMo (mxbai-embed-large)",
        "",
        "Same 5 SOMA configs as `run_recall_boost.py` but with a STRONG "
        f"embedder (mxbai-embed-large, truncated to target_dim={args.target_dim}) "
        "in place of sbert all-MiniLM-L6-v2. Also includes chroma-mxbai "
        "with and without the same cross-encoder reranker for a fair "
        f"SOMA-vs-chroma comparison. LoCoMo subset: {len(turns)} turns, "
        f"{len(queries)} queries.",
        "",
        "## Results",
        "",
        "| Strategy | R@1 | R@5 | R@10 | Retrieve (ms) | ΔR@5 vs soma-mxbai baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in all_results:
        delta5 = (r.r_at_5 - base.r_at_5) * 100
        lines.append(
            f"| {r.name} | {r.r_at_1:.3f} | {r.r_at_5:.3f} | "
            f"{r.r_at_10:.3f} | {r.avg_retrieve_ms:.1f} | "
            f"{delta5:+.2f} pp |"
        )
    lines += [
        "",
        "## Generated by",
        "",
        "`benchmarks/run_recall_boost_mxbai.py`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
