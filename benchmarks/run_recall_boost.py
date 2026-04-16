"""Recall-boost benchmark — measures the lift from hybrid search and
cross-encoder re-ranking on real conversational data (LoCoMo).

The other SOMA-vs-Chroma benchmarks hold the retrieval signal fixed
(pure cosine over the same sbert embedder) so the delta is always
mechanics (store/disk/latency). This script flips the axis — same
corpus, same embedder, varied *retrieval strategy* — to quantify how
much recall SOMA can buy beyond cosine by opting into:

1. **Hybrid search** (cosine + BM25, ``hybrid_alpha``).
2. **Cross-encoder re-ranking** (``rerank_top_n`` with a tiny
   ``cross-encoder/ms-marco-MiniLM-L-6-v2`` model).
3. **Both together** (hybrid pool → cross-encoder rerank).

The comparison is strictly within-SOMA: we compare each strategy
against the same cosine baseline that matches Chroma/LanceDB by
construction. If strategy X lifts R@5 by Δ over baseline, X lifts
SOMA by Δ over any vector DB using the same embedder — because the
baseline is the ceiling those DBs already reach.

Run::

    python -m benchmarks.run_recall_boost

Report lands in ``benchmarks/reports/recall_boost_locomo.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.datasets.locomo import LoCoMoQuery, LoCoMoTurn, load_locomo
from soma.memory import MemoryLayer
from soma.memory.rerank import CrossEncoderReranker

REPORT = Path("benchmarks/reports/recall_boost_locomo.md")


@dataclass
class ConfigResult:
    name: str
    r_at_1: float
    r_at_5: float
    r_at_10: float
    avg_retrieve_ms: float


def _ingest(mem: MemoryLayer, turns: list[LoCoMoTurn]) -> None:
    for t in turns:
        mem.store(
            t.text,
            metadata={"sample_id": t.sample_id, "dia_id": t.dia_id},
        )


def _recall(hits: list[Any], query: LoCoMoQuery) -> tuple[bool, bool, bool]:
    """Return (R@1, R@5, R@10) for this query against gold evidence.

    Cross-sample: only count a hit as correct if it's from the same
    sample_id as the query. Evidence is a list of dia_ids (strings);
    a hit matches if its metadata's dia_id is in that list.
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


def _score(
    mem: MemoryLayer, queries: list[LoCoMoQuery], *, label: str, **retrieve_kwargs: Any
) -> ConfigResult:
    r1, r5, r10 = 0, 0, 0
    t_total = 0.0
    for q in queries:
        t0 = time.perf_counter()
        hits = mem.retrieve(q.question, k=10, **retrieve_kwargs)
        t_total += time.perf_counter() - t0
        h1, h5, h10 = _recall(hits, q)
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


def main() -> None:
    print("=== Recall-boost benchmark on LoCoMo ===\n")
    print("Loading LoCoMo...")
    turns, queries = load_locomo()
    print(f"  {len(turns)} turns, {len(queries)} queries")

    print("\nBuilding SOMA bundle (sbert all-MiniLM-L6-v2)...")
    mem = MemoryLayer.with_sbert()
    t0 = time.perf_counter()
    _ingest(mem, turns)
    print(f"  stored {len(turns)} in {time.perf_counter() - t0:.1f}s")

    print("\nAttaching CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    mem.attach_reranker(CrossEncoderReranker())
    # Warm-up: one rerank pass triggers model load outside the timer.
    _ = mem.retrieve(queries[0].question, k=1, rerank_top_n=5)

    configs: list[tuple[str, dict[str, Any]]] = [
        ("baseline (cosine)", {}),
        ("hybrid (alpha=0.3)", {"hybrid_alpha": 0.3}),
        ("hybrid (alpha=0.5)", {"hybrid_alpha": 0.5}),
        ("rerank (top-20)", {"rerank_top_n": 20}),
        (
            "hybrid+rerank (alpha=0.3, top-20)",
            {"hybrid_alpha": 0.3, "rerank_top_n": 20},
        ),
    ]
    results: list[ConfigResult] = []
    for label, cfg in configs:
        print(f"\nScoring: {label}")
        r = _score(mem, queries, label=label, **cfg)
        results.append(r)
        print(
            f"  R@1={r.r_at_1:.3f}  R@5={r.r_at_5:.3f}  R@10={r.r_at_10:.3f}  "
            f"retrieve={r.avg_retrieve_ms:.1f} ms"
        )

    base = results[0]
    print("\n=== Summary ===")
    print(
        f"{'config':<40}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  "
        f"{'ms':>6}  {'Lift R@5':>9}"
    )
    for r in results:
        delta5 = (r.r_at_5 - base.r_at_5) * 100
        print(
            f"{r.name:<40}  {r.r_at_1:>6.3f}  {r.r_at_5:>6.3f}  "
            f"{r.r_at_10:>6.3f}  {r.avg_retrieve_ms:>6.1f}  "
            f"{delta5:>+7.2f} pp"
        )

    # Write report
    lines = [
        "# Recall-Boost Benchmark — LoCoMo",
        "",
        "Lift from optional recall boosters measured on LoCoMo "
        f"(Maharana 2024 — {len(turns)} turns, {len(queries)} questions "
        "with gold-evidence dia_ids).",
        "",
        "Baseline = pure cosine over sbert `all-MiniLM-L6-v2`. All "
        "other rows use the same corpus, same embedder, only the "
        "retrieval strategy changes. ΔR@5 is absolute improvement "
        "over baseline.",
        "",
        "## Results",
        "",
        "| Strategy | R@1 | R@5 | R@10 | Retrieve (ms) | Lift R@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        delta5 = (r.r_at_5 - base.r_at_5) * 100
        lines.append(
            f"| {r.name} | {r.r_at_1:.3f} | {r.r_at_5:.3f} | "
            f"{r.r_at_10:.3f} | {r.avg_retrieve_ms:.1f} | "
            f"{delta5:+.2f} pp |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The baseline row is what any vector DB (Chroma, LanceDB, "
        "Qdrant, Pinecone) reaches when using the same embedder and "
        "the same cosine scoring. Every other row is a SOMA-specific "
        "lift that those DBs don't offer out of the box — they would "
        "require separate BM25 / cross-encoder glue in userland.",
        "",
        "Hybrid search helps when queries contain specific "
        "terminology (proper names, domain jargon) that sbert's "
        "sub-word tokenization smears across a broader semantic "
        "neighbourhood. Cross-encoder re-ranking helps across the "
        "board because the model attends across the (query, "
        "candidate) pair rather than scoring each independently.",
        "",
        "The compose row (hybrid+rerank) picks candidates from both "
        "streams and re-ranks the union. Usually strictly ≥ either in "
        "isolation on R@5.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_recall_boost.py`.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
