"""LoCoMo retrieval benchmark — SOMA vs Chroma on real long conversations.

LoCoMo (Maharana et al. 2024) is the de-facto agent-memory benchmark:
10 conversations, ~588 turns each, 1,986 question-answer pairs total
with explicit evidence-turn annotations. The full LoCoMo eval scores
QA accuracy via a GPT-4 judge; we deliberately stop at retrieval
(does the system fetch the evidence turns?) so the benchmark stays
runnable without external API access. Recall@k is the cleanest
signal for the memory layer's job — generation quality is on the
LLM, not the memory.

Per conversation:
1. Store every turn as a separate entry with its dia_id in metadata.
2. For each QA, retrieve top-k against the question text.
3. Score Recall@k = "did at least one evidence turn make it into top-k?"

Aggregate Recall@k across conversations + per category. Report the
SOMA-flat / SOMA-hnsw / Chroma three-way comparison so the same
table answers "is SOMA competitive on a real-world workload?" and
"does the HNSW backend hold up at scale (10K+ entries)?"

Run::

    python -m benchmarks.run_locomo --out benchmarks/reports/locomo.md
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.datasets.locomo import (
    CATEGORY_NAMES,
    LoCoMoQuery,
    LoCoMoTurn,
    load_locomo,
    turns_for_sample,
)
from benchmarks.harness.adapters.chroma import ChromaAdapter
from benchmarks.harness.adapters.soma import SomaAdapter

K_VALUES: tuple[int, ...] = (1, 5, 10)


@dataclass
class LoCoMoResult:
    system: str
    n_turns: int
    n_queries: int
    recall_at_k: dict[int, float]
    recall_by_category: dict[str, dict[int, float]]
    store_total_s: float
    retrieve_avg_ms: float
    disk_kb: float


def _score_recall(retrieved_dia_ids: list[str], evidence: list[str], k: int) -> float:
    """1.0 if any evidence dia_id is in retrieved[:k], else 0.0."""
    return 1.0 if any(e in retrieved_dia_ids[:k] for e in evidence) else 0.0


def _run_one_system(
    name: str,
    adapter,
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
) -> LoCoMoResult:
    print(f"  [{name}] preparing...")
    adapter.prepare()

    print(f"  [{name}] storing {len(turns)} turns across 10 conversations...")
    t0 = time.perf_counter()
    # Group turns by sample so we can store per-conversation contexts
    samples = sorted({t.sample_id for t in turns})
    for sid in samples:
        sample_turns = turns_for_sample(turns, sid)
        for turn in sample_turns:
            adapter.store(
                turn.text,
                metadata={"sample_id": sid, "dia_id": turn.dia_id, "speaker": turn.speaker},
            )
    store_total = time.perf_counter() - t0

    # Per-conversation index: collect dia_ids that belong to each sample
    # so we can match retrieved nodes back to evidence dia_ids.
    print(f"  [{name}] running {len(queries)} retrieval queries...")
    recall_sums: dict[int, float] = {k: 0.0 for k in K_VALUES}
    cat_sums: dict[str, dict[int, list[float]]] = {
        CATEGORY_NAMES.get(i, str(i)): {k: [] for k in K_VALUES}
        for i in range(1, 6)
    }
    retrieve_times: list[float] = []
    max_k = max(K_VALUES)

    # Warmup
    adapter.retrieve(queries[0].question, k=max_k)

    for q in queries:
        t1 = time.perf_counter()
        hits = adapter.retrieve(q.question, k=max_k)
        retrieve_times.append(time.perf_counter() - t1)
        # Filter to same sample (cross-sample matches don't count —
        # would be cheating; LoCoMo's evidence is intra-conversation).
        same_sample = [
            h for h in hits if h.metadata.get("sample_id") == q.sample_id
        ]
        retrieved_dia_ids = [h.metadata.get("dia_id", "") for h in same_sample]
        cat_name = CATEGORY_NAMES.get(q.category, str(q.category))
        for k in K_VALUES:
            score = _score_recall(retrieved_dia_ids, q.evidence, k)
            recall_sums[k] += score
            cat_sums[cat_name][k].append(score)

    n = len(queries)
    recall_avg = {k: recall_sums[k] / max(1, n) for k in K_VALUES}
    recall_by_cat = {
        cat: {
            k: sum(scores) / max(1, len(scores))
            for k, scores in by_k.items()
        }
        for cat, by_k in cat_sums.items()
    }
    disk = adapter.disk_footprint_bytes() / 1024.0
    adapter.teardown()

    return LoCoMoResult(
        system=name,
        n_turns=len(turns),
        n_queries=len(queries),
        recall_at_k=recall_avg,
        recall_by_category=recall_by_cat,
        store_total_s=store_total,
        retrieve_avg_ms=sum(retrieve_times) * 1000 / max(1, len(retrieve_times)),
        disk_kb=disk,
    )


def _format_main_table(results: list[LoCoMoResult]) -> str:
    header_cols = ["System"] + [f"R@{k}" for k in K_VALUES] + [
        "Retrieve (ms)", "Store total (s)", "Disk (MB)",
    ]
    sep = ["---"] + [":---:"] * (len(K_VALUES) + 3)
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in results:
        cells = [r.system]
        for k in K_VALUES:
            cells.append(f"{r.recall_at_k[k]:.3f}")
        cells.extend([
            f"{r.retrieve_avg_ms:.2f}",
            f"{r.store_total_s:.1f}",
            f"{r.disk_kb / 1024:.1f}",
        ])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_category_table(results: list[LoCoMoResult], k: int) -> str:
    cat_names = [CATEGORY_NAMES[i] for i in range(1, 6)]
    header_cols = ["System"] + [f"{c} R@{k}" for c in cat_names]
    sep = ["---"] + [":---:"] * len(cat_names)
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in results:
        cells = [r.system]
        for cat in cat_names:
            score = r.recall_by_category.get(cat, {}).get(k, float("nan"))
            cells.append(f"{score:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/locomo.md"),
    )
    args = p.parse_args()

    print("Loading LoCoMo dataset...")
    turns, queries = load_locomo()
    samples = sorted({t.sample_id for t in turns})
    print(
        f"  {len(samples)} conversations, {len(turns)} turns, "
        f"{len(queries)} queries with evidence."
    )

    systems = [
        ("soma-flat", SomaAdapter(use_sbert=True)),
        (
            "soma-hnsw",
            SomaAdapter(
                use_sbert=True, faiss_index_type="hnsw", faiss_threshold=500,
            ),
        ),
        ("chroma", ChromaAdapter()),
    ]

    results: list[LoCoMoResult] = []
    for name, adapter in systems:
        print(f"\n=== {name} ===")
        r = _run_one_system(name, adapter, turns, queries)
        results.append(r)
        print(
            f"  Recall@1={r.recall_at_k[1]:.3f} "
            f"Recall@5={r.recall_at_k[5]:.3f} "
            f"Recall@10={r.recall_at_k[10]:.3f} "
            f"retrieve={r.retrieve_avg_ms:.1f}ms"
        )

    lines = [
        "# LoCoMo Retrieval Benchmark — SOMA vs Chroma",
        "",
        f"**Dataset:** LoCoMo (Maharana et al. 2024) — {len(samples)} "
        f"conversations, {len(turns)} dialogue turns, "
        f"{len(queries)} questions with evidence-turn annotations.",
        "",
        "**What's measured:** retrieval Recall@k (did at least one of "
        "the question's gold-evidence turns make it into the top-k? "
        "Cross-sample retrievals are excluded — LoCoMo's evidence is "
        "intra-conversation so cross-sample hits would be cheating). "
        "We deliberately do *not* run the LoCoMo paper's GPT-4 judge "
        "for QA accuracy; that part is the LLM's job, not the memory "
        "layer's. Recall@k cleanly isolates the memory contribution.",
        "",
        "## Headline",
        "",
        _format_main_table(results),
        "",
        "## Recall@5 by Question Category",
        "",
        _format_category_table(results, k=5),
        "",
        "## Interpretation",
        "",
        "Single-hop questions (one fact lookup) are the bread-and-butter "
        "vector-retrieval case; SOMA and Chroma should be near-tied. "
        "Multi-hop and temporal questions stress the index more — the "
        "evidence may be split across distant turns. Adversarial "
        "questions are designed to be hard or unanswerable; low Recall "
        "there is expected and the gap between systems is the signal "
        "of interest.",
        "",
        "Open-domain questions are the largest category (841 of 1986); "
        "they ask about facts mentioned anywhere in the conversation "
        "and are the closest to a real RAG-over-conversation workload.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_locomo.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
