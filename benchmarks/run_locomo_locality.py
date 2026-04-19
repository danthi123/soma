"""LoCoMo retrieval with and without locality filter.

The synthetic-dataset locality ablation (commit 916a0c7) showed
weak effect — only 1/3 seeds benefited at the most-aggressive alpha.
Possible reasons: 50 facts is too small to stress the graph; the
cosine baseline already saturates at ~0.923 so there's no room
to improve.

LoCoMo is the real-world test: 5000+ turns across 10 long
conversations, 1986 queries with evidence-turn annotations. If
locality helps retrieval at all, this is where we'd see it.

Systems (4):
- ``soma-flat``                 sbert cosine, no graph (baseline).
- ``soma-graph-no-locality``    attach_soma, alpha=0.3, locality=0.0.
- ``soma-graph-locality``       attach_soma, alpha=0.3, locality=0.5.
- ``chroma``                    external reference.

All SOMA variants use active-growth config (synap_interval=10,
rate=2.0, via developmental() base).

Run::

    python -m benchmarks.run_locomo_locality \\
        --out benchmarks/reports/locomo_locality.md
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
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
from benchmarks.run_locomo import (
    K_VALUES,
    LoCoMoResult,
    _run_one_system,
    _score_recall,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/locomo_locality.md"),
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="graph_rerank_alpha for the graph variants",
    )
    args = p.parse_args()

    print("Loading LoCoMo dataset...")
    turns, queries = load_locomo()
    samples = sorted({t.sample_id for t in turns})
    print(
        f"  {len(samples)} conversations, {len(turns)} turns, "
        f"{len(queries)} queries with evidence."
    )

    # Use fixed seed so the locality-on vs locality-off comparison is
    # paired (same rng init, only synap_max_distance differs).
    seed = 0

    systems = [
        ("soma-flat", SomaAdapter(use_sbert=True)),
        (
            "soma-graph-no-locality",
            SomaAdapter(
                use_sbert=True,
                attach_soma=True,
                graph_rerank_alpha=args.alpha,
                graph_rerank_stable_capture=False,
                synap_locality=0.0,
                synap_interval=10,
                synap_rate=2.0,
                seed=seed,
            ),
        ),
        (
            "soma-graph-locality",
            SomaAdapter(
                use_sbert=True,
                attach_soma=True,
                graph_rerank_alpha=args.alpha,
                graph_rerank_stable_capture=False,
                synap_locality=0.5,
                synap_interval=10,
                synap_rate=2.0,
                seed=seed,
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

    # ---- Build report ----
    lines = [
        "# LoCoMo Retrieval: Locality Filter Ablation",
        "",
        f"**Dataset:** LoCoMo — {len(samples)} conversations, "
        f"{len(turns)} turns, {len(queries)} queries with evidence.",
        f"**Alpha (graph variants):** {args.alpha}",
        f"**SOMA seed:** {seed}",
        "",
        "**Question:** does the v0.5 locality finding (commit 2ba566b) "
        "transfer to a real-world agent-memory retrieval workload?",
        "",
        "## Headline",
        "",
        "| System | R@1 | R@5 | R@10 | Retrieve (ms) | Store total (s) | Disk (MB) |",
        "| --- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.system} | {r.recall_at_k[1]:.3f} | {r.recall_at_k[5]:.3f} | "
            f"{r.recall_at_k[10]:.3f} | {r.retrieve_avg_ms:.2f} | "
            f"{r.store_total_s:.1f} | {r.disk_kb / 1024:.1f} |"
        )

    # Paired comparison between the two graph variants
    no_loc = next(r for r in results if r.system == "soma-graph-no-locality")
    with_loc = next(r for r in results if r.system == "soma-graph-locality")
    lines += [
        "",
        "## Paired comparison: graph rerank without vs with locality",
        "",
        "| k | no-locality R@k | locality R@k | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in K_VALUES:
        d = with_loc.recall_at_k[k] - no_loc.recall_at_k[k]
        sign = "+" if d >= 0 else ""
        lines.append(
            f"| {k} | {no_loc.recall_at_k[k]:.3f} | "
            f"{with_loc.recall_at_k[k]:.3f} | {sign}{d:.3f} |"
        )

    lines += [
        "",
        "## Recall@5 by question category",
        "",
        "| System | " + " | ".join(f"{c} R@5" for c in CATEGORY_NAMES) + " |",
        "| --- | " + " | ".join(":---:" for _ in CATEGORY_NAMES) + " |",
    ]
    for r in results:
        row = [r.system]
        for cat in CATEGORY_NAMES:
            val = r.recall_by_category.get(cat, {}).get(5, 0.0)
            row.append(f"{val:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Interpretation",
        "",
        "If `soma-graph-locality` R@k meets or beats `soma-graph-no-locality`, "
        "the v0.5 locality principle transfers to real-world retrieval. "
        "Look especially at multi-hop and temporal categories — these "
        "stress the graph's structural signal the most.",
        "",
        "If the gap is near zero, the v0.5 finding is prediction-task-specific; "
        "retrieval's cosine baseline is already strong enough that graph "
        "adjustments (filtered or not) don't change top-K materially.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_locomo_locality.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
