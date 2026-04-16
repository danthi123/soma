"""Graph re-rank alpha + stable-capture ablation on sbert embeddings.

The headline retrieval benchmark (``run_retrieval.py``) shows that
graph-aware re-ranking as shipped hurts Recall@3 by ~0.13 at the
default ``alpha=0.3``. Two candidate causes:

1. ``alpha`` is too aggressive — cosine on sbert is already strong;
   blending even a modest graph signal drags rankings toward noise.
2. Stored activations are snapshotted mid-consolidation in different
   graph states, so the q_act/stored_act comparison is across
   mismatched graph realities.

This script sweeps alpha in {0.0, 0.05, 0.1, 0.2, 0.3, 0.5} and
optionally forces a stable-capture pass (eval_mode post-growth) so we
can tell which lever matters. alpha=0.0 equals the flat baseline and
serves as the sanity check.

Run::

    python -m benchmarks.run_graph_ablation \\
        --out benchmarks/reports/graph_ablation.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.datasets.synthetic import generate_dataset
from benchmarks.harness.adapters.soma import SomaAdapter
from benchmarks.harness.runner import format_results_table, run_retrieval_benchmark

ALPHAS: list[float] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def _run(alpha: float, stable: bool, facts, queries, k):
    adapter = SomaAdapter(
        use_sbert=True,
        attach_soma=True,
        graph_rerank_alpha=alpha,
        graph_rerank_stable_capture=stable,
    )
    adapter.name = f"soma-sbert-graph(a={alpha:.2f},stable={int(stable)})"
    [result] = run_retrieval_benchmark(
        systems=[adapter],
        dataset=facts,
        queries=queries,
        k=k,
        consolidate_after_store=True,
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/graph_ablation.md"),
    )
    p.add_argument("--k", type=int, default=3)
    p.add_argument(
        "--shuffle",
        action="store_true",
        help=(
            "Shuffle the fact order. Use this to check whether any graph "
            "gain is actually consolidation-order artifact on the "
            "topic-clustered default."
        ),
    )
    args = p.parse_args()

    facts, topic_facts, queries = generate_dataset(shuffle=args.shuffle)
    print(
        f"Dataset: {len(facts)} facts, "
        f"{len({t.topic for t in topic_facts})} topics, "
        f"{len(queries)} queries.\n"
    )

    # Flat baseline (no graph re-rank, alpha irrelevant)
    print("=== Baseline: flat cosine (no graph) ===")
    flat_adapter = SomaAdapter(use_sbert=True, attach_soma=False)
    flat_adapter.name = "soma-sbert-flat"
    [flat_result] = run_retrieval_benchmark(
        systems=[flat_adapter],
        dataset=facts,
        queries=queries,
        k=args.k,
    )

    unstable_results = []
    for alpha in ALPHAS:
        print(f"\n=== alpha={alpha:.2f}, stable=False ===")
        unstable_results.append(_run(alpha, False, facts, queries, args.k))

    stable_results = []
    for alpha in ALPHAS:
        print(f"\n=== alpha={alpha:.2f}, stable=True ===")
        stable_results.append(_run(alpha, True, facts, queries, args.k))

    lines = [
        "# Graph Re-Rank Ablation — Alpha Sweep + Stable Capture",
        "",
        f"**Dataset:** {len(facts)} facts, {len(queries)} queries, k={args.k}.",
        "",
        "## Baseline (no graph re-rank)",
        "",
        format_results_table([flat_result], k=args.k),
        "",
        "## Graph re-rank, mid-consolidation activation capture",
        "",
        "Stored activations are snapshotted during growth (``eval_mode=False``) — "
        "each entry is captured in a different graph state. This is the "
        "shipped behavior before the stable-capture flag.",
        "",
        format_results_table(unstable_results, k=args.k),
        "",
        "## Graph re-rank, post-growth stable activation capture",
        "",
        "After growth completes, the same texts are fed through the final "
        "graph in ``eval_mode=True`` to re-capture all stored activations "
        "in a single consistent graph state, matching the state the query "
        "sees at retrieval time.",
        "",
        format_results_table(stable_results, k=args.k),
        "",
        "## Headline",
        "",
    ]

    best_unstable = max(unstable_results, key=lambda r: r.recall_at_k)
    best_stable = max(stable_results, key=lambda r: r.recall_at_k)

    lines += [
        f"- Flat baseline Recall@{args.k}: **{flat_result.recall_at_k:.3f}**",
        (
            f"- Best unstable-capture graph Recall@{args.k}: "
            f"**{best_unstable.recall_at_k:.3f}** (`{best_unstable.system}`)"
        ),
        (
            f"- Best stable-capture graph Recall@{args.k}: "
            f"**{best_stable.recall_at_k:.3f}** (`{best_stable.system}`)"
        ),
        "",
        "Interpretation: if stable-capture best matches or beats flat, "
        "the non-stationarity hypothesis is supported. If alpha=0.0 rows "
        "match flat exactly, the re-rank machinery is behaving correctly "
        "and the graph signal itself is what hurts.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_graph_ablation.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
