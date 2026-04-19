"""Locality ablation: does synap_locality improve retrieval with graph re-rank?

The v0.5 capacity-schedule experiments established that a hard
positional-locality filter on synaptogenesis turns a harmful mechanism
into a beneficial one (commits 2ba566b, 28c5329). At matched total
admissions the locality variant beats random-selection controls 7-8/8
across seeds on a synthetic prediction task.

This script tests whether the finding TRANSFERS to a semantic
retrieval task — the product-facing workload for SOMA's
agent-memory positioning.

Setup: same synthetic 50-fact / 26-query retrieval dataset used by
``run_graph_ablation.py``. Compares locality on (synap_locality=0.5)
vs off (0.0) at several graph-rerank alpha values. If locality
improves Recall@k at the alphas where graph re-rank helps, the
locality principle transfers to the product layer.

Run::

    python -m benchmarks.run_locality_ablation \\
        --out benchmarks/reports/locality_ablation.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.datasets.synthetic import generate_dataset
from benchmarks.harness.adapters.soma import SomaAdapter
from benchmarks.harness.runner import format_results_table, run_retrieval_benchmark

ALPHAS: list[float] = [0.0, 0.1, 0.2, 0.3]
LOCALITY_VALUES: list[float] = [0.0, 0.5]
SEEDS: list[int] = [0, 1, 42]


def _run(alpha: float, locality: float, seed: int, facts, queries, k):
    adapter = SomaAdapter(
        use_sbert=True,
        attach_soma=True,
        graph_rerank_alpha=alpha,
        graph_rerank_stable_capture=False,  # unstable wins on this dataset
        synap_locality=locality,
        # Active-growth overrides: SOMAConfig whitepaper defaults have
        # synap_interval=100, rate=0.01, which produces 0 admissions on
        # a 50-fact benchmark flow (verified via instrumentation). Use
        # developmental-style cadence so synap actually fires and the
        # locality filter has something to filter.
        synap_interval=10,
        synap_rate=2.0,
        seed=seed,
    )
    adapter.name = f"soma-locality={locality:.2f}-alpha={alpha:.2f}-seed={seed}"
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
        default=Path("benchmarks/reports/locality_ablation.md"),
    )
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    facts, topic_facts, queries = generate_dataset()
    print(
        f"Dataset: {len(facts)} facts, "
        f"{len({t.topic for t in topic_facts})} topics, "
        f"{len(queries)} queries.\n"
    )

    # Flat baseline
    print("=== Baseline: flat cosine (no graph) ===")
    flat_adapter = SomaAdapter(use_sbert=True, attach_soma=False)
    flat_adapter.name = "soma-sbert-flat"
    [flat_result] = run_retrieval_benchmark(
        systems=[flat_adapter],
        dataset=facts,
        queries=queries,
        k=args.k,
    )

    # locality x alpha x seed grid. Seeds make the locality-on vs -off
    # comparison paired: same rng init, only synap_locality differs.
    results_off: dict[tuple[float, int], object] = {}
    results_on: dict[tuple[float, int], object] = {}
    for seed in SEEDS:
        for alpha in ALPHAS:
            print(f"\n=== locality=0.0, alpha={alpha:.2f}, seed={seed} ===")
            results_off[(alpha, seed)] = _run(alpha, 0.0, seed, facts, queries, args.k)
        for alpha in ALPHAS:
            print(f"\n=== locality=0.5, alpha={alpha:.2f}, seed={seed} ===")
            results_on[(alpha, seed)] = _run(alpha, 0.5, seed, facts, queries, args.k)

    # Paired delta per alpha (averaged over seeds) + per-seed deltas
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    per_alpha_summary: list[tuple[float, float, float, float, list[tuple[int, float, float, float]]]] = []
    for alpha in ALPHAS:
        off_vals = [results_off[(alpha, s)].recall_at_k for s in SEEDS]  # type: ignore[attr-defined]
        on_vals = [results_on[(alpha, s)].recall_at_k for s in SEEDS]  # type: ignore[attr-defined]
        per_seed = [
            (
                s,
                results_off[(alpha, s)].recall_at_k,  # type: ignore[attr-defined]
                results_on[(alpha, s)].recall_at_k,  # type: ignore[attr-defined]
                results_on[(alpha, s)].recall_at_k - results_off[(alpha, s)].recall_at_k,  # type: ignore[attr-defined]
            )
            for s in SEEDS
        ]
        per_alpha_summary.append((alpha, _mean(off_vals), _mean(on_vals), _mean(on_vals) - _mean(off_vals), per_seed))

    lines = [
        "# Locality Ablation on Retrieval (paired seeds)",
        "",
        f"**Dataset:** {len(facts)} facts, {len(queries)} queries, k={args.k}.",
        f"**Seeds:** {SEEDS}",
        "",
        "**Question:** does the v0.5 synap_local finding (commits 2ba566b, "
        "28c5329) transfer to a semantic retrieval task?",
        "",
        "## Baseline (no graph re-rank)",
        "",
        format_results_table([flat_result], k=args.k),
        "",
        "## Paired comparison (mean across seeds)",
        "",
        "| alpha | locality=0.0 R@k | locality=0.5 R@k | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for alpha, off_mean, on_mean, d, _per_seed in per_alpha_summary:
        sign = "+" if d >= 0 else ""
        lines.append(f"| {alpha:.2f} | {off_mean:.3f} | {on_mean:.3f} | {sign}{d:.3f} |")

    lines += [
        "",
        "## Per-seed breakdown",
        "",
    ]
    for alpha, _off_mean, _on_mean, _d_mean, per_seed in per_alpha_summary:
        lines.append(f"### alpha = {alpha:.2f}")
        lines.append("")
        lines.append("| seed | locality=0.0 R@k | locality=0.5 R@k | delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for s, off_r, on_r, d in per_seed:
            sign = "+" if d >= 0 else ""
            lines.append(f"| {s} | {off_r:.3f} | {on_r:.3f} | {sign}{d:.3f} |")
        lines.append("")

    best_off_val = max(r.recall_at_k for r in results_off.values())  # type: ignore[attr-defined]
    best_on_val = max(r.recall_at_k for r in results_on.values())  # type: ignore[attr-defined]

    # Scorecard: count seeds where locality-on beats locality-off per alpha.
    lines += [
        "## Win count (locality=0.5 beats locality=0.0 per seed, threshold +0.01)",
        "",
        "| alpha | wins / total |",
        "| --- | ---: |",
    ]
    for alpha, _off_mean, _on_mean, _d_mean, per_seed in per_alpha_summary:
        wins = sum(1 for _, _o, _n, d in per_seed if d > 0.01)
        lines.append(f"| {alpha:.2f} | {wins} / {len(per_seed)} |")

    lines += [
        "",
        "## Headline",
        "",
        f"- Flat baseline Recall@{args.k}: **{flat_result.recall_at_k:.3f}**",
        f"- Best locality-OFF (any seed, any alpha): **{best_off_val:.3f}**",
        f"- Best locality-ON (any seed, any alpha):  **{best_on_val:.3f}**",
        "",
        "Interpretation: if locality-ON's mean-across-seeds beats locality-OFF's "
        "mean (paired, same rng init), the v0.5 locality principle transfers. "
        "Look especially at higher alpha values where the graph signal "
        "dominates — locality should cut the noise floor.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_locality_ablation.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
