"""D3 orchestrator: output scaling + structural plasticity for SOMA attractor mode.

Usage:
    python -m research.associative.run_d3          # full run
    python -m research.associative.run_d3 --quick  # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.associative.datasets import make_binary_patterns
from soma.research.attractor_mode import (
    OutputScaler,
    ZScoreScaler,
    count_nodes_by_type,
    fit_output_scaler,
    fit_zscore_scaler,
    make_attractor_soma,
    measure_recall_accuracy,
    recall_pattern,
    store_pattern,
    trigger_neurogenesis,
)

REPORT_DIR = Path(__file__).parent / "reports"
DIM = 50


def _device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    # D2 found CPU faster for D=50
    return torch.device("cpu")


# ======================================================================
# Experiment 1: Output scaler comparison
# ======================================================================


def exp1_scaler_comparison(
    *,
    dim: int = DIM,
    n_patterns: int = 10,
    n_presentations: int = 3,
    n_iters: int = 50,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """Compare raw / z-score / learned scaler on sign-exact recall."""
    if device is None:
        device = torch.device("cpu")
    print("\n=== Experiment 1: Output Scaler Comparison ===")

    ds = make_binary_patterns(
        n_patterns=n_patterns, dim=dim, mask_frac=0.2, device=device, seed=seed
    )

    soma = make_attractor_soma(
        pattern_dim=dim,
        n_associators=8,
        n_integrators=4,
        hebbian_lr=0.01,
        base_lr=0.001,
        seed=seed,
        device=device,
    )

    # Store
    for i in range(n_patterns):
        store_pattern(soma, ds.patterns[i], n_presentations=n_presentations)

    # Raw recall (no scaler)
    raw = measure_recall_accuracy(soma, ds.patterns, ds.probes, n_iters=n_iters)
    print(
        f"  Raw:     exact={raw['exact_rate']:.2f}  "
        f"nearest={raw['nearest_rate']:.2f}  bits={raw['per_bit_acc']:.3f}"
    )

    # Diagnose magnitude: show mean abs output vs mean abs pattern
    outputs_raw = []
    for i in range(n_patterns):
        rr = recall_pattern(soma, ds.patterns[i], max_iters=n_iters)
        outputs_raw.append(rr.final_output.detach())
    raw_stack = torch.stack(outputs_raw)
    mean_abs_out = raw_stack.abs().mean().item()
    mean_abs_pat = ds.patterns.abs().mean().item()
    print(f"  Magnitude: mean|output|={mean_abs_out:.4f}, mean|pattern|={mean_abs_pat:.4f}")

    # Z-score scaler
    zscaler = fit_zscore_scaler(soma, ds.patterns, n_iters=n_iters)
    zscore = measure_recall_accuracy(soma, ds.patterns, ds.probes, n_iters=n_iters, scaler=zscaler)
    print(
        f"  Z-score: exact={zscore['exact_rate']:.2f}  "
        f"nearest={zscore['nearest_rate']:.2f}  bits={zscore['per_bit_acc']:.3f}"
    )

    # Learned scaler
    lscaler = fit_output_scaler(soma, ds.patterns, n_iters=n_iters, lr=0.01, train_steps=200)
    learned = measure_recall_accuracy(soma, ds.patterns, ds.probes, n_iters=n_iters, scaler=lscaler)
    print(
        f"  Learned: exact={learned['exact_rate']:.2f}  "
        f"nearest={learned['nearest_rate']:.2f}  "
        f"bits={learned['per_bit_acc']:.3f}"
    )

    return {
        "n_patterns": n_patterns,
        "dim": dim,
        "mean_abs_output": mean_abs_out,
        "mean_abs_pattern": mean_abs_pat,
        "raw": raw,
        "zscore": zscore,
        "learned": learned,
    }


# ======================================================================
# Experiment 2: Capacity sweep with scaler
# ======================================================================


def exp2_capacity_with_scaler(
    *,
    dim: int = DIM,
    n_values: list[int] | None = None,
    n_presentations: int = 3,
    n_iters: int = 50,
    seed: int = 42,
    device: torch.device | None = None,
) -> list[dict]:
    """Sweep N with z-score + learned scaler; measure sign-exact recall."""
    if device is None:
        device = torch.device("cpu")
    print("\n=== Experiment 2: Capacity Sweep with Scaler ===")

    if n_values is None:
        n_values = [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150]

    results: list[dict] = []
    for n in n_values:
        ds = make_binary_patterns(n_patterns=n, dim=dim, mask_frac=0.2, device=device, seed=seed)
        soma = make_attractor_soma(
            pattern_dim=dim,
            n_associators=8,
            n_integrators=4,
            hebbian_lr=0.01,
            base_lr=0.001,
            seed=seed,
            device=device,
        )
        for i in range(n):
            store_pattern(soma, ds.patterns[i], n_presentations=n_presentations)

        # Raw
        raw = measure_recall_accuracy(soma, ds.patterns, ds.probes, n_iters=n_iters)

        # Z-score
        zscaler = fit_zscore_scaler(soma, ds.patterns, n_iters=n_iters)
        zscore = measure_recall_accuracy(
            soma, ds.patterns, ds.probes, n_iters=n_iters, scaler=zscaler
        )

        # Learned
        lscaler = fit_output_scaler(soma, ds.patterns, n_iters=n_iters, lr=0.01, train_steps=200)
        learned = measure_recall_accuracy(
            soma, ds.patterns, ds.probes, n_iters=n_iters, scaler=lscaler
        )

        row = {
            "n_patterns": n,
            "ratio": n / dim,
            "raw_exact": raw["exact_rate"],
            "raw_nearest": raw["nearest_rate"],
            "zscore_exact": zscore["exact_rate"],
            "zscore_nearest": zscore["nearest_rate"],
            "learned_exact": learned["exact_rate"],
            "learned_nearest": learned["nearest_rate"],
            "raw_bits": raw["per_bit_acc"],
            "zscore_bits": zscore["per_bit_acc"],
            "learned_bits": learned["per_bit_acc"],
        }
        results.append(row)
        print(
            f"  N={n:3d} (N/D={n / dim:.2f}): "
            f"raw_exact={raw['exact_rate']:.2f} "
            f"zscore_exact={zscore['exact_rate']:.2f} "
            f"learned_exact={learned['exact_rate']:.2f} "
            f"nearest={raw['nearest_rate']:.2f}",
            flush=True,
        )

    return results


# ======================================================================
# Experiment 3: Structural plasticity — growing capacity
# ======================================================================


def _store_and_measure(
    soma: object,
    patterns: torch.Tensor,
    probes: torch.Tensor,
    *,
    n_iters: int = 50,
    scaler_cls: str = "zscore",
) -> dict[str, float]:
    """Store patterns, fit scaler, measure accuracy.

    Returns exact_rate, nearest_rate, per_bit_acc.
    """
    from soma.system import SOMA as SOMAType

    assert isinstance(soma, SOMAType)

    scaler: OutputScaler | ZScoreScaler | None = None
    if scaler_cls == "zscore":
        scaler = fit_zscore_scaler(soma, patterns, n_iters=n_iters)
    elif scaler_cls == "learned":
        scaler = fit_output_scaler(soma, patterns, n_iters=n_iters, lr=0.01, train_steps=100)

    return measure_recall_accuracy(soma, patterns, probes, n_iters=n_iters, scaler=scaler)


def exp3_structural_plasticity(
    *,
    dim: int = DIM,
    max_patterns: int = 60,
    start_integrators: int = 4,
    start_associators: int = 8,
    n_presentations: int = 3,
    n_iters: int = 50,
    recall_threshold: float = 0.80,
    growth_nodes: int = 2,
    growth_check_every: int = 1,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """Three-way ablation: SOMA-fixed vs grow-on-saturation vs grow-random.

    All three start with the same initial topology.
    Patterns are added one at a time. After each pattern:
    - Measure recall on ALL stored patterns so far.
    - SOMA-grow-on-saturation: if recall < threshold, add nodes.
    - SOMA-grow-random: add nodes every `growth_check_every` patterns regardless.
    - SOMA-fixed: never add nodes.
    """
    if device is None:
        device = torch.device("cpu")
    print("\n=== Experiment 3: Structural Plasticity ===")

    conditions = ["fixed", "grow_on_saturation", "grow_random"]
    all_results: dict[str, list[dict]] = {c: [] for c in conditions}

    for cond in conditions:
        print(f"\n  --- {cond} ---")
        torch.manual_seed(seed)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 1)

        soma = make_attractor_soma(
            pattern_dim=dim,
            n_associators=start_associators,
            n_integrators=start_integrators,
            hebbian_lr=0.01,
            base_lr=0.001,
            seed=seed,
            device=device,
        )

        # Generate all patterns up front
        ds = make_binary_patterns(
            n_patterns=max_patterns, dim=dim, mask_frac=0.2, device=device, seed=seed
        )

        total_growth_events = 0
        trajectory: list[dict] = []

        for p_idx in range(max_patterns):
            # Store new pattern
            store_pattern(soma, ds.patterns[p_idx], n_presentations=n_presentations)

            # Measure recall on all stored patterns so far
            stored = ds.patterns[: p_idx + 1]
            probes = ds.probes[: p_idx + 1]
            metrics = _store_and_measure(soma, stored, probes, n_iters=n_iters, scaler_cls="zscore")

            node_counts = count_nodes_by_type(soma)
            total_nodes = soma.graph.num_nodes
            total_edges = soma.graph.num_edges

            grew = False

            if cond == "grow_on_saturation":
                if metrics["exact_rate"] < recall_threshold and (p_idx + 1) >= 3:
                    new_ids = trigger_neurogenesis(soma, num_new_nodes=growth_nodes, rng=gen)
                    if new_ids:
                        grew = True
                        total_growth_events += 1

            elif (
                cond == "grow_random" and (p_idx + 1) % growth_check_every == 0 and (p_idx + 1) >= 3
            ):
                new_ids = trigger_neurogenesis(soma, num_new_nodes=growth_nodes, rng=gen)
                if new_ids:
                    grew = True
                    total_growth_events += 1

            step_info = {
                "pattern_idx": p_idx,
                "n_stored": p_idx + 1,
                "exact_rate": metrics["exact_rate"],
                "nearest_rate": metrics["nearest_rate"],
                "per_bit_acc": metrics["per_bit_acc"],
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "grew": grew,
                "node_counts": node_counts,
            }
            trajectory.append(step_info)

            if (p_idx + 1) % 10 == 0 or p_idx == 0:
                print(
                    f"    N={p_idx + 1:3d}: exact={metrics['exact_rate']:.2f} "
                    f"nearest={metrics['nearest_rate']:.2f} "
                    f"nodes={total_nodes} edges={total_edges} "
                    f"{'GREW' if grew else ''}",
                    flush=True,
                )

        all_results[cond] = trajectory

    return {
        "dim": dim,
        "max_patterns": max_patterns,
        "start_topology": {
            "associators": start_associators,
            "integrators": start_integrators,
        },
        "recall_threshold": recall_threshold,
        "growth_nodes_per_event": growth_nodes,
        "conditions": all_results,
    }


# ======================================================================
# Report generation
# ======================================================================


def _write_report(path: Path, exp1: dict, exp2: list[dict], exp3: dict) -> None:
    lines: list[str] = []
    lines.append("# D3 Report: Capacity Growth via Structural Plasticity")
    lines.append("")
    lines.append("**Date:** 2026-04-16")
    lines.append(f"**Dimension:** D={DIM}")
    lines.append("")

    # Experiment 1
    lines.append("## 1. Output Scaler Comparison (N=10, D=50)")
    lines.append("")
    lines.append(
        f"Mean |output| = {exp1['mean_abs_output']:.4f}, "
        f"mean |pattern| = {exp1['mean_abs_pattern']:.4f}"
    )
    lines.append("")
    lines.append("| Method | Sign Exact | Nearest | Per-bit Acc |")
    lines.append("|--------|-----------|---------|-------------|")
    for label, key in [("Raw", "raw"), ("Z-score", "zscore"), ("Learned", "learned")]:
        d = exp1[key]
        lines.append(
            f"| {label} | {d['exact_rate']:.2f} | "
            f"{d['nearest_rate']:.2f} | {d['per_bit_acc']:.3f} |"
        )
    lines.append("")

    # Experiment 2
    lines.append("## 2. Capacity Sweep with Scaler")
    lines.append("")
    lines.append("| N | N/D | Raw Exact | Z-score Exact | Learned Exact | Nearest |")
    lines.append("|---|-----|-----------|---------------|---------------|---------|")
    for row in exp2:
        lines.append(
            f"| {row['n_patterns']} | {row['ratio']:.2f} "
            f"| {row['raw_exact']:.2f} | {row['zscore_exact']:.2f} "
            f"| {row['learned_exact']:.2f} | {row['raw_nearest']:.2f} |"
        )
    lines.append("")

    # Find capacity cliff
    zscore_50 = None
    learned_50 = None
    nearest_50 = None
    for row in exp2:
        if zscore_50 is None and row["zscore_exact"] < 0.50:
            zscore_50 = row["ratio"]
        if learned_50 is None and row["learned_exact"] < 0.50:
            learned_50 = row["ratio"]
        if nearest_50 is None and row["raw_nearest"] < 0.50:
            nearest_50 = row["ratio"]

    lines.append("### Capacity Cliff (N/D where recall drops below 50%)")
    lines.append("")
    lines.append(f"- **Z-score sign-exact:** N/D = {zscore_50 if zscore_50 else '>max tested'}")
    lines.append(f"- **Learned sign-exact:** N/D = {learned_50 if learned_50 else '>max tested'}")
    lines.append(f"- **Nearest-pattern:** N/D = {nearest_50 if nearest_50 else '>max tested'}")
    lines.append("- Classical Hopfield (D1): N/D ~ 0.22")
    lines.append("- Modern Hopfield (D1): N/D > 3.00 (perfect throughout)")
    lines.append("")

    # Experiment 3
    lines.append("## 3. Structural Plasticity Ablation")
    lines.append("")
    for cond in ["fixed", "grow_on_saturation", "grow_random"]:
        traj = exp3["conditions"][cond]
        lines.append(f"### {cond}")
        lines.append("")
        if traj:
            final = traj[-1]
            lines.append(
                f"- Final: N={final['n_stored']} patterns, "
                f"exact={final['exact_rate']:.2f}, "
                f"nearest={final['nearest_rate']:.2f}, "
                f"nodes={final['total_nodes']}, edges={final['total_edges']}"
            )

            # Find max N with exact >= threshold
            max_n_good = 0
            for step in traj:
                if step["exact_rate"] >= exp3["recall_threshold"]:
                    max_n_good = step["n_stored"]
            lines.append(f"- Max N with exact >= {exp3['recall_threshold']:.0%}: {max_n_good}")

            # Show trajectory at key points
            lines.append("")
            lines.append("| N | Exact | Nearest | Nodes | Edges | Grew |")
            lines.append("|---|-------|---------|-------|-------|------|")
            show_idxs = [0, 2, 4, 9, 14, 19, 24, 29, 39, 49, 59]
            for idx in show_idxs:
                if idx < len(traj):
                    s = traj[idx]
                    lines.append(
                        f"| {s['n_stored']} | {s['exact_rate']:.2f} "
                        f"| {s['nearest_rate']:.2f} | {s['total_nodes']} "
                        f"| {s['total_edges']} | {'Yes' if s['grew'] else ''} |"
                    )
        lines.append("")

    # Conclusions
    lines.append("## Key Findings")
    lines.append("")
    lines.append("See JSON sidecar for full numerical data.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="D3: capacity growth")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep")
    parser.add_argument("--device", type=str, default=None, help="Force device")
    args = parser.parse_args()

    device = _device(args.device)
    print(f"Device: {device}")
    t0 = time.time()

    if args.quick:
        exp1 = exp1_scaler_comparison(n_patterns=5, device=device)
        exp2 = exp2_capacity_with_scaler(n_values=[1, 3, 5, 10, 15, 20], device=device)
        exp3 = exp3_structural_plasticity(max_patterns=15, device=device)
    else:
        exp1 = exp1_scaler_comparison(device=device)
        exp2 = exp2_capacity_with_scaler(device=device)
        exp3 = exp3_structural_plasticity(device=device)

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

    # Write outputs
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "exp1_scaler_comparison": exp1,
        "exp2_capacity_sweep": exp2,
        "exp3_structural_plasticity": exp3,
        "wall_clock_s": elapsed,
        "device": str(device),
    }

    json_path = REPORT_DIR / "d3_capacity_growth.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON results written to {json_path}")

    md_path = REPORT_DIR / "d3_capacity_growth.md"
    _write_report(md_path, exp1, exp2, exp3)
    print(f"Markdown report written to {md_path}")


if __name__ == "__main__":
    main()
