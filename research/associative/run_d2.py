"""D2 orchestrator: SOMA attractor mode on D1 benchmarks.

Usage:
    python -m research.associative.run_d2          # full run
    python -m research.associative.run_d2 --quick  # smoke test
    python -m research.associative.run_d2 --device cuda  # force CUDA
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.associative.soma_attractor import (
    analyse_convergence,
    homeostasis_ablation,
    run_binary_capacity_sweep,
    run_binary_recall,
    run_mnist_recall,
)

REPORT_DIR = Path(__file__).parent / "reports"


def _device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="D2: SOMA attractor mode")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep")
    parser.add_argument("--device", type=str, default=None, help="Force device")
    args = parser.parse_args()

    device = _device(args.device)
    print(f"Device: {device}")
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Convergence analysis
    # ------------------------------------------------------------------
    print("\n=== Convergence Analysis ===")
    if args.quick:
        conv = analyse_convergence(
            dim=50, n_patterns=3, max_iters=10, device=device
        )
    else:
        conv = analyse_convergence(
            dim=50, n_patterns=5, max_iters=100, device=device
        )
    n_conv = sum(1 for t in conv["trajectories"] if t["converged"])
    n_total = len(conv["trajectories"])
    print(f"  Converged: {n_conv}/{n_total}")
    for t in conv["trajectories"]:
        tag = f"iter={t['convergence_iter']}" if t["converged"] else "NOT converged"
        print(
            f"    Pattern {t['pattern_idx']}: {tag}, "
            f"final_norm={t['final_norm']:.4f}"
        )

    # ------------------------------------------------------------------
    # 2. Gate criterion: N=10 exact-recall
    # ------------------------------------------------------------------
    print("\n=== D2 Gate Criterion (N=10, D=50) ===")
    gate_k_values = [1, 5, 10, 50, 100] if not args.quick else [1, 5, 10]
    gate_results: dict[int, dict[str, float]] = {}
    for k in gate_k_values:
        r = run_binary_recall(
            n_patterns=10,
            dim=50,
            n_iters=k,
            seed=42,
            device=device,
        )
        gate_results[k] = {
            "exact": r.exact_match_rate,
            "nearest": r.nearest_pattern_accuracy,
            "bits": r.per_bit_accuracy,
            "conv_rate": r.convergence.convergence_rate,
        }
        print(
            f"  K={k:3d}: exact={r.exact_match_rate:.2f}, "
            f"nearest={r.nearest_pattern_accuracy:.2f}, "
            f"bits={r.per_bit_accuracy:.3f}, "
            f"conv_rate={r.convergence.convergence_rate:.2f}"
        )
    best_k = max(gate_results, key=lambda k: gate_results[k]["exact"])
    best_exact = gate_results[best_k]["exact"]
    best_nearest_k = max(gate_results, key=lambda k: gate_results[k]["nearest"])
    best_nearest = gate_results[best_nearest_k]["nearest"]
    gate_pass = best_exact >= 0.50
    print(
        f"\n  GATE (sign-based): best exact-recall = {best_exact:.2f} at K={best_k} "
        f"-> {'PASS' if gate_pass else 'FAIL'}"
    )
    print(
        f"  INFO (nearest-pattern): best = {best_nearest:.2f} at K={best_nearest_k}"
    )

    # ------------------------------------------------------------------
    # 3. Capacity sweep
    # ------------------------------------------------------------------
    print("\n=== Capacity Sweep (D=50) ===")
    if args.quick:
        capacity = run_binary_capacity_sweep(
            dim=50,
            n_values=[1, 3, 5, 10],
            k_values=[1, 10],
            device=device,
        )
    else:
        capacity = run_binary_capacity_sweep(
            dim=50,
            n_values=[1, 2, 3, 5, 7, 10, 12, 15, 20],
            k_values=[1, 5, 10, 50, 100],
            device=device,
        )

    # ------------------------------------------------------------------
    # 4. Homeostasis ablation
    # ------------------------------------------------------------------
    print("\n=== Homeostasis Ablation (N=10, D=50) ===")
    homeo = homeostasis_ablation(n_patterns=10, dim=50, n_iters=50, device=device)
    for label, m in homeo.items():
        print(
            f"  {label}: exact={m['exact_match_rate']:.2f}, "
            f"nearest={m['nearest_pattern_accuracy']:.2f}, "
            f"bits={m['per_bit_accuracy']:.3f}"
        )

    # ------------------------------------------------------------------
    # 5. MNIST denoising (skip in --quick to save time)
    # ------------------------------------------------------------------
    mnist_results: dict = {}
    if not args.quick:
        print("\n=== MNIST Denoising (N=50) ===")
        for noise_type in ("gaussian", "occlusion"):
            print(f"\n  --- {noise_type} ---")
            mr = run_mnist_recall(
                n_patterns=50,
                noise_type=noise_type,
                n_iters=50,
                device=device,
            )
            mnist_results[noise_type] = {
                "mse": mr.mse,
                "class_accuracy": mr.class_accuracy,
                "per_pixel_accuracy": mr.per_pixel_accuracy,
                "convergence_rate": mr.convergence.convergence_rate,
                "mean_convergence_iter": mr.convergence.mean_convergence_iter,
            }
            print(
                f"  MSE={mr.mse:.4f}, class_acc={mr.class_accuracy:.3f}, "
                f"pixel_acc={mr.per_pixel_accuracy:.3f}, "
                f"conv_rate={mr.convergence.convergence_rate:.2f}"
            )
    else:
        print("\n=== MNIST Denoising (skipped in --quick) ===")

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Assemble JSON sidecar
    # ------------------------------------------------------------------
    # Simplify convergence trajectories for JSON
    conv_json = {
        "n_patterns": conv["n_patterns"],
        "dim": conv["dim"],
        "max_iters": conv["max_iters"],
        "trajectories": [
            {
                "pattern_idx": t["pattern_idx"],
                "converged": t["converged"],
                "convergence_iter": t["convergence_iter"],
                "final_norm": t["final_norm"],
                # Only store a few sample norms to keep JSON small
                "sample_norms": t["norms"][::10] if len(t["norms"]) > 10 else t["norms"],
                "sample_diffs": (
                    t["inter_step_diffs"][::10]
                    if len(t["inter_step_diffs"]) > 10
                    else t["inter_step_diffs"]
                ),
            }
            for t in conv["trajectories"]
        ],
    }

    summary = {
        "convergence_analysis": conv_json,
        "gate_criterion": {
            "n_patterns": 10,
            "dim": 50,
            "results_by_k": {str(k): v for k, v in gate_results.items()},
            "best_k_exact": best_k,
            "best_exact_recall": best_exact,
            "best_k_nearest": best_nearest_k,
            "best_nearest_recall": best_nearest,
            "gate_pass": gate_pass,
        },
        "capacity_sweep": capacity,
        "homeostasis_ablation": homeo,
        "mnist_denoising": mnist_results,
        "wall_clock_s": elapsed,
        "device": str(device),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "d2_soma_attractor.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON results written to {json_path}")

    # ------------------------------------------------------------------
    # Write markdown report
    # ------------------------------------------------------------------
    md_path = REPORT_DIR / "d2_soma_attractor.md"
    _write_report(md_path, summary, capacity, gate_results)
    print(f"Markdown report written to {md_path}")


def _write_report(
    path: Path,
    summary: dict,
    capacity: list[dict],
    gate_results: dict[int, dict[str, float]],
) -> None:
    lines: list[str] = []
    lines.append("# D2 Report: SOMA in Attractor Mode")
    lines.append("")
    lines.append(f"**Device:** {summary['device']}")
    lines.append(f"**Wall-clock:** {summary['wall_clock_s']:.1f}s")
    lines.append("")

    # Gate criterion
    gate = summary["gate_criterion"]
    status = "PASS" if gate["gate_pass"] else "FAIL"
    lines.append(f"## D2 Gate: {status}")
    lines.append("")
    lines.append(
        f"Best sign-based exact-recall at N=10, D=50: "
        f"**{gate['best_exact_recall']:.2f}** at K={gate['best_k_exact']}."
    )
    lines.append(
        f"Best nearest-pattern recall: **{gate['best_nearest_recall']:.2f}** "
        f"at K={gate['best_k_nearest']}."
    )
    lines.append("Threshold: >=0.50 sign-based exact-recall for PASS.")
    lines.append("")
    lines.append("| K | Sign Exact | Nearest Pattern | Per-bit Acc | Conv Rate |")
    lines.append("|---|-----------|-----------------|-------------|-----------|")
    for k, v in sorted(gate_results.items()):
        lines.append(
            f"| {k} | {v['exact']:.2f} | {v['nearest']:.2f} "
            f"| {v['bits']:.3f} | {v['conv_rate']:.2f} |"
        )
    lines.append("")

    # Convergence analysis
    lines.append("## Convergence Analysis")
    lines.append("")
    conv = summary["convergence_analysis"]
    n_conv = sum(1 for t in conv["trajectories"] if t["converged"])
    lines.append(
        f"Tested {conv['n_patterns']} patterns, D={conv['dim']}, "
        f"max_iters={conv['max_iters']}."
    )
    lines.append(f"Converged: {n_conv}/{conv['n_patterns']}.")
    lines.append("")
    lines.append("| Pattern | Converged | Convergence Iter | Final Norm |")
    lines.append("|---------|-----------|------------------|------------|")
    for t in conv["trajectories"]:
        ci = t["convergence_iter"] if t["convergence_iter"] is not None else "-"
        lines.append(
            f"| {t['pattern_idx']} | {'Yes' if t['converged'] else 'No'} "
            f"| {ci} | {t['final_norm']:.4f} |"
        )
    lines.append("")

    # Capacity sweep
    lines.append("## Capacity Sweep (Random Binary, D=50)")
    lines.append("")
    k_cols = sorted(
        {
            int(key.split("_")[0][1:])
            for row in capacity
            for key in row
            if key.startswith("k") and "_exact" in key
        }
    )
    # Sign-based exact
    header = "| N | N/D | " + " | ".join(f"K={k} exact" for k in k_cols) + " |"
    sep = "|---|-----| " + " | ".join("---" for _ in k_cols) + " |"
    lines.append("### Sign-based exact recall")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for row in capacity:
        cells = [f"{row.get(f'k{k}_exact', 0):.2f}" for k in k_cols]
        lines.append(
            f"| {row['n_patterns']} | {row['ratio']:.2f} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")

    # Nearest-pattern accuracy
    lines.append("### Nearest-pattern recall (cosine)")
    lines.append("")
    header2 = "| N | N/D | " + " | ".join(f"K={k} nearest" for k in k_cols) + " |"
    lines.append(header2)
    lines.append(sep)
    for row in capacity:
        cells = [f"{row.get(f'k{k}_nearest', 0):.2f}" for k in k_cols]
        lines.append(
            f"| {row['n_patterns']} | {row['ratio']:.2f} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")

    # Homeostasis ablation
    lines.append("## Homeostasis Ablation")
    lines.append("")
    homeo = summary["homeostasis_ablation"]
    lines.append("| Mode | Sign Exact | Nearest Pattern | Per-bit Acc |")
    lines.append("|------|-----------|-----------------|------------|")
    for label, m in homeo.items():
        lines.append(
            f"| {label} | {m['exact_match_rate']:.2f} "
            f"| {m.get('nearest_pattern_accuracy', 0):.2f} "
            f"| {m['per_bit_accuracy']:.3f} |"
        )
    lines.append("")

    # MNIST
    mnist = summary.get("mnist_denoising", {})
    if mnist:
        lines.append("## MNIST Denoising")
        lines.append("")
        lines.append("| Noise | MSE | Class Acc | Pixel Acc | Conv Rate |")
        lines.append("|-------|-----|-----------|-----------|-----------|")
        for noise_type, m in mnist.items():
            lines.append(
                f"| {noise_type} | {m['mse']:.4f} | {m['class_accuracy']:.3f} "
                f"| {m['per_pixel_accuracy']:.3f} | {m['convergence_rate']:.2f} |"
            )
        lines.append("")
    else:
        lines.append("## MNIST Denoising")
        lines.append("")
        lines.append("Skipped (--quick mode).")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        "1. **Does SOMA converge?** See convergence analysis above. "
        "Convergence is measured as consecutive outputs within tolerance 1e-5."
    )
    lines.append(
        "2. **Does homeostasis fight attractor formation?** See ablation results."
    )
    lines.append(
        "3. **What is the capacity at fixed node count?** See capacity sweep."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
