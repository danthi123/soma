"""D1 orchestrator: run baselines on all benchmarks, emit results.

Usage:
    python -m research.associative.run_d1          # full run
    python -m research.associative.run_d1 --quick  # smoke test (reduced sweep)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.associative.baselines import (
    ClassicalHopfield,
    KNNRecall,
    ModernHopfield,
)
from research.associative.datasets import make_binary_patterns, make_mnist_recall
from research.associative.harness import evaluate_recall, evaluate_recall_nearest

REPORT_DIR = Path(__file__).parent / "reports"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Experiment 1: Random-binary capacity sweep
# ---------------------------------------------------------------------------

def run_capacity_sweep(
    dim: int = 50,
    n_max_ratio: float = 3.0,
    step: int = 1,
    mask_frac: float = 0.2,
    n_trials: int = 5,
    device: torch.device | None = None,
) -> list[dict]:
    """Sweep N from 1 to n_max_ratio*D, measure exact-recall rate."""
    device = device or _device()
    n_max = int(dim * n_max_ratio)
    results: list[dict] = []

    models = {
        "classical_hopfield": ClassicalHopfield(max_iter=200),
        "modern_hopfield_b1": ModernHopfield(beta=1.0, n_iter=10),
        "modern_hopfield_b5": ModernHopfield(beta=5.0, n_iter=10),
        "modern_hopfield_b10": ModernHopfield(beta=10.0, n_iter=10),
        "modern_hopfield_b50": ModernHopfield(beta=50.0, n_iter=10),
        "knn": KNNRecall(),
    }

    for n in range(1, n_max + 1, step):
        row: dict = {"n_patterns": n, "dim": dim, "ratio": n / dim}
        for name, model in models.items():
            # Average over multiple trials for stability
            exact_rates = []
            elem_accs = []
            for trial in range(n_trials):
                ds = make_binary_patterns(
                    n_patterns=n, dim=dim, mask_frac=mask_frac,
                    device=device, seed=trial * 10000 + n,
                )
                bipolar = "classical" in name
                m = evaluate_recall(model, ds.patterns, ds.probes, bipolar=bipolar)
                exact_rates.append(m.exact_match_rate)
                elem_accs.append(m.per_element_accuracy)
            row[f"{name}_exact"] = sum(exact_rates) / n_trials
            row[f"{name}_elem"] = sum(elem_accs) / n_trials
        results.append(row)
        # Progress
        if n % 10 == 0 or n == 1:
            print(f"  capacity sweep: N={n}/{n_max}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Experiment 2: MNIST denoising
# ---------------------------------------------------------------------------

def run_mnist_denoising(
    n_patterns: int = 50,
    device: torch.device | None = None,
) -> dict:
    """Run all baselines on MNIST denoising (Gaussian + occlusion)."""
    device = device or _device()
    results: dict = {}

    for noise_type in ("gaussian", "occlusion"):
        ds = make_mnist_recall(
            n_patterns=n_patterns,
            noise_type=noise_type,  # type: ignore[arg-type]
            noise_level=0.5,
            occlusion_frac=0.3,
            device=device,
            seed=42,
        )

        models = {
            "classical_hopfield": ClassicalHopfield(max_iter=100),
            "modern_hopfield_b5": ModernHopfield(beta=5.0, n_iter=10),
            "modern_hopfield_b10": ModernHopfield(beta=10.0, n_iter=10),
            "knn": KNNRecall(),
        }

        noise_results: dict = {}
        for name, model in models.items():
            m = evaluate_recall_nearest(
                model, ds.patterns, ds.labels, ds.probes,
            )
            noise_results[name] = m.to_dict()
            print(
                f"  MNIST {noise_type} | {name}: "
                f"MSE={m.mse:.4f}, class_acc={m.extra.get('class_accuracy', 0):.3f}",
                flush=True,
            )
        results[noise_type] = noise_results

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="D1 baselines")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for smoke test")
    args = parser.parse_args()

    device = _device()
    print(f"Device: {device}")
    t0 = time.time()

    # --- Capacity sweep ---
    print("\n=== Random-binary capacity sweep ===")
    if args.quick:
        capacity = run_capacity_sweep(
            dim=50, n_max_ratio=0.5, step=5, n_trials=2, device=device,
        )
    else:
        capacity = run_capacity_sweep(
            dim=50, n_max_ratio=3.0, step=1, n_trials=5, device=device,
        )

    # --- MNIST denoising ---
    print("\n=== MNIST denoising ===")
    n_mnist = 10 if args.quick else 50
    mnist = run_mnist_denoising(n_patterns=n_mnist, device=device)

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock: {elapsed:.1f}s")

    # --- Find capacity thresholds ---
    def find_threshold(data: list[dict], key: str, threshold: float = 0.5) -> float | None:
        """Find the N/D ratio where exact-recall drops below *threshold*."""
        for row in data:
            if row[key] < threshold:
                return row["ratio"]
        return None

    classical_thresh = find_threshold(capacity, "classical_hopfield_exact")
    modern_b10_thresh = find_threshold(capacity, "modern_hopfield_b10_exact")

    summary = {
        "capacity_sweep": capacity,
        "mnist_denoising": mnist,
        "thresholds": {
            "classical_hopfield_50pct": classical_thresh,
            "modern_hopfield_b10_50pct": modern_b10_thresh,
        },
        "wall_clock_s": elapsed,
        "device": str(device),
        "word_association": "deferred",
    }

    # --- Write JSON sidecar ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "d1_baselines.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON results written to {json_path}")

    # --- Write markdown report ---
    md_path = REPORT_DIR / "d1_baselines.md"
    _write_report(md_path, summary, capacity)
    print(f"Markdown report written to {md_path}")


def _write_report(path: Path, summary: dict, capacity: list[dict]) -> None:
    """Generate the D1 baselines markdown report."""
    lines: list[str] = []
    lines.append("# D1 Baselines Report: Associative / Hopfield-Style Recall")
    lines.append("")
    lines.append(f"**Device:** {summary['device']}")
    lines.append(f"**Wall-clock:** {summary['wall_clock_s']:.1f}s")
    lines.append(f"**Word-association benchmark:** {summary['word_association']}")
    lines.append("")

    # Capacity thresholds
    lines.append("## Capacity Thresholds (exact-recall drops below 50%)")
    lines.append("")
    thresh = summary["thresholds"]
    ct = thresh["classical_hopfield_50pct"]
    mt = thresh["modern_hopfield_b10_50pct"]
    lines.append(f"- **Classical Hopfield:** N/D = {ct if ct else 'never dropped'} "
                 f"(theoretical: 0.14)")
    lines.append(f"- **Modern Hopfield (beta=10):** N/D = {mt if mt else 'never dropped'}")
    lines.append("")

    # Capacity curve table (sampled)
    lines.append("## Capacity Curve (Random Binary, D=50)")
    lines.append("")
    lines.append("| N | N/D | Classical | Modern b=1 | Modern b=5 "
                 "| Modern b=10 | Modern b=50 | kNN |")
    lines.append("|---|-----|-----------|------------|------------|"
                 "-------------|-------------|-----|")
    for row in capacity:
        n = row["n_patterns"]
        # Show every 5th row for readability, plus first and last
        if n == 1 or n % 5 == 0 or n == capacity[-1]["n_patterns"]:
            lines.append(
                f"| {n} | {row['ratio']:.2f} "
                f"| {row['classical_hopfield_exact']:.2f} "
                f"| {row['modern_hopfield_b1_exact']:.2f} "
                f"| {row['modern_hopfield_b5_exact']:.2f} "
                f"| {row['modern_hopfield_b10_exact']:.2f} "
                f"| {row['modern_hopfield_b50_exact']:.2f} "
                f"| {row['knn_exact']:.2f} |"
            )
    lines.append("")

    # MNIST results
    lines.append("## MNIST Denoising")
    lines.append("")
    mnist = summary["mnist_denoising"]
    for noise_type, models in mnist.items():
        lines.append(f"### {noise_type.capitalize()} noise")
        lines.append("")
        lines.append("| Model | MSE | Per-pixel Acc | Class Acc |")
        lines.append("|-------|-----|---------------|-----------|")
        for name, m in models.items():
            lines.append(
                f"| {name} | {m['mse']:.4f} "
                f"| {m['per_element_accuracy']:.3f} "
                f"| {m.get('class_accuracy', 0):.3f} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Classical Hopfield uses bipolar (+/-1) patterns with sign activation.")
    lines.append("- Modern Hopfield uses continuous patterns with softmax attention.")
    lines.append("- kNN is 1-nearest-neighbour by Euclidean distance (lower bound).")
    lines.append("- MNIST patterns normalised to [-1, 1].")
    lines.append("- Word-association benchmark deferred to later in D1 or D2.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
