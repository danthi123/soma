"""B3 orchestrator: SOMA CL ablations on Permuted-MNIST.

Runs four SOMA ablation modes and compares against B2 baseline numbers.

SOMA processes samples one-at-a-time through its graph (~66 samples/sec
on CUDA), so training on the full 60K MNIST per task is impractical.
The ``--train-samples`` flag (default 2000) subsamples each task's
training set to keep wall-clock reasonable (~30 min per ablation).
Test sets are always full-size for accurate evaluation.

Usage::

    python -m research.cl.run_b3                        # full 10-task run
    python -m research.cl.run_b3 --tasks 3 --epochs 1   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from research.cl import datasets, harness
from research.cl.soma_cl import make_soma_cl_components

REPORTS_DIR = Path("research/cl/reports")


def _subsample_loader(
    loader: DataLoader,
    max_samples: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Subsample a DataLoader to at most *max_samples*."""
    all_x = []
    all_y = []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    n = all_x.size(0)
    if n > max_samples:
        idx = torch.randperm(n)[:max_samples]
        all_x = all_x[idx]
        all_y = all_y[idx]

    return DataLoader(
        TensorDataset(all_x, all_y),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _subsample_tasks(
    tasks: list[tuple[int, DataLoader, DataLoader]],
    max_train_samples: int,
    max_test_samples: int,
    batch_size: int,
) -> list[tuple[int, DataLoader, DataLoader]]:
    """Subsample both train and test data in each task."""
    out = []
    for task_id, train_loader, test_loader in tasks:
        new_train = _subsample_loader(
            train_loader, max_train_samples, batch_size, shuffle=True
        )
        new_test = _subsample_loader(
            test_loader, max_test_samples, batch_size, shuffle=False
        )
        out.append((task_id, new_train, new_test))
    return out


def _print_matrix(A: np.ndarray, label: str) -> None:
    T = A.shape[0]
    print(f"\n{'=' * 60}")
    print(f"Accuracy matrix ({label})")
    print(f"{'=' * 60}")
    header = "       " + "".join(f"  t{j:<5d}" for j in range(T))
    print(header)
    for i in range(T):
        row = f"  t{i:<3d} " + "".join(f"  {A[i, j]:.4f}" for j in range(T))
        print(row)
    print()


def _print_results(result: dict, label: str) -> None:
    _print_matrix(result["accuracy_matrix"], label)
    print(f"ACC = {result['acc']:.4f}")
    print(f"BWT = {result['bwt']:.4f}")
    print(f"FWT = {result['fwt']:.4f}")
    print(f"Wall-clock: {result['wall_clock']:.1f}s")
    print()


def _run_benchmark(
    name: str,
    model_factory,
    tasks,
    n_epochs: int,
    device: torch.device,
    train_one_epoch=None,
    on_task_end=None,
) -> dict:
    print(f"\n>>> {name}")
    result = harness.run(
        model_factory=model_factory,
        tasks=tasks,
        n_epochs=n_epochs,
        device=device,
        verbose=True,
        train_one_epoch=train_one_epoch,
        on_task_end=on_task_end,
    )
    _print_results(result, name)
    return result


def _result_summary(result: dict) -> dict:
    """Extract JSON-serializable summary."""
    return {
        "acc": round(result["acc"], 4),
        "bwt": round(result["bwt"], 4),
        "fwt": round(result["fwt"], 4),
        "wall_clock_s": round(result["wall_clock"], 1),
    }


# ------------------------------------------------------------------
# SOMA ablation configs
# ------------------------------------------------------------------

ABLATIONS = [
    {
        "name": "soma-plastic",
        "frozen": False,
        "enable_consolidation": True,
        "disable_critical_periods": False,
        "head_replay": False,
    },
    {
        "name": "soma-frozen",
        "frozen": True,
        "enable_consolidation": False,
        "disable_critical_periods": False,
        "head_replay": False,
    },
    {
        "name": "soma-no-consolidation",
        "frozen": False,
        "enable_consolidation": False,
        "disable_critical_periods": False,
        "head_replay": False,
    },
    {
        "name": "soma-no-critical-periods",
        "frozen": False,
        "enable_consolidation": True,
        "disable_critical_periods": True,
        "head_replay": False,
    },
    {
        "name": "soma-head-replay",
        "frozen": True,
        "enable_consolidation": False,
        "disable_critical_periods": False,
        "head_replay": True,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="B3: SOMA CL ablations")
    parser.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Number of Permuted-MNIST tasks (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs per task (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=2000,
        help=(
            "Max training samples per task (default: 2000). "
            "SOMA processes ~66 samples/sec on CUDA; full 60K is "
            "impractical."
        ),
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        help=(
            "Max test samples per task for evaluation (default: 1000). "
            "SOMA eval also requires per-sample graph execution."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="SOMA seed (default: 42)",
    )
    parser.add_argument(
        "--integrators",
        type=int,
        default=8,
        help=(
            "Number of initial integrator nodes (default: 8). "
            "Associators are set to 2x this value."
        ),
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help=(
            "Run only a specific ablation (default: all). "
            "Choices: soma-plastic, soma-frozen, soma-no-consolidation, "
            "soma-no-critical-periods"
        ),
    )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0_total = time.perf_counter()

    # Load Permuted-MNIST tasks
    print(f"\n{'#' * 60}")
    print(
        f"# Permuted-MNIST ({args.tasks} tasks, "
        f"{args.epochs} epochs/task, "
        f"{args.train_samples} train / {args.test_samples} test per task)"
    )
    print(f"{'#' * 60}")

    mnist_tasks = datasets.permuted_mnist(
        n_tasks=args.tasks,
        batch_size=args.batch_size,
        seed=0,  # same permutation seed as B1/B2
    )

    # Subsample for SOMA's per-sample throughput
    mnist_tasks = _subsample_tasks(
        mnist_tasks,
        max_train_samples=args.train_samples,
        max_test_samples=args.test_samples,
        batch_size=args.batch_size,
    )

    all_results: dict = {}

    ablations = ABLATIONS
    if args.ablation:
        ablations = [a for a in ABLATIONS if a["name"] == args.ablation]
        if not ablations:
            parser.error(f"Unknown ablation {args.ablation!r}")

    for ablation in ablations:
        factory, train_fn, hook = make_soma_cl_components(
            dataset="mnist",
            frozen=ablation["frozen"],
            enable_consolidation=ablation["enable_consolidation"],
            disable_critical_periods=ablation["disable_critical_periods"],
            seed=args.seed,
            integrator_count=args.integrators,
            head_replay=ablation.get("head_replay", False),
        )
        result = _run_benchmark(
            f"{ablation['name']} (Permuted-MNIST)",
            factory,
            mnist_tasks,
            args.epochs,
            device,
            train_one_epoch=train_fn,
            on_task_end=hook,
        )
        all_results[ablation["name"]] = _result_summary(result)

    total_wall = time.perf_counter() - t0_total

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'=' * 72}")
    print("B3 SUMMARY TABLE (Permuted-MNIST)")
    print(f"{'=' * 72}")
    print(
        f"{'Method':<30s} {'ACC':>8s} {'BWT':>8s} "
        f"{'FWT':>8s} {'Time':>8s}"
    )
    print("-" * 72)

    for method, vals in all_results.items():
        print(
            f"{method:<30s} {vals['acc']:>8.4f} "
            f"{vals['bwt']:>8.4f} {vals['fwt']:>8.4f} "
            f"{vals['wall_clock_s']:>7.1f}s"
        )

    # B2 reference numbers for comparison
    print()
    print("B2 reference (full 60K train, same harness):")
    print(f"{'  Naive':<30s} {'0.70':>8s} {'-0.23':>8s}")
    print(f"{'  EWC (lam=1000)':<30s} {'0.72':>8s} {'-0.22':>8s}")
    print(f"{'  A-GEM (buf=256)':<30s} {'0.82':>8s} {'-0.10':>8s}")

    print(f"\nTotal wall-clock: {total_wall:.1f}s")

    # ================================================================
    # Save JSON sidecar
    # ================================================================
    all_results["meta"] = {
        "tasks": args.tasks,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples_per_task": args.train_samples,
        "test_samples_per_task": args.test_samples,
        "soma_seed": args.seed,
        "device": str(device),
        "total_wall_clock_s": round(total_wall, 1),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "b3_soma_cl.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
