"""B2 orchestrator: run EWC + A-GEM on Permuted-MNIST and Split-CIFAR-10.

Compares against the naive (no-defense) baseline from B1.

Usage::

    python -m research.cl.run_b2                    # full run
    python -m research.cl.run_b2 --tasks 3 --epochs 2  # quick smoke test
    python -m research.cl.run_b2 --sweep-ewc        # sweep EWC lambda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research.cl import datasets, harness
from research.cl.baselines.agem import make_agem_components
from research.cl.baselines.ewc import make_ewc_components
from research.cl.baselines.naive import cifar_factory, mnist_factory

REPORTS_DIR = Path("research/cl/reports")


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
    """Extract JSON-serializable summary from a harness result."""
    return {
        "acc": round(result["acc"], 4),
        "bwt": round(result["bwt"], 4),
        "fwt": round(result["fwt"], 4),
        "wall_clock_s": round(result["wall_clock"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B2: EWC + A-GEM benchmarks")
    parser.add_argument(
        "--tasks", type=int, default=10,
        help="Number of Permuted-MNIST tasks (default: 10)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Epochs per task (default: 5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Permutation seed (default: 0)",
    )
    parser.add_argument(
        "--ewc-lambda", type=float, default=400.0,
        help="EWC regularization strength (default: 400)",
    )
    parser.add_argument(
        "--agem-buffer", type=int, default=256,
        help="A-GEM buffer size per task (default: 256)",
    )
    parser.add_argument(
        "--sweep-ewc", action="store_true",
        help="Sweep EWC lambda over {100, 400, 1000}",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0_total = time.perf_counter()
    all_results: dict = {}

    # ================================================================
    # Permuted-MNIST
    # ================================================================
    print(f"\n{'#' * 60}")
    print(f"# Permuted-MNIST ({args.tasks} tasks, {args.epochs} epochs/task)")
    print(f"{'#' * 60}")

    mnist_tasks = datasets.permuted_mnist(
        n_tasks=args.tasks,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # --- Naive ---
    naive_mnist = _run_benchmark(
        "Naive (Permuted-MNIST)", mnist_factory, mnist_tasks,
        args.epochs, device,
    )

    # --- EWC ---
    ewc_train, ewc_hook = make_ewc_components(lam=args.ewc_lambda)
    ewc_mnist = _run_benchmark(
        f"EWC lam={args.ewc_lambda} (Permuted-MNIST)",
        mnist_factory, mnist_tasks, args.epochs, device,
        train_one_epoch=ewc_train, on_task_end=ewc_hook,
    )

    # --- A-GEM ---
    agem_train, agem_hook = make_agem_components(
        buffer_per_task=args.agem_buffer,
    )
    agem_mnist = _run_benchmark(
        f"A-GEM buf={args.agem_buffer} (Permuted-MNIST)",
        mnist_factory, mnist_tasks, args.epochs, device,
        train_one_epoch=agem_train, on_task_end=agem_hook,
    )

    all_results["permuted_mnist"] = {
        "naive": _result_summary(naive_mnist),
        "ewc": _result_summary(ewc_mnist),
        "agem": _result_summary(agem_mnist),
    }

    # ================================================================
    # Split-CIFAR-10
    # ================================================================
    print(f"\n{'#' * 60}")
    print("# Split-CIFAR-10 (5 tasks, 2 classes each)")
    print(f"{'#' * 60}")

    cifar_tasks = datasets.split_cifar10(batch_size=args.batch_size)

    # --- Naive ---
    naive_cifar = _run_benchmark(
        "Naive (Split-CIFAR-10)", cifar_factory, cifar_tasks,
        args.epochs, device,
    )

    # --- EWC ---
    ewc_train_c, ewc_hook_c = make_ewc_components(lam=args.ewc_lambda)
    ewc_cifar = _run_benchmark(
        f"EWC lam={args.ewc_lambda} (Split-CIFAR-10)",
        cifar_factory, cifar_tasks, args.epochs, device,
        train_one_epoch=ewc_train_c, on_task_end=ewc_hook_c,
    )

    # --- A-GEM ---
    agem_train_c, agem_hook_c = make_agem_components(
        buffer_per_task=args.agem_buffer,
    )
    agem_cifar = _run_benchmark(
        f"A-GEM buf={args.agem_buffer} (Split-CIFAR-10)",
        cifar_factory, cifar_tasks, args.epochs, device,
        train_one_epoch=agem_train_c, on_task_end=agem_hook_c,
    )

    all_results["split_cifar10"] = {
        "naive": _result_summary(naive_cifar),
        "ewc": _result_summary(ewc_cifar),
        "agem": _result_summary(agem_cifar),
    }

    # ================================================================
    # Optional: EWC lambda sweep
    # ================================================================
    ewc_sweep_results: dict = {}
    if args.sweep_ewc:
        print(f"\n{'#' * 60}")
        print("# EWC Lambda Sweep (Permuted-MNIST)")
        print(f"{'#' * 60}")

        for lam in [100.0, 400.0, 1000.0]:
            ewc_t, ewc_h = make_ewc_components(lam=lam)
            r = _run_benchmark(
                f"EWC lam={lam} (Permuted-MNIST)",
                mnist_factory, mnist_tasks, args.epochs, device,
                train_one_epoch=ewc_t, on_task_end=ewc_h,
            )
            ewc_sweep_results[str(int(lam))] = _result_summary(r)

        all_results["ewc_lambda_sweep"] = ewc_sweep_results

    total_wall = time.perf_counter() - t0_total

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'=' * 72}")
    print("SUMMARY TABLE")
    print(f"{'=' * 72}")
    print(f"{'Method':<30s} {'ACC':>8s} {'BWT':>8s} {'FWT':>8s}")
    print("-" * 72)

    for ds_name, ds_results in all_results.items():
        if ds_name == "ewc_lambda_sweep":
            continue
        print(f"  {ds_name}")
        for method, vals in ds_results.items():
            label = f"    {method}"
            print(
                f"{label:<30s} {vals['acc']:>8.4f} "
                f"{vals['bwt']:>8.4f} {vals['fwt']:>8.4f}"
            )
    print(f"\nTotal wall-clock: {total_wall:.1f}s")

    if ewc_sweep_results:
        print(f"\n{'=' * 72}")
        print("EWC LAMBDA SWEEP (Permuted-MNIST)")
        print(f"{'=' * 72}")
        print(f"{'Lambda':<10s} {'ACC':>8s} {'BWT':>8s} {'FWT':>8s}")
        print("-" * 40)
        for lam_str, vals in ewc_sweep_results.items():
            print(
                f"{lam_str:<10s} {vals['acc']:>8.4f} "
                f"{vals['bwt']:>8.4f} {vals['fwt']:>8.4f}"
            )

    # ================================================================
    # Save JSON sidecar
    # ================================================================
    all_results["meta"] = {
        "tasks_mnist": args.tasks,
        "tasks_cifar": 5,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "ewc_lambda": args.ewc_lambda,
        "agem_buffer_per_task": args.agem_buffer,
        "device": str(device),
        "total_wall_clock_s": round(total_wall, 1),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "b2_cl_defenses.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
