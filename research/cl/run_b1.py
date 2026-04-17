"""B1 orchestrator: run vanilla MLP on Permuted-MNIST and Split-CIFAR-10.

Usage::

    python -m research.cl.run_b1                # full 10-task Permuted-MNIST
    python -m research.cl.run_b1 --tasks 3 --epochs 2   # quick smoke test
    python -m research.cl.run_b1 --cifar         # also run Split-CIFAR-10
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from research.cl import datasets, harness
from research.cl.baselines.naive import cifar_factory, mnist_factory


def _print_matrix(A: np.ndarray, label: str) -> None:
    T = A.shape[0]
    print(f"\n{'=' * 60}")
    print(f"Accuracy matrix ({label})  —  A[i][j] = acc on task j after training task i")
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
    print(f"Per-task times: {[f'{t:.1f}s' for t in result['task_times']]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="B1: vanilla MLP CL benchmark")
    parser.add_argument("--tasks", type=int, default=10, help="Number of Permuted-MNIST tasks")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--cifar", action="store_true", help="Also run Split-CIFAR-10")
    parser.add_argument("--seed", type=int, default=0, help="Permutation seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Permuted-MNIST ----
    print(f"\n>>> Permuted-MNIST ({args.tasks} tasks, {args.epochs} epochs/task)")
    mnist_tasks = datasets.permuted_mnist(
        n_tasks=args.tasks,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    mnist_result = harness.run(
        model_factory=mnist_factory,
        tasks=mnist_tasks,
        n_epochs=args.epochs,
        device=device,
    )
    _print_results(mnist_result, "Permuted-MNIST")

    # ---- Split-CIFAR-10 (optional) ----
    if args.cifar:
        print("\n>>> Split-CIFAR-10 (5 tasks, 2 classes each)")
        cifar_tasks = datasets.split_cifar10(batch_size=args.batch_size)
        cifar_result = harness.run(
            model_factory=cifar_factory,
            tasks=cifar_tasks,
            n_epochs=args.epochs,
            device=device,
        )
        _print_results(cifar_result, "Split-CIFAR-10")


if __name__ == "__main__":
    main()
