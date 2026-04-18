"""Node-count sweep: frozen SOMA with {8, 16, 32, 64, 128} integrators.

Tests whether SOMA's CL capacity scales with node count. D4 theory
predicts capacity = O(output_dim) not O(node_count), so accuracy
should be flat while wall-clock scales linearly.

Usage::

    python -m research.cl.run_node_sweep                  # default 5 tasks
    python -m research.cl.run_node_sweep --tasks 10       # full scale
    CUDA_VISIBLE_DEVICES="" python -m research.cl.run_node_sweep  # CPU only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.cl import datasets, harness
from research.cl.run_b3 import _print_results, _subsample_tasks
from research.cl.soma_cl import SomaClassifier, make_soma_cl_components

INTEGRATOR_COUNTS = [8, 16, 32, 64]
REPORTS_DIR = Path("research/cl/reports")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Node-count sweep: frozen SOMA on Permuted-MNIST"
    )
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load and subsample dataset once (shared across all node counts)
    mnist_tasks = datasets.permuted_mnist(
        n_tasks=args.tasks,
        batch_size=args.batch_size,
        seed=0,
    )
    mnist_tasks = _subsample_tasks(
        mnist_tasks,
        max_train_samples=args.train_samples,
        max_test_samples=args.test_samples,
        batch_size=args.batch_size,
    )

    results: dict[str, dict] = {}

    for n_int in INTEGRATOR_COUNTS:
        n_assoc = n_int * 2
        # sensor(1) + output(1) + integrators + associators + 4 infrastructure
        total_nodes = n_int + n_assoc + 6

        print(f"\n{'=' * 60}")
        print(
            f"Frozen SOMA: {n_int} integrators, {n_assoc} associators "
            f"({total_nodes} total nodes)"
        )
        print(f"{'=' * 60}")

        factory, train_fn, hook = make_soma_cl_components(
            dataset="mnist",
            frozen=True,
            enable_consolidation=False,
            disable_critical_periods=False,
            seed=args.seed,
            integrator_count=n_int,
        )

        # Cache SOMA features for this architecture
        _model, _opt = factory(device)
        run_tasks = mnist_tasks
        if isinstance(_model, SomaClassifier):
            run_tasks = _model.precompute_all_tasks(mnist_tasks, device)
        _built = [(_model, _opt)]

        def _prebuilt_factory(
            dev: torch.device, _ref: list = _built,
        ) -> tuple:
            return _ref[0]

        t0 = time.perf_counter()
        result = harness.run(
            model_factory=_prebuilt_factory,
            tasks=run_tasks,
            n_epochs=args.epochs,
            device=device,
            verbose=True,
            train_one_epoch=train_fn,
            on_task_end=hook,
        )
        wall = time.perf_counter() - t0

        _print_results(result, f"frozen (int={n_int})")

        results[f"int_{n_int}"] = {
            "integrators": n_int,
            "associators": n_assoc,
            "total_nodes": total_nodes,
            "acc": round(result["acc"], 4),
            "bwt": round(result["bwt"], 4),
            "fwt": round(result["fwt"], 4),
            "wall_clock_s": round(wall, 1),
        }

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print("NODE-COUNT SWEEP RESULTS (frozen SOMA, Permuted-MNIST)")
    print(f"{'=' * 60}")
    print(
        f"{'Integrators':>12s} {'Nodes':>8s} {'ACC':>8s} "
        f"{'BWT':>8s} {'FWT':>8s} {'Wall(s)':>8s}"
    )
    print("-" * 60)
    for _key, r in results.items():
        print(
            f"{r['integrators']:>12d} {r['total_nodes']:>8d} "
            f"{r['acc']:>8.4f} {r['bwt']:>8.4f} "
            f"{r['fwt']:>8.4f} {r['wall_clock_s']:>8.1f}"
        )

    # D4 prediction check
    accs = [r["acc"] for r in results.values()]
    acc_range = max(accs) - min(accs)
    print(f"\nACC range across node counts: {acc_range:.4f}")
    if acc_range < 0.03:
        print(
            "CONFIRMS D4 theory: capacity = O(output_dim), "
            "not O(node_count). Accuracy flat across node counts."
        )
    else:
        print(
            "CONTRADICTS D4 theory: accuracy varies meaningfully "
            "with node count. Investigate further."
        )

    # Save
    results["meta"] = {
        "tasks": args.tasks,
        "epochs": args.epochs,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "seed": args.seed,
        "device": str(device),
        "integrator_counts": INTEGRATOR_COUNTS,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "node_sweep_cl.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
