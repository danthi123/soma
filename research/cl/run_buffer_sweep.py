"""Buffer-size sweep for SOMA head-replay on Permuted-MNIST.

Caches frozen SOMA features ONCE, then sweeps buffer sizes on the
cached tensors.  Each buffer size takes ~3-5s (head-only training).

Usage::

    python -m research.cl.run_buffer_sweep                # coarse pass
    python -m research.cl.run_buffer_sweep --fine 175 225  # zoom in
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from research.cl import datasets, harness
from research.cl.run_b3 import _print_results, _result_summary, _subsample_tasks
from research.cl.soma_cl import SomaClassifier, make_soma_cl_components

REPORTS_DIR = Path("research/cl/reports")


def _run_single(
    buf_size: int,
    ratio: float,
    cached_tasks: list[tuple[int, DataLoader, DataLoader]],
    model_ref: list,
    n_epochs: int,
    device: torch.device,
    seed: int,
    integrators: int,
) -> dict:
    """Run one buffer-size config on pre-cached features."""
    factory, train_fn, hook = make_soma_cl_components(
        dataset="mnist",
        frozen=True,
        enable_consolidation=False,
        disable_critical_periods=False,
        seed=seed,
        integrator_count=integrators,
        head_replay=True,
        replay_buffer_size=buf_size,
        replay_mix_ratio=ratio,
    )

    # Re-use the pre-built SOMA model but reset the head + input_proj
    _model, _opt = model_ref
    import copy
    fresh_model = copy.deepcopy(_model)
    from torch.optim import SGD
    fresh_opt = SGD(
        [p for p in fresh_model.parameters() if p.requires_grad],
        lr=0.1,
    )

    def _factory(dev):
        return fresh_model, fresh_opt

    result = harness.run(
        model_factory=_factory,
        tasks=cached_tasks,
        n_epochs=n_epochs,
        device=device,
        verbose=False,
        train_one_epoch=train_fn,
        on_task_end=hook,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Buffer-size sweep")
    parser.add_argument(
        "--fine", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"),
        help="Fine sweep range (e.g. --fine 175 225)",
    )
    parser.add_argument("--step", type=int, default=None, help="Step size")
    parser.add_argument("--ratio", type=float, default=0.5, help="Mix ratio")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--integrators", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    print(f"Device: {device}")

    # Determine sweep points
    if args.fine:
        lo, hi = args.fine
        step = args.step or 10
        buf_sizes = list(range(lo, hi + 1, step))
    else:
        step = args.step or 50
        buf_sizes = list(range(100, 351, step))

    print(f"Buffer sizes to sweep: {buf_sizes}")
    print(f"Mix ratio: {args.ratio}")

    # Load and subsample data
    mnist_tasks = datasets.permuted_mnist(
        n_tasks=args.tasks, batch_size=args.batch_size, seed=0,
    )
    mnist_tasks = _subsample_tasks(
        mnist_tasks, args.train_samples, args.test_samples, args.batch_size,
    )

    # Build model and cache features ONCE
    factory, train_fn, hook = make_soma_cl_components(
        dataset="mnist",
        frozen=True,
        enable_consolidation=False,
        disable_critical_periods=False,
        seed=args.seed,
        integrator_count=args.integrators,
        head_replay=True,
        replay_buffer_size=200,
        replay_mix_ratio=args.ratio,
    )
    model, opt = factory(device)
    assert isinstance(model, SomaClassifier)
    cached_tasks = model.precompute_all_tasks(mnist_tasks, device)
    model_ref = (model, opt)

    # Sweep
    results: dict[int, dict] = {}
    t0 = time.perf_counter()

    for buf_size in buf_sizes:
        print(f"\n--- Buffer size: {buf_size} ---")
        t1 = time.perf_counter()
        result = _run_single(
            buf_size, args.ratio, cached_tasks, model_ref,
            args.epochs, device, args.seed, args.integrators,
        )
        dt = time.perf_counter() - t1
        summary = _result_summary(result)
        results[buf_size] = summary
        print(f"  ACC={summary['acc']:.4f}  BWT={summary['bwt']:.4f}  ({dt:.1f}s)")

    total = time.perf_counter() - t0

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"BUFFER SWEEP (ratio={args.ratio})")
    print(f"{'=' * 60}")
    print(f"{'Buffer':>8s} {'ACC':>8s} {'BWT':>8s} {'FWT':>8s}")
    print("-" * 40)

    best_acc = -1.0
    best_buf = -1
    for buf_size in buf_sizes:
        s = results[buf_size]
        marker = ""
        if s["acc"] > best_acc:
            best_acc = s["acc"]
            best_buf = buf_size
        print(f"{buf_size:>8d} {s['acc']:>8.4f} {s['bwt']:>8.4f} {s['fwt']:>8.4f}")

    print(f"\nBest: buffer={best_buf} -> ACC={best_acc:.4f}")
    print(f"Total: {total:.1f}s")

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "sweep_type": "buffer_size",
        "ratio": args.ratio,
        "results": {str(k): v for k, v in results.items()},
        "best_buffer": best_buf,
        "best_acc": best_acc,
        "total_wall_clock_s": round(total, 1),
    }
    json_path = REPORTS_DIR / "buffer_sweep.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {json_path}")


if __name__ == "__main__":
    main()
