"""Phase 5 (audit plan Phase 2): Learning rules sweep.

Tests the impact of SOMA's learning hyperparameters on memory retrieval
quality after consolidation.

Sweeps:
- hebbian_lr in {0, 0.0001, 0.001, 0.01}
- edge_weight_decay in {1.0, 0.9999, 0.999}
- youth_lr_multiplier in {1.0, 3.0, 5.0}

Each condition: store 50 facts, consolidate, flood 200, query 20.
Measures: retrieval recall, consolidation speed.

Usage::

    python -m research.audit.phase5_learning_rules
    python -m research.audit.phase5_learning_rules --quick
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from soma.core.config import SOMAConfig
from research.audit.synthetic_corpus import (
    populate_and_measure,
)

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("research/audit/results")

# Base config
BASE_CONFIG = SOMAConfig(
    vocab_size=256,
    text_embed_dim=64,
    sensor_output_dim=64,
    max_input_tokens=128,
    initial_integrator_count=8,
    initial_associator_count=16,
    seed=42,
)


def sweep_single_param(
    param_name: str,
    values: list[Any],
    *,
    device: torch.device,
    k: int = 5,
) -> list[dict]:
    """Sweep one parameter while holding others at defaults."""
    results = []
    for val in values:
        print(f"  {param_name}={val} ...", end=" ", flush=True)
        cfg = replace(BASE_CONFIG, **{param_name: val})
        r = populate_and_measure(
            cfg,
            use_soma=True,
            do_consolidate=True,
            do_flood=True,
            k=k,
            device=device,
        )
        r["param_name"] = param_name
        r["param_value"] = val
        results.append(r)
        print(
            f"recall={r['recall']:.4f}  "
            f"consol={r['consolidation_time_s']:.2f}s  "
            f"total={r['total_time_s']:.2f}s"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Learning rules sweep"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke test with fewer sweep values",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else "cpu"
    )
    print(f"Device: {device}")

    all_results: dict[str, list[dict]] = {}

    # --- hebbian_lr ---
    print("\n=== hebbian_lr ===")
    heb_values = [0.0, 0.001] if args.quick else [0.0, 0.0001, 0.001, 0.01]
    all_results["hebbian_lr"] = sweep_single_param(
        "hebbian_lr", heb_values, device=device, k=args.k,
    )

    # --- edge_weight_decay ---
    print("\n=== edge_weight_decay ===")
    decay_values = [1.0, 0.999] if args.quick else [1.0, 0.9999, 0.999]
    all_results["edge_weight_decay"] = sweep_single_param(
        "edge_weight_decay", decay_values, device=device, k=args.k,
    )

    # --- youth_lr_multiplier ---
    print("\n=== youth_lr_multiplier ===")
    youth_values = [1.0, 5.0] if args.quick else [1.0, 3.0, 5.0]
    all_results["youth_lr_multiplier"] = sweep_single_param(
        "youth_lr_multiplier", youth_values, device=device, k=args.k,
    )

    # Summary table
    print(f"\n{'=' * 72}")
    print("PHASE 5: LEARNING RULES SWEEP")
    print(f"{'=' * 72}")

    for sweep_name, sweep_results in all_results.items():
        print(f"\n--- {sweep_name} ---")
        print(
            f"{'Value':>12s} {'Recall':>8s} {'Nodes':>7s} "
            f"{'Edges':>7s} {'Consol(s)':>10s} {'Total(s)':>9s}"
        )
        print("-" * 57)
        for r in sweep_results:
            val = str(r["param_value"])
            print(
                f"{val:>12s} {r['recall']:>8.4f} {r['node_count']:>7d} "
                f"{r['edge_count']:>7d} {r['consolidation_time_s']:>9.2f}s "
                f"{r['total_time_s']:>8.2f}s"
            )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase5_learning_rules.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
