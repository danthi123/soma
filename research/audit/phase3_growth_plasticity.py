"""Phase 3 (audit plan Phase 3): Growth and plasticity sweep.

Tests whether SOMA's structural plasticity mechanisms (synaptogenesis,
neurogenesis, pruning) improve or harm memory retrieval.

Sweeps:
- synaptogenesis_interval in {0, 100, 500}
- neurogenesis_interval in {0, 500, 1000}
- pruning_interval in {0, 1000, 5000}
- Compound: static vs growth-only vs growth+pruning vs full pipeline

Each condition: store 50 facts, consolidate, flood 200, query 20.
Measures: retrieval recall, graph size, consolidation time.

Usage::

    python -m research.audit.phase3_growth_plasticity
    python -m research.audit.phase3_growth_plasticity --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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

# Base config: small SOMA for fast sweeps
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
        print(f"recall={r['recall']:.4f}  nodes={r['node_count']}  "
              f"edges={r['edge_count']}  consol={r['consolidation_time_s']:.2f}s")
    return results


def sweep_compound(
    *,
    device: torch.device,
    k: int = 5,
) -> list[dict]:
    """Test compound configurations: static, growth-only, growth+pruning, full."""
    configs: list[tuple[str, dict[str, Any]]] = [
        ("static_graph", {
            "synaptogenesis_interval": 0,
            "neurogenesis_interval": 0,
            "pruning_interval": 0,
        }),
        ("growth_only", {
            "synaptogenesis_interval": 100,
            "neurogenesis_interval": 500,
            "pruning_interval": 0,
        }),
        ("growth_and_pruning", {
            "synaptogenesis_interval": 100,
            "neurogenesis_interval": 500,
            "pruning_interval": 1000,
        }),
        ("full_pipeline", {
            # Full pipeline: growth + pruning + myelination (default intervals)
            "synaptogenesis_interval": 100,
            "neurogenesis_interval": 500,
            "pruning_interval": 1000,
            # Myelination fires during consolidation based on thresholds;
            # just ensuring growth is active is enough for it to trigger.
        }),
    ]

    results = []
    for label, overrides in configs:
        print(f"  compound: {label} ...", end=" ", flush=True)
        cfg = replace(BASE_CONFIG, **overrides)
        r = populate_and_measure(
            cfg,
            use_soma=True,
            do_consolidate=True,
            do_flood=True,
            k=k,
            device=device,
        )
        r["param_name"] = "compound"
        r["param_value"] = label
        results.append(r)
        print(f"recall={r['recall']:.4f}  nodes={r['node_count']}  "
              f"edges={r['edge_count']}  consol={r['consolidation_time_s']:.2f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Growth and plasticity sweep"
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

    # --- Synaptogenesis interval ---
    print("\n=== Synaptogenesis interval ===")
    syn_values = [0, 100] if args.quick else [0, 100, 500]
    all_results["synaptogenesis_interval"] = sweep_single_param(
        "synaptogenesis_interval", syn_values, device=device, k=args.k,
    )

    # --- Neurogenesis interval ---
    print("\n=== Neurogenesis interval ===")
    neuro_values = [0, 500] if args.quick else [0, 500, 1000]
    all_results["neurogenesis_interval"] = sweep_single_param(
        "neurogenesis_interval", neuro_values, device=device, k=args.k,
    )

    # --- Pruning interval ---
    print("\n=== Pruning interval ===")
    prune_values = [0, 1000] if args.quick else [0, 1000, 5000]
    all_results["pruning_interval"] = sweep_single_param(
        "pruning_interval", prune_values, device=device, k=args.k,
    )

    # --- Compound configurations ---
    print("\n=== Compound configurations ===")
    if args.quick:
        # Just static vs full
        configs_quick: list[tuple[str, dict[str, Any]]] = [
            ("static_graph", {
                "synaptogenesis_interval": 0,
                "neurogenesis_interval": 0,
                "pruning_interval": 0,
            }),
            ("full_pipeline", {
                "synaptogenesis_interval": 100,
                "neurogenesis_interval": 500,
                "pruning_interval": 1000,
            }),
        ]
        compound_results = []
        for label, overrides in configs_quick:
            print(f"  compound: {label} ...", end=" ", flush=True)
            cfg = replace(BASE_CONFIG, **overrides)
            r = populate_and_measure(
                cfg, use_soma=True, do_consolidate=True, do_flood=True,
                k=args.k, device=device,
            )
            r["param_name"] = "compound"
            r["param_value"] = label
            compound_results.append(r)
            print(f"recall={r['recall']:.4f}")
        all_results["compound"] = compound_results
    else:
        all_results["compound"] = sweep_compound(device=device, k=args.k)

    # Summary table
    print(f"\n{'=' * 80}")
    print("PHASE 3: GROWTH & PLASTICITY SWEEP")
    print(f"{'=' * 80}")

    for sweep_name, sweep_results in all_results.items():
        print(f"\n--- {sweep_name} ---")
        print(
            f"{'Value':<20s} {'Recall':>8s} {'Nodes':>7s} "
            f"{'Edges':>7s} {'Consol(s)':>10s} {'Total(s)':>9s}"
        )
        print("-" * 65)
        for r in sweep_results:
            val = str(r["param_value"])
            print(
                f"{val:<20s} {r['recall']:>8.4f} {r['node_count']:>7d} "
                f"{r['edge_count']:>7d} {r['consolidation_time_s']:>9.2f}s "
                f"{r['total_time_s']:>8.2f}s"
            )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase3_growth_plasticity.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
