"""Phase 4 (audit plan 1.1-1.2): Graph dimension sweeps.

Tests the impact of SOMA's core dimensional parameters on memory
retrieval quality, consolidation speed, and resource usage.

Sweeps:
- sensor_output_dim in {16, 32, 64, 128}
- initial_integrator_count in {1, 4, 8, 16, 32}
- position_dim in {1, 8, 16}
  (Note: position_dim=0 rejected by SOMAConfig validation; 1 is min)

Each condition: store 50 facts, consolidate, flood 200, query 20.
Measures: retrieval recall, consolidation speed, memory usage.

Usage::

    python -m research.audit.phase4_dimensions
    python -m research.audit.phase4_dimensions --quick
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


def _get_memory_mb() -> float:
    """Return current process RSS in MB (best-effort)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def sweep_single_param(
    param_name: str,
    values: list[Any],
    *,
    coupled: dict[str, Any] | None = None,
    device: torch.device,
    k: int = 5,
) -> list[dict]:
    """Sweep one parameter while holding others at defaults.

    Parameters
    ----------
    coupled : dict, optional
        Extra config overrides that must change together with the swept
        param (e.g., when sensor_output_dim changes, associator_input_dim
        must match).
    """
    results = []
    for val in values:
        print(f"  {param_name}={val} ...", end=" ", flush=True)

        overrides: dict[str, Any] = {param_name: val}
        if coupled:
            for ck, cv in coupled.items():
                # If coupled value is a callable, call it with the swept value
                overrides[ck] = cv(val) if callable(cv) else cv

        mem_before = _get_memory_mb()
        cfg = replace(BASE_CONFIG, **overrides)
        r = populate_and_measure(
            cfg,
            use_soma=True,
            do_consolidate=True,
            do_flood=True,
            k=k,
            device=device,
        )
        mem_after = _get_memory_mb()
        r["param_name"] = param_name
        r["param_value"] = val
        r["memory_delta_mb"] = round(mem_after - mem_before, 2)

        results.append(r)
        print(
            f"recall={r['recall']:.4f}  nodes={r['node_count']}  "
            f"consol={r['consolidation_time_s']:.2f}s  "
            f"mem_delta={r['memory_delta_mb']:+.1f}MB"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Graph dimension sweeps"
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

    # --- sensor_output_dim ---
    # When sensor_output_dim changes, associator_input_dim must match
    # (it's the input to associator nodes which receive from sensor nodes).
    print("\n=== sensor_output_dim ===")
    sod_values = [16, 64] if args.quick else [16, 32, 64, 128]
    all_results["sensor_output_dim"] = sweep_single_param(
        "sensor_output_dim",
        sod_values,
        coupled={
            "associator_input_dim": lambda v: v,
            "text_embed_dim": lambda v: v,
        },
        device=device,
        k=args.k,
    )

    # --- initial_integrator_count ---
    print("\n=== initial_integrator_count ===")
    integ_values = [1, 8] if args.quick else [1, 4, 8, 16, 32]
    all_results["initial_integrator_count"] = sweep_single_param(
        "initial_integrator_count",
        integ_values,
        coupled={
            # Associator count conventionally set to 2x integrators
            "initial_associator_count": lambda v: max(2, v * 2),
        },
        device=device,
        k=args.k,
    )

    # --- position_dim ---
    # Note: SOMAConfig requires position_dim >= 1 (positive int validation).
    # We use 1 as the "effectively off" value.
    print("\n=== position_dim ===")
    pos_values = [1, 16] if args.quick else [1, 8, 16]
    all_results["position_dim"] = sweep_single_param(
        "position_dim",
        pos_values,
        device=device,
        k=args.k,
    )

    # Summary table
    print(f"\n{'=' * 80}")
    print("PHASE 4: GRAPH DIMENSIONS SWEEP")
    print(f"{'=' * 80}")

    for sweep_name, sweep_results in all_results.items():
        print(f"\n--- {sweep_name} ---")
        print(
            f"{'Value':>8s} {'Recall':>8s} {'Nodes':>7s} "
            f"{'Edges':>7s} {'Consol(s)':>10s} {'Mem(MB)':>9s}"
        )
        print("-" * 56)
        for r in sweep_results:
            print(
                f"{r['param_value']:>8} {r['recall']:>8.4f} "
                f"{r['node_count']:>7d} {r['edge_count']:>7d} "
                f"{r['consolidation_time_s']:>9.2f}s "
                f"{r['memory_delta_mb']:>+8.1f}"
            )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase4_dimensions.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
