"""v0.5 structure-amount probe: is MORE structure the problem, or ONLINE addition?

The three prior v0.5 experiments (interval, pe_gated, init_scale)
established that growth *of any kind* loses to the 14-node
no_growth baseline. The disruption is not trigger timing and not
initial edge magnitude — it must come from the structural event
itself (new nodes entering topological order, Hebbian updates on
edges that did not exist before, homeostatic re-balancing).

This sweep separates two indistinguishable hypotheses:

- H_structure: "More nodes, period, is the problem." A graph with
  49 frozen nodes from t=0 should also lose to the 14-node frozen
  graph, for the same reason the grown 49-node variant does.
- H_online: "Adding nodes *during* training is the problem." A
  graph with 49 frozen nodes from t=0 should match or beat the
  14-node baseline; it is only the online insertion that
  disturbs the circuit.

Variants:
  - no_growth_14 (initial_associator_count=8 + 4 integrators + 2
    boundary = 14 nodes, all frozen from t=0)
  - no_growth_25 (25-node frozen, matches pe_gated endpoint)
  - no_growth_49 (49-node frozen, matches full-interval endpoint)

All variants have growth fully disabled
(synaptogenesis/neurogenesis/pruning intervals = 0) and differ
only in starting graph size. The schedule and seed match the
other v0.5 probes exactly.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.environments import SequenceEnv, make_capacity_schedule


@dataclass
class VariantResult:
    name: str
    mse_per_step: list[float]
    regime_per_step: list[int]
    n_nodes: list[float]
    n_edges: list[float]
    wall_time: float


def build_config(dim: int, variant: str) -> SOMAConfig:
    """Build a frozen-topology config with a target initial node count."""
    initial_associator_counts = {
        "no_growth_14": 8,   # 8 + 4 integrators + 2 boundary = 14
        "no_growth_25": 19,  # 19 + 4 + 2 = 25
        "no_growth_49": 43,  # 43 + 4 + 2 = 49
    }
    if variant not in initial_associator_counts:
        raise ValueError(f"unknown variant {variant!r}")
    n_assoc = initial_associator_counts[variant]
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
        initial_associator_count=n_assoc,
        synaptogenesis_interval=0,
        neurogenesis_interval=0,
        pruning_interval=0,
    )
    return SOMAConfig.developmental(**overrides)


def run_variant(
    variant: str,
    env: SequenceEnv,
    device: torch.device,
    dim: int,
) -> VariantResult:
    config = build_config(dim, variant)
    ps = PredictiveSOMA(config, device=device)
    mses: list[float] = []
    regimes: list[int] = []
    n_nodes: list[float] = []
    n_edges: list[float] = []
    env.reset()
    total = env.schedule.total_steps
    report_every = max(1, total // 10)
    t0 = time.perf_counter()
    for step in range(total):
        obs = env.step().to(device)
        result = ps.process_input(obs)
        mses.append(float(result["prediction_error"]))
        regimes.append(env.current_regime)
        n_nodes.append(float(result["num_nodes"]))
        n_edges.append(float(result["num_edges"]))
        if (step + 1) % report_every == 0:
            recent = mses[-100:] if len(mses) >= 100 else mses
            mean_mse = sum(recent) / len(recent)
            print(
                f"    {variant} step {step + 1}/{total} "
                f"regime={env.current_regime} "
                f"mse100={mean_mse:.4f} "
                f"nodes={int(n_nodes[-1])} edges={int(n_edges[-1])}",
                flush=True,
            )
    dt = time.perf_counter() - t0
    print(
        f"  {variant} wall: {dt:.1f}s, final "
        f"nodes={int(n_nodes[-1])} edges={int(n_edges[-1])}"
    )
    return VariantResult(variant, mses, regimes, n_nodes, n_edges, dt)


def regime_mean_mse(
    mses: list[float],
    regimes: list[int],
    warmup: int = 100,
) -> dict[int, float]:
    by_regime: dict[int, list[float]] = defaultdict(list)
    prev = -1
    regime_start = 0
    for i, r in enumerate(regimes):
        if r != prev:
            regime_start = i
            prev = r
        if i - regime_start >= warmup:
            by_regime[r].append(mses[i])
    return {r: sum(v) / len(v) if v else 0.0 for r, v in by_regime.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    DIM = 16
    STEPS_PER_REGIME = 500
    schedule = make_capacity_schedule(
        dim=DIM, steps_per_regime=STEPS_PER_REGIME,
    )
    env = SequenceEnv(dim=DIM, schedule=schedule, seed=42)
    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps: {schedule.total_steps}\n")

    variants = ["no_growth_14", "no_growth_25", "no_growth_49"]
    results: dict[str, VariantResult] = {}
    for v in variants:
        print(f"=== {v} ===")
        results[v] = run_variant(v, env, device, DIM)
        print()

    print("=" * 90)
    print("REGIME MEAN MSE BY VARIANT (lower is better)")
    print("=" * 90)
    print(f"  {'Regime':<12}", end="")
    for v in variants:
        print(f"{v:<18}", end="")
    print()
    for regime_idx, regime in enumerate(schedule.regimes):
        print(f"  {regime.name:<12}", end="")
        for v in variants:
            mean = regime_mean_mse(
                results[v].mse_per_step, results[v].regime_per_step,
            ).get(regime_idx, 0.0)
            print(f"{mean:<18.4f}", end="")
        print()

    print("\n" + "=" * 78)
    print("FINAL GRAPH SIZE + WALL TIME BY VARIANT")
    print("=" * 78)
    print(f"  {'Variant':<16}{'Nodes':<8}{'Edges':<10}{'Wall(s)':<10}")
    for v in variants:
        r = results[v]
        print(
            f"  {v:<16}{int(r.n_nodes[-1]):<8}{int(r.n_edges[-1]):<10}"
            f"{r.wall_time:<10.1f}"
        )

    out = {
        "config": {
            "dim": DIM,
            "steps_per_regime": STEPS_PER_REGIME,
            "regimes": [r.name for r in schedule.regimes],
        },
        "variants": {
            v: {
                "mse_per_step": results[v].mse_per_step,
                "regime_per_step": results[v].regime_per_step,
                "n_nodes": results[v].n_nodes,
                "n_edges": results[v].n_edges,
                "wall_time": results[v].wall_time,
            } for v in results
        },
        "schedule_boundaries": schedule.boundaries,
    }
    out_path = "research/developmental/results/env_sequence_v05_pre_add_nodes.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
