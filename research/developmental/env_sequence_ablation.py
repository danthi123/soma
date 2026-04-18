"""v0 ablation: isolate which SOMA mechanisms drive the adaptation gains.

The base v0 result showed SOMA 3-33x lower MSE than an online-MLP
capacity-matched baseline across 4 regimes. This ablation asks:
which mechanisms are load-bearing? We run the same sequence env
(2000 steps, 4 regimes) with the following configs:

  full:       all mechanisms enabled (reference)
  no_growth:  synaptogenesis, neurogenesis, pruning all disabled
  no_consol:  consolidation cycles disabled
  no_hebbian: Hebbian learning rate set to 0 (edge weights only
              via backprop)

Reports regime mean MSE and adaptation windows per variant.
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
from soma.environments import SequenceEnv, make_default_schedule


@dataclass
class AblationResult:
    name: str
    mse_per_step: list[float]
    regime_per_step: list[int]
    n_nodes: list[float]
    n_edges: list[float]
    wall_time: float


def build_config(dim: int, variant: str) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
    )
    if variant == "no_growth":
        overrides.update(
            synaptogenesis_interval=0,
            neurogenesis_interval=0,
            pruning_interval=0,
        )
    elif variant == "no_consol":
        overrides.update(consolidation_interval=0)
    elif variant == "no_hebbian":
        overrides.update(hebbian_lr=0.0)
    elif variant == "full":
        pass
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def run_variant(
    variant: str,
    env: SequenceEnv,
    device: torch.device,
    dim: int,
) -> AblationResult:
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
    print(f"  {variant} wall: {dt:.1f}s")
    return AblationResult(
        variant, mses, regimes, n_nodes, n_edges, dt,
    )


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
    schedule = make_default_schedule(dim=DIM, steps_per_regime=STEPS_PER_REGIME)
    env = SequenceEnv(dim=DIM, schedule=schedule, seed=42)

    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps: {schedule.total_steps}\n")

    variants = ["full", "no_growth", "no_consol", "no_hebbian"]
    results: dict[str, AblationResult] = {}
    for v in variants:
        print(f"=== {v} ===")
        results[v] = run_variant(v, env, device, DIM)
        print()

    print("=" * 78)
    print("REGIME MEAN MSE BY VARIANT (lower is better)")
    print("=" * 78)
    print(f"  {'Regime':<20}", end="")
    for v in variants:
        print(f"{v:<14}", end="")
    print()
    for regime_idx, regime in enumerate(schedule.regimes):
        print(f"  {regime.name:<20}", end="")
        for v in variants:
            mean = regime_mean_mse(
                results[v].mse_per_step, results[v].regime_per_step,
            ).get(regime_idx, 0.0)
            print(f"{mean:<14.4f}", end="")
        print()

    print("\n" + "=" * 78)
    print("FINAL GRAPH SIZE BY VARIANT")
    print("=" * 78)
    print(f"  {'Variant':<15}{'Nodes':<10}{'Edges':<10}{'Wall(s)':<10}")
    for v in variants:
        r = results[v]
        print(
            f"  {v:<15}{int(r.n_nodes[-1]):<10}{int(r.n_edges[-1]):<10}"
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
            } for v in variants
        },
        "schedule_boundaries": schedule.boundaries,
    }
    out_path = "research/developmental/results/env_sequence_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
