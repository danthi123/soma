"""v0.5 synap-volume probe: was the 2x2 full worse because of synap *mechanism* or *volume*?

The growth_2x2 sweep showed:
  synap_only (14n/181e, 157 synap events): hurts on 7/8
  full       (48n/960e, 653 synap events): worst variant

These differ in BOTH graph size and synaptogenesis volume.
Synaptogenesis rate scales with node count, so we cannot tell
whether `full` is worse because:
  (a) synaptogenesis is bad in proportion to how many events
      fire — volume hypothesis
  (b) synaptogenesis is bad regardless of volume — mechanism
      hypothesis

This runner isolates by pre-adding nodes WITHOUT neurogenesis so
the graph starts near `full`'s endpoint size, and letting
synaptogenesis + Hebbian run. Matching volume of synaptogenesis
firing while disabling neurogenesis.

Variants:
- neuro_only          (50n/384e, 0 synap) — from 2x2, reproduced
- synap_preadded_48   (48n pre-added frozen + synap on, no neuro,
                       no prune) — expect ~650 synap events
- synap_preadded_25   (25n + synap on) — intermediate
- no_growth_14        (14n/24e) — reference baseline

If synap_preadded_48 still hurts like 2x2 `full`, mechanism is
the issue (volume is incidental). If it matches or beats
no_growth, volume at large graphs was driving the damage.
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
    neurogenesis_event_steps: list[int]
    synaptogenesis_event_steps: list[int]


def build_config(dim: int, variant: str) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
        pruning_interval=0,
    )
    if variant == "no_growth_14":
        overrides.update(
            initial_associator_count=8,
            synaptogenesis_interval=0, neurogenesis_interval=0,
        )
    elif variant == "neuro_only_14start":
        overrides.update(
            initial_associator_count=8,
            synaptogenesis_interval=0,
        )
    elif variant == "synap_preadded_25":
        overrides.update(
            initial_associator_count=19,  # 19 + 4 + 2 = 25
            neurogenesis_interval=0,
        )
    elif variant == "synap_preadded_48":
        overrides.update(
            initial_associator_count=42,  # 42 + 4 + 2 = 48
            neurogenesis_interval=0,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def run_variant(
    variant: str, env: SequenceEnv, device: torch.device, dim: int,
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
    neuro = [int(e["step"]) for e in ps.soma.growth_log
             if e.get("event_type") == "neurogenesis"]
    synap = [int(e["step"]) for e in ps.soma.growth_log
             if e.get("event_type") == "synaptogenesis"]
    print(
        f"  {variant} wall: {dt:.1f}s, "
        f"neuro={len(neuro)} synap={len(synap)}"
    )
    return VariantResult(variant, mses, regimes, n_nodes, n_edges, dt, neuro, synap)


def regime_mean_mse(
    mses: list[float], regimes: list[int], warmup: int = 100,
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
    schedule = make_capacity_schedule(dim=DIM, steps_per_regime=STEPS_PER_REGIME)
    env = SequenceEnv(dim=DIM, schedule=schedule, seed=42)
    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps: {schedule.total_steps}\n")

    variants = [
        "no_growth_14",
        "synap_preadded_25",
        "synap_preadded_48",
        "neuro_only_14start",
    ]
    results: dict[str, VariantResult] = {}
    for v in variants:
        print(f"=== {v} ===")
        results[v] = run_variant(v, env, device, DIM)
        print()

    print("=" * 110)
    print("REGIME MEAN MSE BY VARIANT (lower is better)")
    print("=" * 110)
    print(f"  {'Regime':<12}", end="")
    for v in variants:
        print(f"{v:<22}", end="")
    print()
    for regime_idx, regime in enumerate(schedule.regimes):
        print(f"  {regime.name:<12}", end="")
        for v in variants:
            mean = regime_mean_mse(
                results[v].mse_per_step, results[v].regime_per_step,
            ).get(regime_idx, 0.0)
            print(f"{mean:<22.4f}", end="")
        print()

    print("\n" + "=" * 90)
    print("FINAL GRAPH + GROWTH EVENT COUNTS")
    print("=" * 90)
    print(
        f"  {'Variant':<22}{'Nodes':<8}{'Edges':<8}"
        f"{'NeuroEv':<10}{'SynapEv':<10}{'Wall(s)':<10}"
    )
    for v in variants:
        r = results[v]
        print(
            f"  {v:<22}{int(r.n_nodes[-1]):<8}{int(r.n_edges[-1]):<8}"
            f"{len(r.neurogenesis_event_steps):<10}"
            f"{len(r.synaptogenesis_event_steps):<10}"
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
                "neurogenesis_event_steps": results[v].neurogenesis_event_steps,
                "synaptogenesis_event_steps": results[v].synaptogenesis_event_steps,
            } for v in results
        },
        "schedule_boundaries": schedule.boundaries,
    }
    out_path = "research/developmental/results/env_sequence_v05_synap_volume.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
