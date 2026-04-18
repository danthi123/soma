"""v0.5 growth-decomposition: split synaptogenesis from neurogenesis.

Prior v0.5 ablations treated growth as a single lever. This runner
decomposes it: 2x2 on {synaptogenesis on/off} x {neurogenesis on/off}
while keeping everything else fixed.

- no_growth      (syn off, neuro off): 14n / 24e frozen; baseline
- synap_only     (syn on,  neuro off): 14 nodes, edges grow via
                 synaptogenesis; tests "is more edge density at
                 fixed node count helpful or harmful?"
- neuro_only     (syn off, neuro on):  new nodes appear but
                 existing associators never gain new edges to
                 each other; tests "does adding nodes alone
                 degrade, separate from adding assoc-assoc edges?"
- full           (syn on,  neuro on):  reproduces v0.5 `full`

Pruning is disabled in every variant so removed edges/nodes don't
add a confound.

The edge-count confound in the prior pre-add-nodes finding was
that pre-added 49n frozen (94 edges, no synap) lost badly vs the
grown 49n graph (~980 edges). This 2x2 lets us see whether the
edge density itself is what helps the grown variant, and whether
the act of adding nodes (without new assoc-assoc edges) is the
dominant cost.
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
        pruning_interval=0,  # disable in every variant for cleanliness
    )
    if variant == "no_growth":
        overrides.update(synaptogenesis_interval=0, neurogenesis_interval=0)
    elif variant == "synap_only":
        overrides.update(neurogenesis_interval=0)
    elif variant == "neuro_only":
        overrides.update(synaptogenesis_interval=0)
    elif variant == "full":
        pass  # developmental defaults
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
    neuro_events = [
        int(e["step"]) for e in ps.soma.growth_log
        if e.get("event_type") == "neurogenesis"
    ]
    synap_events = [
        int(e["step"]) for e in ps.soma.growth_log
        if e.get("event_type") == "synaptogenesis"
    ]
    print(
        f"  {variant} wall: {dt:.1f}s, "
        f"neuro={len(neuro_events)} synap={len(synap_events)}"
    )
    return VariantResult(
        variant, mses, regimes, n_nodes, n_edges, dt,
        neuro_events, synap_events,
    )


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

    variants = ["no_growth", "synap_only", "neuro_only", "full"]
    results: dict[str, VariantResult] = {}
    for v in variants:
        print(f"=== {v} ===")
        results[v] = run_variant(v, env, device, DIM)
        print()

    print("=" * 100)
    print("REGIME MEAN MSE BY VARIANT (lower is better)")
    print("=" * 100)
    print(f"  {'Regime':<12}", end="")
    for v in variants:
        print(f"{v:<14}", end="")
    print()
    for regime_idx, regime in enumerate(schedule.regimes):
        print(f"  {regime.name:<12}", end="")
        for v in variants:
            mean = regime_mean_mse(
                results[v].mse_per_step, results[v].regime_per_step,
            ).get(regime_idx, 0.0)
            print(f"{mean:<14.4f}", end="")
        print()

    print("\n" + "=" * 85)
    print("FINAL GRAPH SIZE + GROWTH EVENT COUNTS BY VARIANT")
    print("=" * 85)
    print(
        f"  {'Variant':<14}{'Nodes':<8}{'Edges':<8}"
        f"{'NeuroEv':<10}{'SynapEv':<10}{'Wall(s)':<10}"
    )
    for v in variants:
        r = results[v]
        print(
            f"  {v:<14}{int(r.n_nodes[-1]):<8}{int(r.n_edges[-1]):<8}"
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
    out_path = "research/developmental/results/env_sequence_v05_growth_2x2.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
