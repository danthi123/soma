"""v0.5 Direction 2 probe (cheapest): does activation diversification help?

PredictiveSOMA's ``_diversify_activations()`` multiplies each
associator's activation by a gain derived from aligning the input
with a FROZEN random projection. The plan-doc hypothesis for
Direction 2 is that these random projections are the root cause of
"structural diversity without semantic grounding" — they separate
nodes into different response profiles for different inputs, but
the separations are arbitrary.

Before doing the full Direction 2 work (making projections learnable
and plumbing gradients through them), run the cheapest ablation:
monkey-patch ``_diversify_activations`` to be a no-op and see what
happens to prediction MSE. If diversification is net-negative,
Direction 2's learnable variant should be easier to beat; if it's
net-positive, learnable-and-grounded projections have a clear lift
target (beat the random-but-helpful baseline).

Variants (matches prior multi-seed runs for direct comparison):

- ``no_growth``: reference, no plasticity.
- ``full``: baseline with growth, default (divers ON).
- ``full_nodiv``: baseline with growth, diversification disabled.
- ``neuro_only``: reference.
- ``neuro_only_nodiv``: does divers help the good mechanism?

Seeds: {0, 1, 42}. ~5 variants × 3 seeds = 15 runs ≈ 60-80 min.
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
class RunResult:
    seed: int
    variant: str
    mse_per_step: list[float]
    regime_per_step: list[int]
    n_nodes: list[float]
    n_edges: list[float]
    wall_time: float
    neurogenesis_event_steps: list[int]
    synaptogenesis_event_steps: list[int]


def build_config(dim: int, variant: str, seed: int) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
        pruning_interval=0,
        seed=seed,
    )
    base = variant.replace("_nodiv", "")
    if base == "no_growth":
        overrides.update(synaptogenesis_interval=0, neurogenesis_interval=0)
    elif base == "synap_only":
        overrides.update(neurogenesis_interval=0)
    elif base == "neuro_only":
        overrides.update(synaptogenesis_interval=0)
    elif base == "full":
        pass
    else:
        raise ValueError(f"unknown base variant {base!r}")
    return SOMAConfig.developmental(**overrides)


def _noop_diversify(self, input_tensor, temperature=5.0):
    """Replacement for PredictiveSOMA._diversify_activations that does
    nothing. Attaches as a bound method; 'self' is the PredictiveSOMA
    instance."""
    return None


def run_variant(
    seed: int, variant: str, env: SequenceEnv, device: torch.device, dim: int,
) -> RunResult:
    config = build_config(dim, variant, seed)
    ps = PredictiveSOMA(config, device=device)

    # Monkey-patch diversification off when variant has _nodiv suffix.
    if variant.endswith("_nodiv"):
        import types
        ps._diversify_activations = types.MethodType(_noop_diversify, ps)

    mses: list[float] = []
    regimes: list[int] = []
    n_nodes: list[float] = []
    n_edges: list[float] = []
    env.reset()
    total = env.schedule.total_steps
    t0 = time.perf_counter()
    for _step in range(total):
        obs = env.step().to(device)
        result = ps.process_input(obs)
        mses.append(float(result["prediction_error"]))
        regimes.append(env.current_regime)
        n_nodes.append(float(result["num_nodes"]))
        n_edges.append(float(result["num_edges"]))
    dt = time.perf_counter() - t0
    neuro_events = [
        int(e["step"]) for e in ps.soma.growth_log
        if e.get("event_type") == "neurogenesis"
    ]
    synap_events = [
        int(e["step"]) for e in ps.soma.growth_log
        if e.get("event_type") == "synaptogenesis"
    ]
    return RunResult(
        seed, variant, mses, regimes, n_nodes, n_edges, dt,
        neuro_events, synap_events,
    )


def regime_mean_mse(
    mses: list[float], regimes: list[int], warmup: int = 100,
) -> dict[int, float]:
    by_regime: dict[int, list[float]] = defaultdict(list)
    prev = -1
    start = 0
    for i, r in enumerate(regimes):
        if r != prev:
            start = i
            prev = r
        if i - start >= warmup:
            by_regime[r].append(mses[i])
    return {r: sum(v) / len(v) if v else 0.0 for r, v in by_regime.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    DIM = 16
    STEPS_PER_REGIME = 500
    SEEDS = [0, 1, 42]
    schedule = make_capacity_schedule(dim=DIM, steps_per_regime=STEPS_PER_REGIME)
    env = SequenceEnv(dim=DIM, schedule=schedule, seed=42)
    variants = [
        "no_growth",
        "neuro_only", "neuro_only_nodiv",
        "full", "full_nodiv",
    ]

    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps/variant: {schedule.total_steps}")
    print(f"Seeds: {SEEDS}")
    print(f"Variants: {variants}")
    print()

    results: dict[int, dict[str, RunResult]] = {s: {} for s in SEEDS}
    for seed in SEEDS:
        print(f"=== seed={seed} ===")
        for v in variants:
            r = run_variant(seed, v, env, device, DIM)
            results[seed][v] = r
            means = regime_mean_mse(r.mse_per_step, r.regime_per_step)
            mean_over_regimes = sum(means.values()) / len(means)
            print(
                f"  seed={seed} {v:<20} "
                f"mse={mean_over_regimes:.4f} "
                f"edges={int(r.n_edges[-1])} "
                f"synap={len(r.synaptogenesis_event_steps)} "
                f"neuro={len(r.neurogenesis_event_steps)} "
                f"wall={r.wall_time:.0f}s",
                flush=True,
            )
        print()

    regime_names = [r.name for r in schedule.regimes]

    # Summary: mean across seeds per regime
    print("=" * 110)
    print("PER-REGIME MEAN MSE (MEAN +/- STDDEV across 3 seeds)")
    print("=" * 110)
    print(f"  {'Regime':<12}", end="")
    for v in variants:
        print(f"{v:<22}", end="")
    print()
    for regime_idx, regime in enumerate(regime_names):
        print(f"  {regime:<12}", end="")
        for v in variants:
            vals = []
            for s in SEEDS:
                r = results[s][v]
                means = regime_mean_mse(r.mse_per_step, r.regime_per_step)
                vals.append(means.get(regime_idx, 0.0))
            m = sum(vals) / len(vals)
            sd = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
            print(f"{m:.4f}+-{sd:.4f}     ", end="")
        print()

    # Treatment deltas
    for pair in [("full", "full_nodiv"), ("neuro_only", "neuro_only_nodiv")]:
        base, treat = pair
        print()
        print("=" * 80)
        print(f"TREATMENT DELTAS: {treat} - {base} (negative = nodiv helps)")
        print("=" * 80)
        print(f"  {'Regime':<12}", end="")
        for s in SEEDS:
            print(f"seed={s:<10}", end="")
        print(f"{'mean':<12}")
        for regime_idx, regime in enumerate(regime_names):
            print(f"  {regime:<12}", end="")
            deltas = []
            for s in SEEDS:
                b = regime_mean_mse(
                    results[s][base].mse_per_step,
                    results[s][base].regime_per_step,
                ).get(regime_idx, 0.0)
                t = regime_mean_mse(
                    results[s][treat].mse_per_step,
                    results[s][treat].regime_per_step,
                ).get(regime_idx, 0.0)
                d = t - b
                deltas.append(d)
                print(f"{d:+.4f}     ", end="")
            mean_d = sum(deltas) / len(deltas)
            print(f"{mean_d:+.4f}")

    # Wins scorecard
    for pair in [("full", "full_nodiv"), ("neuro_only", "neuro_only_nodiv")]:
        base, treat = pair
        print()
        print(f"Improvement scorecard ({treat} beats {base} per seed per regime, noise floor 0.00005):")
        for s in SEEDS:
            wins = 0
            total = 0
            for regime_idx in range(len(regime_names)):
                b = regime_mean_mse(
                    results[s][base].mse_per_step,
                    results[s][base].regime_per_step,
                ).get(regime_idx, 0.0)
                t = regime_mean_mse(
                    results[s][treat].mse_per_step,
                    results[s][treat].regime_per_step,
                ).get(regime_idx, 0.0)
                if t < b - 0.00005:
                    wins += 1
                total += 1
            print(f"  seed={s}: {treat} wins {wins}/{total}")

    # Raw data
    out: dict[str, Any] = {
        "config": {
            "dim": DIM,
            "steps_per_regime": STEPS_PER_REGIME,
            "seeds": SEEDS,
            "regimes": regime_names,
        },
        "results": {
            str(s): {
                v: {
                    "mse_per_step": results[s][v].mse_per_step,
                    "regime_per_step": results[s][v].regime_per_step,
                    "n_nodes": results[s][v].n_nodes,
                    "n_edges": results[s][v].n_edges,
                    "wall_time": results[s][v].wall_time,
                    "neurogenesis_event_steps":
                        results[s][v].neurogenesis_event_steps,
                    "synaptogenesis_event_steps":
                        results[s][v].synaptogenesis_event_steps,
                } for v in variants
            } for s in SEEDS
        },
    }
    out_path = "research/developmental/results/env_sequence_v05_no_diversify.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
