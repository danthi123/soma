"""v0.5 rate-based sparsity control — clean matched-total test.

The admission-cap experiment (commit 771172d) showed that the per-call
cap doesn't actually reduce total admissions — it just spreads them
temporally. A clean sparsity control requires matching total admission
counts between a locality-biased variant and a random-selection variant
at the SAME total count.

Approach: reduce ``synaptogenesis_rate`` so that unfiltered synap admits
roughly as many edges as ``synap_only_local`` does. If those fewer
edges (randomly placed) match ``synap_only_local``'s MSE benefit,
sparsity is the driver. If ``synap_only_local`` still wins, locality
is primary.

Calibration (from synap sparsity-control run data):
- synap_only at rate=2.0 admits 158 edges.
- synap_only_local admits 32-64 edges (seed-dependent).
- Linearly interpolated: rate=0.63 ~ 50 edges; rate=0.40 ~ 32 edges.

Variants (6):
- ``no_growth``               baseline.
- ``synap_only``              rate=2.0, ~158 admissions (reference harmful).
- ``synap_only_r063``         rate=0.63, target ~50 admissions.
- ``synap_only_r030``         rate=0.30, target ~25 admissions.
- ``synap_only_local``        locality filter @ max_distance=0.5, ~32-64.
- ``neuro_only``              reference positive.

If ``synap_only_r063`` matches ``synap_only_local`` MSE at matched
event count, sparsity is the dominant factor. If ``synap_only_local``
still wins across seeds, locality drives the benefit.

Seeds {0, 1, 42}. 6 variants x 3 seeds = 18 runs, ~80-100 min.

Success criterion:
- LOCALITY WINS if ``synap_only_local`` beats ``synap_only_r063`` and
  ``synap_only_r030`` on 5+/8 regimes across all seeds. Paper narrative
  stands: positional locality is the key principle.
- SPARSITY WINS if ``synap_only_r063`` matches (ties or beats)
  ``synap_only_local`` on 5+/8 regimes across all seeds. Paper narrative
  weakens: fewer random edges suffice, locality is epiphenomenal.
- MIXED: each seed gives different verdicts. Needs more seeds or
  design iteration.
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


RATES: dict[str, float] = {
    "synap_only_r063": 0.63,
    "synap_only_r030": 0.30,
}


def build_config(dim: int, variant: str, seed: int) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
        pruning_interval=0,
        seed=seed,
    )
    LOCALITY = 0.5
    if variant == "no_growth":
        overrides.update(synaptogenesis_interval=0, neurogenesis_interval=0)
    elif variant == "synap_only":
        overrides.update(neurogenesis_interval=0)
    elif variant in RATES:
        overrides.update(
            neurogenesis_interval=0,
            synaptogenesis_rate=RATES[variant],
        )
    elif variant == "synap_only_local":
        overrides.update(
            neurogenesis_interval=0,
            synaptogenesis_max_distance=LOCALITY,
        )
    elif variant == "neuro_only":
        overrides.update(synaptogenesis_interval=0)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def run_variant(
    seed: int, variant: str, env: SequenceEnv, device: torch.device, dim: int,
) -> RunResult:
    config = build_config(dim, variant, seed)
    ps = PredictiveSOMA(config, device=device)
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
        "synap_only",
        "synap_only_r063", "synap_only_r030",
        "synap_only_local",
        "neuro_only",
    ]

    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps/variant: {schedule.total_steps}")
    print(f"Seeds: {SEEDS}, Variants: {variants}")
    print(f"Rates: {RATES}")
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

    print("=" * 140)
    print("PER-REGIME MEAN MSE (MEAN +/- STDDEV across 3 seeds)")
    print("=" * 140)
    print(f"  {'Regime':<12}", end="")
    for v in variants:
        print(f"{v:<20}", end="")
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
            print(f"{m:.4f}+-{sd:.4f}   ", end="")
        print()

    pairs = [
        ("synap_only", "synap_only_r063"),
        ("synap_only", "synap_only_r030"),
        ("synap_only", "synap_only_local"),
        ("synap_only_r063", "synap_only_local"),
        ("synap_only_r030", "synap_only_local"),
    ]
    for base, treat in pairs:
        print()
        print("=" * 90)
        print(f"TREATMENT DELTAS: {treat} - {base} (negative = treat beats base)")
        print("=" * 90)
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

    for base, treat in pairs:
        print()
        print(f"Improvement scorecard ({treat} beats {base} per seed, noise floor 0.00005):")
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

    out: dict[str, Any] = {
        "config": {
            "dim": DIM,
            "steps_per_regime": STEPS_PER_REGIME,
            "seeds": SEEDS,
            "regimes": regime_names,
            "locality_cutoff": 0.5,
            "rates": RATES,
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
    out_path = (
        "research/developmental/results/"
        "env_sequence_v05_synap_rate_control_multiseed.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
