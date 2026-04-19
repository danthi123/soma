"""Multi-seed Direction 3 — neuromodulator plasticity broadcast.

Direction 1's single-seed Phase 1 result turned out to be a seed=42
outlier (multi-seed killed 7/8 -> 1/8 on seeds 0 and 1). Direction 3
has two genuinely different properties (proportional gating + gates
Hebbian LR too) so it's worth testing, but we skip the single-seed
"looks promising" step and go straight to multi-seed.

Variants (7): matches plan doc's 2x2 + no_growth + synap arms.

Seeds: {0, 1, 42} same as Direction 1 multi-seed.
Expected runtime: 7 * 3 * ~5min = ~105 min at 4000 steps/variant.

Success criterion (revised from plan doc for post-multi-seed era):
- full_bcast beats full on 5+/8 regimes on ALL three seeds (not just
  mean), OR
- mean MSE improvement across seeds AND regimes is < -0.0005, AND
- neuro_only_bcast is not worse than neuro_only on 6+/8 regimes mean
  (doesn't break the one good mechanism).
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
    plasticity_gain_per_step: list[float]


def build_config(dim: int, variant: str, seed: int) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
        pruning_interval=0,
        seed=seed,
    )
    if variant == "no_growth":
        overrides.update(synaptogenesis_interval=0, neurogenesis_interval=0)
    elif variant == "synap_only":
        overrides.update(neurogenesis_interval=0)
    elif variant == "synap_only_bcast":
        overrides.update(
            neurogenesis_interval=0,
            plasticity_broadcast_mode="pe_scaled",
        )
    elif variant == "neuro_only":
        overrides.update(synaptogenesis_interval=0)
    elif variant == "neuro_only_bcast":
        overrides.update(
            synaptogenesis_interval=0,
            plasticity_broadcast_mode="pe_scaled",
        )
    elif variant == "full":
        pass
    elif variant == "full_bcast":
        overrides.update(plasticity_broadcast_mode="pe_scaled")
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
    gains: list[float] = []
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
        gains.append(float(ps.soma.plasticity_gain))
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
        neuro_events, synap_events, gains,
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
        "synap_only", "synap_only_bcast",
        "neuro_only", "neuro_only_bcast",
        "full", "full_bcast",
    ]

    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps/variant: {schedule.total_steps}")
    print(f"Seeds: {SEEDS}")
    print(f"Variants: {variants}")
    print(f"Total runs: {len(SEEDS) * len(variants)} ({len(SEEDS) * len(variants) * 5} min estimated)")
    print()

    results: dict[int, dict[str, RunResult]] = {s: {} for s in SEEDS}
    for seed in SEEDS:
        print(f"=== seed={seed} ===")
        for v in variants:
            r = run_variant(seed, v, env, device, DIM)
            results[seed][v] = r
            means = regime_mean_mse(r.mse_per_step, r.regime_per_step)
            mean_over_regimes = sum(means.values()) / len(means)
            gain_mean = sum(r.plasticity_gain_per_step) / len(r.plasticity_gain_per_step)
            gain_max = max(r.plasticity_gain_per_step)
            gain_min = min(r.plasticity_gain_per_step)
            print(
                f"  seed={seed} {v:<18} "
                f"mse={mean_over_regimes:.4f} "
                f"edges={int(r.n_edges[-1])} "
                f"synap={len(r.synaptogenesis_event_steps)} "
                f"neuro={len(r.neurogenesis_event_steps)} "
                f"gain[{gain_min:.2f},{gain_mean:.2f},{gain_max:.2f}] "
                f"wall={r.wall_time:.0f}s",
                flush=True,
            )
        print()

    regime_names = [r.name for r in schedule.regimes]

    print("=" * 130)
    print("PER-REGIME MEAN MSE (MEAN +/- STDDEV across 3 seeds)")
    print("=" * 130)
    print(f"  {'Regime':<12}", end="")
    for v in variants:
        print(f"{v:<21}", end="")
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
            print(f"{m:.4f}+-{sd:.4f}    ", end="")
        print()

    # Key comparisons
    for pair in [("synap_only", "synap_only_bcast"),
                 ("neuro_only", "neuro_only_bcast"),
                 ("full", "full_bcast")]:
        base, treat = pair
        print()
        print("=" * 100)
        print(f"TREATMENT DELTAS: {treat} - {base} (negative = broadcast helps)")
        print("=" * 100)
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

    # Wins-per-seed scorecards
    for pair in [("synap_only", "synap_only_bcast"),
                 ("neuro_only", "neuro_only_bcast"),
                 ("full", "full_bcast")]:
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

    # Raw-data dump
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
                    "plasticity_gain_per_step":
                        results[s][v].plasticity_gain_per_step,
                } for v in variants
            } for s in SEEDS
        },
    }
    out_path = "research/developmental/results/env_sequence_v05_broadcast_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
