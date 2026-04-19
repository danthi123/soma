"""v0.5 Direction 2B — learnable input projections, multi-seed.

Directions 1 and 3 (PE-gated synap, broadcast) both failed multi-seed
validation — the PE-based signal was degenerate on this graph.

Direction 2B attacks the root cause directly: the input projections
that turn sensor input into per-node "receptive field" views are
frozen random matrices. Phase 2B (this experiment) makes them
learnable nn.Parameter and trains them with the prediction loss
(projections are included in the prediction-head optimizer; the
projected summary feeds the prediction head, so gradients flow
into projection weights).

Multi-seed from day 1. No single-seed confusion like Phase 1.

Variants (5):
- ``no_growth``              baseline, no plasticity.
- ``full_frozen``            growth + frozen random projections.
- ``full_learnable``         growth + learnable projections.
- ``neuro_only_frozen``      neuro only + frozen (the best baseline).
- ``neuro_only_learnable``   neuro only + learnable.

Seeds: {0, 1, 42}. 15 runs x ~5 min = ~75 min expected.

Success criteria (strict, per-seed):
- ``full_learnable`` beats ``full_frozen`` on 5+/8 regimes for ALL
  three seeds (not just mean).
- ``neuro_only_learnable`` does not regress vs ``neuro_only_frozen``.

If signal is positive: Direction 2B is validated; projection
learnability closes the random-projection semantic-gap.
If null: full Direction 2C (gradient through graph) is the next
bigger lift, OR we pivot to Direction 4 (LLM-as-teacher) which
requires a text task.
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
    projection_norm_initial: float
    projection_norm_final: float


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
    elif variant == "full_frozen":
        overrides.update(projection_mode="frozen_random")
    elif variant == "full_learnable":
        overrides.update(projection_mode="learnable")
    elif variant == "neuro_only_frozen":
        overrides.update(
            synaptogenesis_interval=0,
            projection_mode="frozen_random",
        )
    elif variant == "neuro_only_learnable":
        overrides.update(
            synaptogenesis_interval=0,
            projection_mode="learnable",
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def _mean_proj_norm(ps: PredictiveSOMA) -> float:
    if not ps._input_projections:
        return 0.0
    norms = [float(p.detach().norm().item()) for p in ps._input_projections.values()]
    return sum(norms) / len(norms)


def run_variant(
    seed: int, variant: str, env: SequenceEnv, device: torch.device, dim: int,
) -> RunResult:
    config = build_config(dim, variant, seed)
    ps = PredictiveSOMA(config, device=device)
    proj_norm_initial = _mean_proj_norm(ps)

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
    proj_norm_final = _mean_proj_norm(ps)
    return RunResult(
        seed, variant, mses, regimes, n_nodes, n_edges, dt,
        neuro_events, synap_events,
        proj_norm_initial, proj_norm_final,
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
        "full_frozen", "full_learnable",
        "neuro_only_frozen", "neuro_only_learnable",
    ]

    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps/variant: {schedule.total_steps}")
    print(f"Seeds: {SEEDS}, Variants: {variants}")
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
                f"  seed={seed} {v:<22} "
                f"mse={mean_over_regimes:.4f} "
                f"edges={int(r.n_edges[-1])} "
                f"synap={len(r.synaptogenesis_event_steps)} "
                f"neuro={len(r.neurogenesis_event_steps)} "
                f"proj[{r.projection_norm_initial:.2f}->{r.projection_norm_final:.2f}] "
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
        print(f"{v:<23}", end="")
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
            print(f"{m:.4f}+-{sd:.4f}      ", end="")
        print()

    for pair in [("full_frozen", "full_learnable"),
                 ("neuro_only_frozen", "neuro_only_learnable")]:
        base, treat = pair
        print()
        print("=" * 80)
        print(f"TREATMENT DELTAS: {treat} - {base} (negative = learnable helps)")
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

    for pair in [("full_frozen", "full_learnable"),
                 ("neuro_only_frozen", "neuro_only_learnable")]:
        base, treat = pair
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
                    "projection_norm_initial":
                        results[s][v].projection_norm_initial,
                    "projection_norm_final":
                        results[s][v].projection_norm_final,
                } for v in variants
            } for s in SEEDS
        },
    }
    out_path = (
        "research/developmental/results/env_sequence_v05_learnable_proj_multiseed.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
