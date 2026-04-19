"""v0.5 position-scramble control — does the SPECIFIC position matter?

If the rate-based sparsity control (commit e973f39) shows that
``synap_only_local`` beats a random low-rate variant at matched
total admissions, we know locality wins over sparsity. The next
question: is the benefit from the SPECIFIC initial positions, or
merely from ANY coherent distance metric in ANY random space?

Hypothesis 1 (specific positions matter): the random positions at
init happen to correlate with something substantive (e.g., the
simultaneous random draw that produced the node's projection
weights). Then filtering by position proximity tends to connect
projection-similar nodes, and those connections are useful.

Hypothesis 2 (any coherent metric works): the locality filter
simply restricts edge-addition to a spatially-coherent subset,
reducing the arbitrariness of which pairs wire. Any fixed random
metric would work equally well.

Test: run ``synap_only_local`` variants where positions have been
REPLACED with a fresh random draw immediately before training.
The scrambled positions preserve the DISTANCE DISTRIBUTION (same
jitter scale) but break any correlation between position and other
node properties.

If scrambled-position synap_local ~= original synap_local, H2 is
correct — any coherent metric works.
If scrambled-position synap_local << original synap_local (closer
to plain synap_only), H1 is correct — the original positions carry
signal.

Variants (5):
- ``no_growth``                     baseline.
- ``synap_only``                    reference harmful.
- ``synap_only_local``              reference positive (original positions).
- ``synap_only_local_scramble``     same locality filter, scrambled positions.
- ``neuro_only``                    reference positive.

Seeds {0, 1, 42}. 5 variants x 3 seeds = 15 runs, ~60-80 min.

Implementation note on scrambling: positions are scrambled via
``_scramble_positions`` helper which draws fresh ``randn * 0.1``
values for each existing node. Called once immediately after
PredictiveSOMA construction, before the first training step.
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


def _scramble_positions(ps: PredictiveSOMA, seed: int) -> None:
    """Replace each node's position with a fresh randn * 0.1 draw.

    Uses an INDEPENDENT seeded generator so the scramble doesn't
    interfere with subsequent rng-driven mechanisms.
    """
    gen = torch.Generator().manual_seed(10000 + seed)
    for node in ps.soma.graph.nodes.values():
        dim = node.position.shape[0]
        new_pos = torch.randn(dim, generator=gen) * 0.1
        # Preserve device + dtype.
        node.position = new_pos.to(node.position.device).to(node.position.dtype)


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
    elif variant in ("synap_only_local", "synap_only_local_scramble"):
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
    # Variant-specific state mutation: scramble positions if requested.
    if variant == "synap_only_local_scramble":
        _scramble_positions(ps, seed)
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
        "synap_only_local",
        "synap_only_local_scramble",
        "neuro_only",
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
                f"  seed={seed} {v:<30} "
                f"mse={mean_over_regimes:.4f} "
                f"edges={int(r.n_edges[-1])} "
                f"synap={len(r.synaptogenesis_event_steps)} "
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
        print(f"{v:<30}", end="")
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
            print(f"{m:.4f}+-{sd:.4f}             ", end="")
        print()

    pairs = [
        ("synap_only", "synap_only_local"),
        ("synap_only", "synap_only_local_scramble"),
        ("synap_only_local", "synap_only_local_scramble"),
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
        "env_sequence_v05_synap_position_scramble_multiseed.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
