"""v0.5 synap_pe — multi-seed validation of Phase 1 finding.

Phase 1 (single seed=42) showed:
- synap_only_pe beats synap_only on 7/8 regimes (plan criterion met).
- full_pe regresses vs full on 6/8 regimes (plan criterion missed).

Single-run GPU variance is ~0.001-0.002 MSE on the same config/seed
(empirically from comparing Phase 1, waiver v1, and waiver v2 runs).
The Phase 1 effect size is ~0.0005 MSE. So single-seed conclusions
rest on variance comparable to signal.

This runner executes all 5 variants across 3 seeds {0, 1, 42} and
reports:
- Mean regime MSE per variant (averaged across seeds).
- Stddev across seeds.
- Per-seed synap_only_pe - synap_only deltas.
- Per-seed full_pe - full deltas.

Waiver is left at default (off) — Phase 1.2 experiments showed the
waiver is effectively a no-op on this graph. Revisit later if
threshold/activation changes make per-pair EMAs actually
discriminative.
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
    if variant == "no_growth":
        overrides.update(synaptogenesis_interval=0, neurogenesis_interval=0)
    elif variant == "synap_only":
        overrides.update(neurogenesis_interval=0)
    elif variant == "synap_only_pe":
        overrides.update(
            neurogenesis_interval=0,
            synaptogenesis_supervision="pe_conditional",
        )
    elif variant == "full":
        pass
    elif variant == "full_pe":
        overrides.update(synaptogenesis_supervision="pe_conditional")
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
    variants = ["no_growth", "synap_only", "synap_only_pe", "full", "full_pe"]

    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps per variant: {schedule.total_steps}")
    print(f"Seeds: {SEEDS}")
    print(f"Variants: {variants}")
    print(f"Total runs: {len(SEEDS) * len(variants)} "
          f"= ~{len(SEEDS) * len(variants) * 6}min at ~6min/variant")
    print()

    results: dict[int, dict[str, RunResult]] = {s: {} for s in SEEDS}
    for seed in SEEDS:
        print(f"=== seed={seed} ===")
        for v in variants:
            t0 = time.perf_counter()
            r = run_variant(seed, v, env, device, DIM)
            results[seed][v] = r
            regime_means = regime_mean_mse(r.mse_per_step, r.regime_per_step)
            mean_over_regimes = sum(regime_means.values()) / len(regime_means)
            print(
                f"  seed={seed} {v:<16} "
                f"mean_mse={mean_over_regimes:.4f} "
                f"edges={int(r.n_edges[-1])} "
                f"synap={len(r.synaptogenesis_event_steps)} "
                f"neuro={len(r.neurogenesis_event_steps)} "
                f"wall={r.wall_time:.0f}s",
                flush=True,
            )
        print()

    # Per-regime mean ± stddev across seeds
    print("=" * 110)
    print("PER-REGIME MEAN MSE (MEAN +/- STDDEV across 3 seeds)")
    print("=" * 110)
    print(f"  {'Regime':<12}", end="")
    for v in variants:
        print(f"{v:<18}", end="")
    print()
    regime_names = [r.name for r in schedule.regimes]
    for regime_idx, regime in enumerate(regime_names):
        print(f"  {regime:<12}", end="")
        for v in variants:
            vals = []
            for s in SEEDS:
                r = results[s][v]
                means = regime_mean_mse(r.mse_per_step, r.regime_per_step)
                vals.append(means.get(regime_idx, 0.0))
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            print(f"{m:.4f}+-{sd:.4f}   ", end="")
        print()

    # Per-seed treatment deltas
    print()
    print("=" * 110)
    print("TREATMENT DELTAS: synap_only_pe - synap_only (per seed, negative = _pe helps)")
    print("=" * 110)
    print(f"  {'Regime':<12}", end="")
    for s in SEEDS:
        print(f"seed={s:<12}", end="")
    print(f"{'mean':<12}")
    for regime_idx, regime in enumerate(regime_names):
        print(f"  {regime:<12}", end="")
        deltas = []
        for s in SEEDS:
            base = regime_mean_mse(
                results[s]["synap_only"].mse_per_step,
                results[s]["synap_only"].regime_per_step,
            ).get(regime_idx, 0.0)
            treat = regime_mean_mse(
                results[s]["synap_only_pe"].mse_per_step,
                results[s]["synap_only_pe"].regime_per_step,
            ).get(regime_idx, 0.0)
            d = treat - base
            deltas.append(d)
            print(f"{d:+.4f}       ", end="")
        mean_d = sum(deltas) / len(deltas)
        print(f"{mean_d:+.4f}")

    print()
    print("=" * 110)
    print("TREATMENT DELTAS: full_pe - full (per seed, negative = _pe helps)")
    print("=" * 110)
    print(f"  {'Regime':<12}", end="")
    for s in SEEDS:
        print(f"seed={s:<12}", end="")
    print(f"{'mean':<12}")
    for regime_idx, regime in enumerate(regime_names):
        print(f"  {regime:<12}", end="")
        deltas = []
        for s in SEEDS:
            base = regime_mean_mse(
                results[s]["full"].mse_per_step,
                results[s]["full"].regime_per_step,
            ).get(regime_idx, 0.0)
            treat = regime_mean_mse(
                results[s]["full_pe"].mse_per_step,
                results[s]["full_pe"].regime_per_step,
            ).get(regime_idx, 0.0)
            d = treat - base
            deltas.append(d)
            print(f"{d:+.4f}       ", end="")
        mean_d = sum(deltas) / len(deltas)
        print(f"{mean_d:+.4f}")

    # Improvement counts across seeds
    print()
    print("Improvement scorecard (synap_only_pe beats synap_only per seed per regime):")
    for s in SEEDS:
        wins = 0
        total = 0
        for regime_idx in range(len(regime_names)):
            base = regime_mean_mse(
                results[s]["synap_only"].mse_per_step,
                results[s]["synap_only"].regime_per_step,
            ).get(regime_idx, 0.0)
            treat = regime_mean_mse(
                results[s]["synap_only_pe"].mse_per_step,
                results[s]["synap_only_pe"].regime_per_step,
            ).get(regime_idx, 0.0)
            if treat < base - 0.00005:  # 0.00005 = noise floor
                wins += 1
            total += 1
        print(f"  seed={s}: synap_only_pe wins {wins}/{total}")

    print()
    print("Improvement scorecard (full_pe beats full per seed per regime):")
    for s in SEEDS:
        wins = 0
        total = 0
        for regime_idx in range(len(regime_names)):
            base = regime_mean_mse(
                results[s]["full"].mse_per_step,
                results[s]["full"].regime_per_step,
            ).get(regime_idx, 0.0)
            treat = regime_mean_mse(
                results[s]["full_pe"].mse_per_step,
                results[s]["full_pe"].regime_per_step,
            ).get(regime_idx, 0.0)
            if treat < base - 0.00005:
                wins += 1
            total += 1
        print(f"  seed={s}: full_pe wins {wins}/{total}")

    # Save raw data
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
    out_path = "research/developmental/results/env_sequence_v05_synap_pe_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
