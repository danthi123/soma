"""v0.5 Direction 3 probe: neuromodulator-style plasticity broadcast.

Direction 1 (synap_pe) showed that the per-pair EMA was degenerate on
the 14-associator graph — every pair moves together, so the filter
effectively collapsed to a global PE-trend gate on synap admission.
It helped synap_only (7/8 regimes) but didn't compose with
neurogenesis.

Direction 3 generalizes that insight into an explicitly-global
broadcast: a scalar gain that rises on PE spikes and falls during
stable phases, multiplying BOTH synaptogenesis_rate AND hebbian_lr.
The hypothesis: during informative surprise, push plasticity harder;
during steady-state, suppress it so Hebbian doesn't reinforce ambient
noise.

Variants (matches plan doc's 2x2 plus no_growth baseline):

- ``no_growth``         reference baseline (no plasticity).
- ``full``              baseline with growth, broadcast off.
- ``full_bcast``        baseline growth + broadcast on.
- ``neuro_only``        reference (neuro is the one good mechanism).
- ``neuro_only_bcast``  does broadcast break neuro?
- ``synap_only``        for completeness; known to underperform.
- ``synap_only_bcast``  does broadcast rescue synap like synap_pe did?

7 variants, single seed=42. Multi-seed later if signal is positive.

Expected:
- full_bcast improves over full on 5+/8 regimes (plan success).
- neuro_only_bcast >= neuro_only (broadcast doesn't break the good
  mechanism).
- Bonus: synap_only_bcast matches or beats synap_only_pe result
  (7/8). If yes, broadcast subsumes Direction 1.

If null:
- Broadcast is too coarse; move to Direction 2 (learnable projs).
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
    plasticity_gain_per_step: list[float]


def build_config(dim: int, variant: str) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
        pruning_interval=0,
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
    variant: str, env: SequenceEnv, device: torch.device, dim: int,
) -> VariantResult:
    config = build_config(dim, variant)
    ps = PredictiveSOMA(config, device=device)
    mses: list[float] = []
    regimes: list[int] = []
    n_nodes: list[float] = []
    n_edges: list[float] = []
    gains: list[float] = []
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
        gains.append(float(ps.soma.plasticity_gain))
        if (step + 1) % report_every == 0:
            recent = mses[-100:] if len(mses) >= 100 else mses
            mean_mse = sum(recent) / len(recent)
            print(
                f"    {variant} step {step + 1}/{total} "
                f"regime={env.current_regime} "
                f"mse100={mean_mse:.4f} "
                f"nodes={int(n_nodes[-1])} edges={int(n_edges[-1])} "
                f"gain={gains[-1]:.3f}",
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
    gain_mean = sum(gains) / len(gains) if gains else 0.0
    gain_max = max(gains) if gains else 0.0
    gain_min = min(gains) if gains else 0.0
    print(
        f"  {variant} wall: {dt:.0f}s, "
        f"neuro={len(neuro_events)} synap={len(synap_events)} "
        f"gain avg={gain_mean:.3f} min={gain_min:.3f} max={gain_max:.3f}"
    )
    return VariantResult(
        variant, mses, regimes, n_nodes, n_edges, dt,
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
    schedule = make_capacity_schedule(dim=DIM, steps_per_regime=STEPS_PER_REGIME)
    env = SequenceEnv(dim=DIM, schedule=schedule, seed=42)
    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps: {schedule.total_steps}\n")

    variants = [
        "no_growth",
        "synap_only", "synap_only_bcast",
        "neuro_only", "neuro_only_bcast",
        "full", "full_bcast",
    ]
    results: dict[str, VariantResult] = {}
    for v in variants:
        print(f"=== {v} ===")
        results[v] = run_variant(v, env, device, DIM)
        print()

    print("=" * 130)
    print("REGIME MEAN MSE BY VARIANT (lower is better)")
    print("=" * 130)
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

    # Treatment deltas
    print("\nTREATMENT DELTAS (negative = broadcast helps):\n")
    for pair in [("synap_only", "synap_only_bcast"),
                 ("neuro_only", "neuro_only_bcast"),
                 ("full", "full_bcast")]:
        base, treat = pair
        print(f"  {treat} - {base}:")
        for regime_idx, regime in enumerate(schedule.regimes):
            b = regime_mean_mse(
                results[base].mse_per_step, results[base].regime_per_step
            ).get(regime_idx, 0.0)
            t = regime_mean_mse(
                results[treat].mse_per_step, results[treat].regime_per_step
            ).get(regime_idx, 0.0)
            print(f"    {regime.name:<12} delta={t - b:+.4f}")
        print()

    print("FINAL GRAPH / GAIN STATE:")
    for v in variants:
        r = results[v]
        print(
            f"  {v:<18} nodes={int(r.n_nodes[-1]):3d} "
            f"edges={int(r.n_edges[-1]):4d} "
            f"synap={len(r.synaptogenesis_event_steps):3d} "
            f"neuro={len(r.neurogenesis_event_steps):2d} "
            f"gain avg={sum(r.plasticity_gain_per_step)/len(r.plasticity_gain_per_step):.3f} "
            f"min={min(r.plasticity_gain_per_step):.3f} "
            f"max={max(r.plasticity_gain_per_step):.3f}"
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
                "plasticity_gain_per_step": results[v].plasticity_gain_per_step,
            } for v in results
        },
    }
    out_path = "research/developmental/results/env_sequence_v05_broadcast.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
