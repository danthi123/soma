"""v0.5 Direction 1 probe: PE-supervised synaptogenesis.

The 2x2 decomposition (env_sequence_v05_growth_2x2) showed that synap
*by itself* hurts on v0.5 capacity schedules, while neurogenesis helps
and partially rescues the synap damage. The working diagnosis: synap
admits edges between any co-active pair, but since SOMA's sensor
projections are random, co-activation is structurally diverse yet
semantically arbitrary. Direction 1 (grounding-plasticity plan)
proposes to filter admissions through a per-pair EMA of the
prediction-error delta observed when the pair is co-active — admit
only pairs whose co-activation has historically coincided with PE
going down.

Variants (single seed=42, same v0.5 capacity schedule as 2x2):

- ``no_growth``     syn off, neuro off — frozen baseline.
- ``synap_only``    syn on,  neuro off, supervision="none" —
                    reference arm, known to underperform no_growth.
- ``synap_only_pe`` syn on,  neuro off, supervision="pe_conditional" —
                    core Direction 1 test: does filtering the noise
                    rescue the synap-only regime?
- ``full``          syn on,  neuro on,  supervision="none" —
                    reproduces the v0.5 developmental default.
- ``full_pe``       syn on,  neuro on,  supervision="pe_conditional" —
                    does supervision help even when neurogenesis is
                    already providing useful structure?

Pruning is disabled in every variant (matches 2x2) so admissions
accumulate and the gate's effect is visible in final edge counts.

Expected signal if Direction 1 is correct:
- synap_only_pe MSE <= synap_only (should be closer to no_growth at
  minimum; ideally at or below full).
- Edge counts should be significantly lower under _pe variants,
  reflecting the filter rejecting arbitrary pairs.
- full_pe should match full or improve; not regress.

Null outcome (filter doesn't help):
- synap_only_pe behaves like no_growth (everything filtered) or like
  synap_only (filter too loose).
- full_pe matches full.

If null, the plan doc's Direction 3 (neuromodulator broadcast) becomes
the next-quickest probe.
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
    # Direction 1 introspection: final EMA state snapshot.
    final_pe_ema_mean: float
    final_pe_ema_count: int
    final_pe_ema_negative_fraction: float


def build_config(dim: int, variant: str) -> SOMAConfig:
    """Return the SOMAConfig for a given variant.

    Base config is SOMAConfig.developmental() with pruning disabled;
    individual variants flip synaptogenesis/neurogenesis intervals and
    the new supervision knob.
    """
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim,
        text_embed_dim=dim,
        associator_input_dim=dim,
        associator_hidden_dim=dim * 2,
        associator_output_dim=dim,
        integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2,
        integrator_output_dim=dim,
        pruning_interval=0,
    )
    if variant == "no_growth":
        overrides.update(synaptogenesis_interval=0, neurogenesis_interval=0)
    elif variant == "synap_only":
        overrides.update(neurogenesis_interval=0)
    elif variant == "synap_only_pe":
        overrides.update(
            neurogenesis_interval=0,
            synaptogenesis_supervision="pe_conditional",
            # Leave EMA / threshold / min_obs at defaults (0.99 / 0.0 / 5)
            # — probing whether the default gate is the right shape before
            # sweeping the hyperparameters.
        )
    elif variant == "full":
        pass  # developmental defaults, no supervision
    elif variant == "full_pe":
        overrides.update(synaptogenesis_supervision="pe_conditional")
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def _ema_summary(
    pe_ema: dict[tuple[str, str], float],
) -> tuple[float, int, float]:
    """Return (mean_ema, count, fraction_negative) for logging."""
    if not pe_ema:
        return 0.0, 0, 0.0
    values = list(pe_ema.values())
    mean = sum(values) / len(values)
    neg = sum(1 for v in values if v < 0.0)
    return mean, len(values), neg / len(values)


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
            # Snapshot EMA health mid-run too so we can see how
            # filtering behaves as the schedule progresses.
            ema_mean, ema_n, ema_neg = _ema_summary(ps.soma._synap_pe_ema)
            print(
                f"    {variant} step {step + 1}/{total} "
                f"regime={env.current_regime} "
                f"mse100={mean_mse:.4f} "
                f"nodes={int(n_nodes[-1])} edges={int(n_edges[-1])} "
                f"ema_pairs={ema_n} ema_mean={ema_mean:+.5f} "
                f"neg%={ema_neg * 100:.0f}",
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
    ema_mean, ema_n, ema_neg = _ema_summary(ps.soma._synap_pe_ema)
    print(
        f"  {variant} wall: {dt:.1f}s, "
        f"neuro={len(neuro_events)} synap={len(synap_events)} "
        f"ema_pairs={ema_n} neg%={ema_neg * 100:.0f}"
    )
    return VariantResult(
        variant, mses, regimes, n_nodes, n_edges, dt,
        neuro_events, synap_events,
        ema_mean, ema_n, ema_neg,
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

    variants = ["no_growth", "synap_only", "synap_only_pe", "full", "full_pe"]
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

    print("\n" + "=" * 110)
    print("FINAL GRAPH SIZE + GROWTH EVENT COUNTS + EMA STATE BY VARIANT")
    print("=" * 110)
    print(
        f"  {'Variant':<18}{'Nodes':<7}{'Edges':<7}"
        f"{'NeuroEv':<9}{'SynapEv':<9}{'EmaPairs':<10}"
        f"{'EmaMean':<12}{'Neg%':<7}{'Wall(s)':<10}"
    )
    for v in variants:
        r = results[v]
        print(
            f"  {v:<18}{int(r.n_nodes[-1]):<7}{int(r.n_edges[-1]):<7}"
            f"{len(r.neurogenesis_event_steps):<9}"
            f"{len(r.synaptogenesis_event_steps):<9}"
            f"{r.final_pe_ema_count:<10}"
            f"{r.final_pe_ema_mean:<+12.5f}"
            f"{r.final_pe_ema_negative_fraction * 100:<7.0f}"
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
                "final_pe_ema_mean": results[v].final_pe_ema_mean,
                "final_pe_ema_count": results[v].final_pe_ema_count,
                "final_pe_ema_negative_fraction":
                    results[v].final_pe_ema_negative_fraction,
            } for v in results
        },
        "schedule_boundaries": schedule.boundaries,
    }
    import sys
    # Allow overriding the output suffix so we can rerun under different
    # config tweaks (e.g., waiver on/off) without clobbering prior data.
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    out_path = (
        f"research/developmental/results/env_sequence_v05_synap_pe{suffix}.json"
    )
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
