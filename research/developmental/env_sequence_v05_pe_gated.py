"""v0.5 extension: does PE-gated neurogenesis recover the plasticity benefit?

v0.5 (commit 0390590) found that `no_growth` beat `full` SOMA on
7 of 8 capacity regimes under the default interval-based
neurogenesis trigger. The `full` variant grew to 49 nodes / 980
edges but was 5-10x worse on prediction MSE than the frozen
14-node graph. The hypothesis: interval triggering fires on wall-
clock cadence regardless of whether prediction error is actually
elevated, so it adds random-weight noise.

This runner tests the alternative trigger mode shipped in
feat(developmental): PE-gated neurogenesis mode. It reuses the
same 8-regime capacity schedule (dim=16, 500 steps/regime) and
compares:

- full_interval   → default mode, reproduces the v0.5 result
- full_pe_gated   → opt-in pe_gated mode, cooldown=200
- no_growth       → reference variant (still wins under interval)
- online_mlp      → capacity-matched online MLP baseline

Reports per-regime MSE (warmup-excluded) and final graph size.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

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


class OnlineMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def step(self, x_t: torch.Tensor, x_next: torch.Tensor) -> float:
        pred = self.net(x_t)
        loss = torch.nn.functional.mse_loss(pred, x_next)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


def build_config(dim: int, variant: str) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
    )
    if variant == "no_growth":
        overrides.update(
            synaptogenesis_interval=0,
            neurogenesis_interval=0,
            pruning_interval=0,
        )
    elif variant == "full_interval":
        pass  # developmental() defaults: interval mode, interval=25
    elif variant == "full_pe_gated":
        overrides.update(
            neurogenesis_mode="pe_gated",
            neurogenesis_cooldown=200,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def run_variant(
    variant: str,
    env: SequenceEnv,
    device: torch.device,
    dim: int,
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
    # Pull neurogenesis event steps out of the growth log for analysis.
    event_steps = [
        int(e["step"])
        for e in ps.soma.growth_log
        if e.get("event_type") == "neurogenesis"
    ]
    print(f"  {variant} wall: {dt:.1f}s, neurogenesis_events={len(event_steps)}")
    return VariantResult(
        variant, mses, regimes, n_nodes, n_edges, dt, event_steps,
    )


def run_online_mlp(
    env: SequenceEnv, device: torch.device, dim: int,
) -> VariantResult:
    net = OnlineMLP(dim).to(device)
    mses: list[float] = []
    regimes: list[int] = []
    env.reset()
    total = env.schedule.total_steps
    prev_obs: torch.Tensor | None = None
    t0 = time.perf_counter()
    for step in range(total):
        obs = env.step().to(device)
        if prev_obs is not None:
            mse = net.step(prev_obs, obs)
            mses.append(mse)
        else:
            mses.append(0.0)
        regimes.append(env.current_regime)
        prev_obs = obs
    dt = time.perf_counter() - t0
    print(f"  online_mlp wall: {dt:.1f}s")
    return VariantResult(
        "online_mlp", mses, regimes, [0.0], [0.0], dt, [],
    )


def regime_mean_mse(
    mses: list[float],
    regimes: list[int],
    warmup: int = 100,
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
    schedule = make_capacity_schedule(
        dim=DIM, steps_per_regime=STEPS_PER_REGIME,
    )
    env = SequenceEnv(dim=DIM, schedule=schedule, seed=42)
    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps: {schedule.total_steps}\n")

    variants = ["full_interval", "full_pe_gated", "no_growth"]
    results: dict[str, VariantResult] = {}
    for v in variants:
        print(f"=== {v} ===")
        results[v] = run_variant(v, env, device, DIM)
        print()

    print("=== online_mlp ===")
    results["online_mlp"] = run_online_mlp(env, device, DIM)
    print()

    print("=" * 100)
    print("REGIME MEAN MSE BY VARIANT (lower is better)")
    print("=" * 100)
    variant_order = [
        "full_interval", "full_pe_gated", "no_growth", "online_mlp",
    ]
    print(f"  {'Regime':<12}", end="")
    for v in variant_order:
        print(f"{v:<16}", end="")
    print()
    for regime_idx, regime in enumerate(schedule.regimes):
        print(f"  {regime.name:<12}", end="")
        for v in variant_order:
            mean = regime_mean_mse(
                results[v].mse_per_step, results[v].regime_per_step,
            ).get(regime_idx, 0.0)
            print(f"{mean:<16.4f}", end="")
        print()

    print("\n" + "=" * 78)
    print("FINAL GRAPH SIZE + NEUROGENESIS EVENTS BY SOMA VARIANT")
    print("=" * 78)
    print(
        f"  {'Variant':<18}{'Nodes':<8}{'Edges':<8}"
        f"{'NeuroEvents':<14}{'Wall(s)':<10}"
    )
    soma_variants = ["full_interval", "full_pe_gated", "no_growth"]
    for v in soma_variants:
        r = results[v]
        print(
            f"  {v:<18}{int(r.n_nodes[-1]):<8}{int(r.n_edges[-1]):<8}"
            f"{len(r.neurogenesis_event_steps):<14}{r.wall_time:<10.1f}"
        )

    out = {
        "config": {
            "dim": DIM,
            "steps_per_regime": STEPS_PER_REGIME,
            "regimes": [r.name for r in schedule.regimes],
            "cooldown_steps": 200,
        },
        "variants": {
            v: {
                "mse_per_step": results[v].mse_per_step,
                "regime_per_step": results[v].regime_per_step,
                "n_nodes": results[v].n_nodes,
                "n_edges": results[v].n_edges,
                "wall_time": results[v].wall_time,
                "neurogenesis_event_steps": results[v].neurogenesis_event_steps,
            } for v in results
        },
        "schedule_boundaries": schedule.boundaries,
    }
    out_path = "research/developmental/results/env_sequence_v05_pe_gated.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
