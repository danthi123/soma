"""v0 runner for the sequence-prediction environment.

Drives three learners through the same regime schedule:
- PredictiveSOMA with full developmental mechanisms
- Online MLP baseline (same input/output dim, Adam)
- Frozen MLP baseline (trained during regime 0 only, then frozen)

Reports:
- Per-step prediction MSE
- Per-regime mean error (after warmup)
- Adaptation window after each boundary (steps to recover to
  within 20% of pre-switch baseline)
- Graph growth aligned to regimes (SOMA only)

This is the minimum viable test that the retrieval-ceiling paper's
central claim — "SOMA's mechanisms are for adaptation, not static
retrieval" — has legs. If SOMA adapts faster than the online MLP
after regime boundaries, we have a measurable developmental
advantage. If not, the scope-doc's stop criterion triggers.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.environments import SequenceEnv, make_default_schedule


@dataclass
class RunResult:
    name: str
    mse_per_step: list[float]
    regime_per_step: list[int]
    extra: dict[str, list[float]]


class OnlineMLP(nn.Module):
    """Next-obs prediction MLP, trained online."""

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
        """One online-prediction step. Returns MSE between pred and x_next."""
        self.train()
        pred = self.net(x_t)
        loss = torch.nn.functional.mse_loss(pred, x_next)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


class FrozenMLP(nn.Module):
    """Same architecture, trained on regime 0 only then frozen."""

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
        self.frozen = False

    def step(self, x_t: torch.Tensor, x_next: torch.Tensor) -> float:
        if not self.frozen:
            self.train()
            pred = self.net(x_t)
            loss = torch.nn.functional.mse_loss(pred, x_next)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            return loss.item()
        with torch.no_grad():
            pred = self.net(x_t)
            return torch.nn.functional.mse_loss(pred, x_next).item()

    def freeze(self) -> None:
        self.frozen = True
        for p in self.parameters():
            p.requires_grad = False


def run_soma(
    env: SequenceEnv,
    device: torch.device,
    dim: int,
) -> RunResult:
    config = SOMAConfig.developmental(
        sensor_output_dim=dim,
        text_embed_dim=dim,
        associator_input_dim=dim,
        associator_hidden_dim=dim * 2,
        associator_output_dim=dim,
        integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2,
        integrator_output_dim=dim,
    )
    ps = PredictiveSOMA(config, device=device)

    mses: list[float] = []
    regimes: list[int] = []
    n_nodes: list[float] = []
    n_edges: list[float] = []

    env.reset()
    prev_obs: torch.Tensor | None = None

    total = env.schedule.total_steps
    report_every = max(1, total // 20)
    t0 = time.perf_counter()
    for step in range(total):
        obs = env.step().to(device)
        result = ps.process_input(obs)

        # prediction_error is accumulated internally — we want the
        # per-step MSE measured against the current observation.
        mses.append(float(result["prediction_error"]))
        regimes.append(env.current_regime)
        n_nodes.append(float(result["num_nodes"]))
        n_edges.append(float(result["num_edges"]))

        prev_obs = obs

        if (step + 1) % report_every == 0:
            recent = mses[-100:] if len(mses) >= 100 else mses
            mean_mse = sum(recent) / len(recent)
            print(
                f"    SOMA step {step + 1}/{total} "
                f"regime={env.current_regime} "
                f"mse100={mean_mse:.4f} "
                f"nodes={int(n_nodes[-1])} edges={int(n_edges[-1])}",
                flush=True,
            )
    dt = time.perf_counter() - t0
    print(f"  SOMA wall: {dt:.1f}s")
    return RunResult(
        "soma", mses, regimes, {"n_nodes": n_nodes, "n_edges": n_edges}
    )


def run_online_mlp(
    env: SequenceEnv,
    device: torch.device,
    dim: int,
) -> RunResult:
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
    print(f"  OnlineMLP wall: {dt:.1f}s")
    return RunResult("online_mlp", mses, regimes, {})


def run_frozen_mlp(
    env: SequenceEnv,
    device: torch.device,
    dim: int,
) -> RunResult:
    net = FrozenMLP(dim).to(device)
    mses: list[float] = []
    regimes: list[int] = []

    env.reset()
    total = env.schedule.total_steps
    boundaries = env.schedule.boundaries

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
        # Freeze after regime 0 completes
        if (step + 1) == boundaries[0] and not net.frozen:
            net.freeze()
            print(f"    FrozenMLP frozen at step {step + 1}")
    dt = time.perf_counter() - t0
    print(f"  FrozenMLP wall: {dt:.1f}s")
    return RunResult("frozen_mlp", mses, regimes, {})


def adaptation_window(
    mses: list[float],
    boundaries: list[int],
    prebound_window: int = 200,
    max_window: int = 500,
    recovery_ratio: float = 1.2,
) -> list[int]:
    """For each regime-boundary, how many steps until the running MSE
    returns to within recovery_ratio x pre-boundary baseline.

    `max_window` caps the search so we don't scan forever. A return
    value of `max_window` means the learner did not recover within
    that window.
    """
    if not boundaries:
        return []
    out: list[int] = []
    # Skip the final boundary (it's the end of the trace)
    for b in boundaries[:-1]:
        start = max(0, b - prebound_window)
        if start >= b:
            out.append(max_window)
            continue
        pre = sum(mses[start:b]) / (b - start)
        target = pre * recovery_ratio

        found = max_window
        for i in range(min(max_window, len(mses) - b)):
            # Use a short running mean post-boundary
            lo = b + max(0, i - 20)
            hi = b + i + 1
            if hi <= lo:
                continue
            run_mean = sum(mses[lo:hi]) / (hi - lo)
            if run_mean <= target:
                found = i
                break
        out.append(found)
    return out


def regime_mean_mse(
    mses: list[float],
    regimes: list[int],
    warmup: int = 100,
) -> dict[int, float]:
    """Mean MSE per regime, excluding the first `warmup` steps of each."""
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
    schedule = make_default_schedule(dim=DIM, steps_per_regime=STEPS_PER_REGIME)
    env = SequenceEnv(dim=DIM, schedule=schedule, seed=42)

    print(f"Schedule: {[r.name for r in schedule.regimes]}")
    print(f"Total steps: {schedule.total_steps}")
    print(f"Boundaries: {schedule.boundaries}")

    print("\n=== SOMA ===")
    soma_result = run_soma(env, device, DIM)

    print("\n=== OnlineMLP ===")
    online_result = run_online_mlp(env, device, DIM)

    print("\n=== FrozenMLP (regime 0 only) ===")
    frozen_result = run_frozen_mlp(env, device, DIM)

    print("\n" + "=" * 70)
    print("REGIME MEAN MSE (after 100-step warmup per regime)")
    print("=" * 70)
    results = [soma_result, online_result, frozen_result]
    print(f"  {'Regime':<20}{'SOMA':<10}{'OnlineMLP':<12}{'FrozenMLP'}")
    for regime_idx, regime in enumerate(schedule.regimes):
        row = f"  {regime.name:<20}"
        for r in results:
            mean = regime_mean_mse(r.mse_per_step, r.regime_per_step).get(
                regime_idx, 0.0
            )
            row += f"{mean:<10.4f}" if r.name == "soma" else f"{mean:<12.4f}"
        print(row)

    print("\n" + "=" * 70)
    print("ADAPTATION WINDOWS (steps to recover to 1.2x pre-boundary MSE)")
    print("=" * 70)
    print(f"  {'Boundary':<20}{'SOMA':<10}{'OnlineMLP':<12}{'FrozenMLP'}")
    soma_aw = adaptation_window(soma_result.mse_per_step, schedule.boundaries)
    online_aw = adaptation_window(online_result.mse_per_step, schedule.boundaries)
    frozen_aw = adaptation_window(frozen_result.mse_per_step, schedule.boundaries)
    for i, b in enumerate(schedule.boundaries[:-1]):
        label = f"{schedule.regimes[i].name} -> {schedule.regimes[i+1].name}"
        print(
            f"  {label:<20}{soma_aw[i]:<10}{online_aw[i]:<12}{frozen_aw[i]}"
        )

    # Save raw data
    out = {
        "config": {
            "dim": DIM,
            "steps_per_regime": STEPS_PER_REGIME,
            "regimes": [r.name for r in schedule.regimes],
        },
        "soma": {
            "mse_per_step": soma_result.mse_per_step,
            "regime_per_step": soma_result.regime_per_step,
            "n_nodes": soma_result.extra["n_nodes"],
            "n_edges": soma_result.extra["n_edges"],
        },
        "online_mlp": {
            "mse_per_step": online_result.mse_per_step,
            "regime_per_step": online_result.regime_per_step,
        },
        "frozen_mlp": {
            "mse_per_step": frozen_result.mse_per_step,
            "regime_per_step": frozen_result.regime_per_step,
        },
        "schedule_boundaries": schedule.boundaries,
    }
    out_path = "research/developmental/results/env_sequence_v0.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
