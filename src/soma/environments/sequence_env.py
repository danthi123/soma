"""Synthetic sequence-prediction environment with regime changes.

Generates a stream of D-dimensional observations from a sequence of
latent dynamics "regimes." Each regime has its own transition rule.
At fixed step counts, the regime switches without warning — the
adaptation challenge is: how fast does the learner recover after a
boundary?

This is v0 of the open-ended learning environment. It tests SOMA's
core mechanisms (neurogenesis, consolidation, structural plasticity)
on a task where adaptation speed — not retrieval accuracy — is the
metric.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch


Regime = Callable[[torch.Tensor, torch.Generator], torch.Tensor]


def _bounded(x: torch.Tensor) -> torch.Tensor:
    """Keep observations in [-1, 1] to prevent magnitude drift across
    regimes. All regime step functions apply this so the observation
    space is stationary (only dynamics change across regimes).
    """
    return torch.tanh(x)


def _make_random_walk(dim: int, noise: float = 0.1) -> Regime:
    """x_{t+1} = tanh(x_t + N(0, noise))."""

    def step(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        noise_t = torch.randn(dim, generator=gen) * noise
        return _bounded(x + noise_t)

    return step


def _make_linear_rotation(
    dim: int,
    seed: int,
    noise: float = 0.1,
) -> Regime:
    """x_{t+1} = tanh(A x_t + N(0, noise)), A is a random orthogonal matrix."""
    gen = torch.Generator().manual_seed(seed)
    # Random orthogonal matrix via QR decomposition
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen))

    def step(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        noise_t = torch.randn(dim, generator=rng) * noise
        return _bounded(q @ x + noise_t)

    return step


def _make_nonlinear(dim: int, seed: int, noise: float = 0.1) -> Regime:
    """x_{t+1} = tanh(sign(x_t) * sqrt(|x_t|) + noise).

    Sqrt contracts large values, expands small ones; tanh bounds.
    """

    def step(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        noise_t = torch.randn(dim, generator=rng) * noise
        return _bounded(
            torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8) + noise_t
        )

    return step


def _make_mlp_dynamics(
    dim: int,
    seed: int,
    hidden: int = 32,
    noise: float = 0.1,
) -> Regime:
    """x_{t+1} = tanh(MLP(x_t) + noise).

    A fixed (frozen) shallow MLP maps x_t to x_{t+1}, creating
    nonlinear deterministic dynamics that the learner cannot exploit
    via linearity alone. Final tanh keeps observations bounded.
    """
    gen = torch.Generator().manual_seed(seed)
    w1 = torch.randn(dim, hidden, generator=gen) / (dim ** 0.5)
    b1 = torch.zeros(hidden)
    w2 = torch.randn(hidden, dim, generator=gen) / (hidden ** 0.5)
    b2 = torch.zeros(dim)

    def step(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        h = torch.tanh(x @ w1 + b1)
        y = h @ w2 + b2
        noise_t = torch.randn(dim, generator=rng) * noise
        return _bounded(y + noise_t)

    return step


@dataclass
class RegimeSpec:
    """A single regime in the schedule."""
    name: str
    step_fn: Regime
    duration: int  # steps this regime lasts


@dataclass
class RegimeSchedule:
    """An ordered sequence of regimes with durations."""
    regimes: list[RegimeSpec]

    @property
    def total_steps(self) -> int:
        return sum(r.duration for r in self.regimes)

    @property
    def boundaries(self) -> list[int]:
        """Step indices where regimes change (exclusive end of each)."""
        out: list[int] = []
        cum = 0
        for r in self.regimes:
            cum += r.duration
            out.append(cum)
        return out

    def regime_at(self, step: int) -> int:
        """Return the index of the regime active at the given step."""
        cum = 0
        for i, r in enumerate(self.regimes):
            cum += r.duration
            if step < cum:
                return i
        return len(self.regimes) - 1


def make_default_schedule(
    dim: int,
    steps_per_regime: int = 1500,
    noise: float = 0.1,
    seed: int = 42,
) -> RegimeSchedule:
    """Default 4-regime schedule used by v0 experiments.

    R1: random walk (simple drift)
    R2: linear rotation (periodic-structured dynamics)
    R3: elementwise sqrt (nonlinear fixed-point dynamics)
    R4: frozen-MLP dynamics (nonlinear high-capacity dynamics)
    """
    return RegimeSchedule(
        regimes=[
            RegimeSpec(
                "random_walk",
                _make_random_walk(dim, noise),
                steps_per_regime,
            ),
            RegimeSpec(
                "linear_rotation",
                _make_linear_rotation(dim, seed + 1, noise),
                steps_per_regime,
            ),
            RegimeSpec(
                "nonlinear_sqrt",
                _make_nonlinear(dim, seed + 2, noise),
                steps_per_regime,
            ),
            RegimeSpec(
                "mlp_dynamics",
                _make_mlp_dynamics(dim, seed + 3, noise=noise),
                steps_per_regime,
            ),
        ],
    )


@dataclass
class SequenceEnv:
    """A sequence-prediction environment driven by a RegimeSchedule.

    Use pattern:
        env = SequenceEnv(dim=16, schedule=make_default_schedule(16))
        for step in range(env.schedule.total_steps):
            obs = env.step()  # also advances internal state
            # use obs to train your learner
    """
    dim: int
    schedule: RegimeSchedule
    seed: int = 42
    _state: torch.Tensor = field(init=False)
    _step_count: int = field(default=0, init=False)
    _rng: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = torch.Generator().manual_seed(self.seed)
        # Initialize in the bounded [-1, 1] range so observations are
        # on the same scale from step 0 as all subsequent steps.
        self._state = torch.tanh(torch.randn(self.dim, generator=self._rng))

    @property
    def current_step(self) -> int:
        return self._step_count

    @property
    def current_regime(self) -> int:
        return self.schedule.regime_at(self._step_count)

    @property
    def current_regime_name(self) -> str:
        return self.schedule.regimes[self.current_regime].name

    def observe(self) -> torch.Tensor:
        """Return the current observation without advancing."""
        return self._state.clone()

    def step(self) -> torch.Tensor:
        """Advance one step; return the new observation."""
        regime_idx = self.current_regime
        step_fn = self.schedule.regimes[regime_idx].step_fn
        self._state = step_fn(self._state, self._rng)
        self._step_count += 1
        return self._state.clone()

    def reset(self) -> None:
        self._rng = torch.Generator().manual_seed(self.seed)
        self._state = torch.tanh(torch.randn(self.dim, generator=self._rng))
        self._step_count = 0
