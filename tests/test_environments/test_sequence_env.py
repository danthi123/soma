"""Tests for the sequence-prediction environment."""

from __future__ import annotations

import torch

from soma.environments import SequenceEnv, make_default_schedule


def test_schedule_total_steps_matches_sum() -> None:
    schedule = make_default_schedule(dim=4, steps_per_regime=100)
    assert schedule.total_steps == 400
    assert schedule.boundaries == [100, 200, 300, 400]


def test_schedule_regime_at_boundaries() -> None:
    schedule = make_default_schedule(dim=4, steps_per_regime=100)
    assert schedule.regime_at(0) == 0
    assert schedule.regime_at(99) == 0
    assert schedule.regime_at(100) == 1
    assert schedule.regime_at(250) == 2
    assert schedule.regime_at(399) == 3
    # Beyond total_steps, clamps to last regime
    assert schedule.regime_at(1000) == 3


def test_env_observe_does_not_advance() -> None:
    env = SequenceEnv(dim=4, schedule=make_default_schedule(4, 10), seed=0)
    obs1 = env.observe()
    obs2 = env.observe()
    assert torch.equal(obs1, obs2)
    assert env.current_step == 0


def test_env_step_advances_and_returns_new_state() -> None:
    env = SequenceEnv(dim=4, schedule=make_default_schedule(4, 10), seed=0)
    initial = env.observe().clone()
    after = env.step()
    assert not torch.equal(initial, after)
    assert env.current_step == 1


def test_env_current_regime_tracks_schedule() -> None:
    env = SequenceEnv(dim=4, schedule=make_default_schedule(4, 10), seed=0)
    assert env.current_regime == 0
    assert env.current_regime_name == "random_walk"
    for _ in range(10):
        env.step()
    assert env.current_regime == 1
    assert env.current_regime_name == "linear_rotation"


def test_env_reset() -> None:
    env = SequenceEnv(dim=4, schedule=make_default_schedule(4, 10), seed=42)
    initial_state = env.observe().clone()
    for _ in range(5):
        env.step()
    assert env.current_step == 5
    env.reset()
    assert env.current_step == 0
    # With the same seed, resetting should recover the same initial state
    assert torch.equal(env.observe(), initial_state)


def test_deterministic_under_same_seed() -> None:
    env1 = SequenceEnv(dim=8, schedule=make_default_schedule(8, 20), seed=7)
    env2 = SequenceEnv(dim=8, schedule=make_default_schedule(8, 20), seed=7)
    for _ in range(50):
        assert torch.equal(env1.step(), env2.step())


def test_regimes_produce_different_trajectories() -> None:
    """Sanity: two different regimes on the same seed produce different output."""
    torch.manual_seed(0)
    env = SequenceEnv(dim=8, schedule=make_default_schedule(8, 100), seed=0)
    # Collect trajectory in regime 0
    r0_trajectory = []
    for _ in range(100):
        r0_trajectory.append(env.step().clone())
    r0_trajectory_tensor = torch.stack(r0_trajectory)
    # Continue into regime 1
    r1_trajectory = []
    for _ in range(100):
        r1_trajectory.append(env.step().clone())
    r1_trajectory_tensor = torch.stack(r1_trajectory)
    # Mean trajectory magnitudes should differ (regimes have different dynamics)
    r0_std = r0_trajectory_tensor.std()
    r1_std = r1_trajectory_tensor.std()
    # Std should be measurably different (regimes have different distributions)
    assert not torch.isclose(r0_std, r1_std, atol=1e-3), (
        f"R0 std {r0_std:.4f} vs R1 std {r1_std:.4f} — regimes look identical"
    )


def test_observations_are_bounded() -> None:
    """All regimes apply tanh so observations stay in [-1, 1]."""
    env = SequenceEnv(dim=16, schedule=make_default_schedule(16, 200), seed=0)
    # Run through all regimes
    for _ in range(800):
        obs = env.step()
        assert torch.all(obs >= -1.0), f"observed <-1: min={obs.min().item()}"
        assert torch.all(obs <= 1.0), f"observed >1: max={obs.max().item()}"
