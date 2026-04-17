"""Generic agent-loop runner for agentic benchmarks.

The harness is model-agnostic: it takes an Agent protocol and a Task protocol,
runs the loop, records per-step latency, and returns a RunResult.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from benchmarks.agentic.metrics import TaskResult


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class Agent(Protocol):
    """Anything that can observe-and-act in a loop."""

    def reset(self) -> None: ...
    def step(self, observation: str) -> str: ...
    def get_metrics(self) -> dict[str, Any]: ...


@runtime_checkable
class Task(Protocol):
    """A self-contained benchmark scenario."""

    def setup(self) -> str: ...
    def execute_action(self, action: str) -> str: ...
    def is_complete(self) -> bool: ...
    def score(self) -> TaskResult: ...


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Full result of a single harness run."""

    task_name: str
    agent_name: str
    model_name: str
    seed: int
    task_result: TaskResult
    step_latencies: list[float] = field(default_factory=list)
    total_steps: int = 0
    wall_clock_s: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEPS = 500


def run(
    agent: Agent,
    task: Task,
    *,
    task_name: str = "",
    agent_name: str = "",
    model_name: str = "",
    seed: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> RunResult:
    """Execute the agent-task loop and return metrics.

    Parameters
    ----------
    agent : Agent
        The agent under test.
    task : Task
        The benchmark task.
    max_steps : int
        Safety cap to prevent infinite loops.

    Returns
    -------
    RunResult
    """
    agent.reset()
    obs = task.setup()

    step_latencies: list[float] = []
    steps = 0
    wall_start = time.perf_counter()

    while not task.is_complete() and steps < max_steps:
        t0 = time.perf_counter()
        action = agent.step(obs)
        t1 = time.perf_counter()
        step_latencies.append(t1 - t0)

        obs = task.execute_action(action)
        steps += 1

    wall_clock = time.perf_counter() - wall_start
    task_result = task.score()
    # Overlay wall-clock from harness measurement
    task_result.wall_clock_s = wall_clock

    return RunResult(
        task_name=task_name,
        agent_name=agent_name,
        model_name=model_name,
        seed=seed,
        task_result=task_result,
        step_latencies=step_latencies,
        total_steps=steps,
        wall_clock_s=wall_clock,
    )
