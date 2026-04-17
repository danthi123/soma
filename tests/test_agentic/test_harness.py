"""Tests for the agentic benchmark harness."""

from __future__ import annotations

from benchmarks.agentic.harness import RunResult, run
from benchmarks.agentic.metrics import TaskResult


# ---------------------------------------------------------------------------
# Stub Agent / Task
# ---------------------------------------------------------------------------
class StubAgent:
    def __init__(self) -> None:
        self._steps = 0

    def reset(self) -> None:
        self._steps = 0

    def step(self, observation: str) -> str:
        self._steps += 1
        return f"action-{self._steps}"

    def get_metrics(self) -> dict:
        return {"total_steps": self._steps}


class StubTask:
    """Completes after *limit* actions."""

    def __init__(self, limit: int = 3) -> None:
        self._limit = limit
        self._step = 0
        self._actions: list[str] = []

    def setup(self) -> str:
        self._step = 0
        self._actions = []
        return "initial-observation"

    def execute_action(self, action: str) -> str:
        self._actions.append(action)
        self._step += 1
        return f"obs-{self._step}"

    def is_complete(self) -> bool:
        return self._step >= self._limit

    def score(self) -> TaskResult:
        return TaskResult(
            completion=True,
            accuracy=100.0,
            steps=self._step,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_harness_terminates_on_completion() -> None:
    agent = StubAgent()
    task = StubTask(limit=3)
    result = run(agent, task, task_name="stub", max_steps=100)

    assert isinstance(result, RunResult)
    assert result.total_steps == 3
    assert result.task_result.completion is True
    assert result.task_result.accuracy == 100.0
    assert len(result.step_latencies) == 3


def test_harness_respects_max_steps() -> None:
    """Task never completes -> harness stops at max_steps."""

    class NeverDoneTask(StubTask):
        def is_complete(self) -> bool:
            return False

    agent = StubAgent()
    task = NeverDoneTask(limit=9999)
    result = run(agent, task, task_name="infinite", max_steps=5)

    assert result.total_steps == 5


def test_harness_resets_agent() -> None:
    agent = StubAgent()
    task = StubTask(limit=2)

    run(agent, task, max_steps=100)
    # Agent should have been reset at the start of run()
    # Run a second time and verify it still works
    result = run(agent, task, max_steps=100)
    assert result.total_steps == 2


def test_harness_records_wall_clock() -> None:
    agent = StubAgent()
    task = StubTask(limit=1)
    result = run(agent, task, max_steps=10)
    assert result.wall_clock_s > 0
    assert result.task_result.wall_clock_s > 0
