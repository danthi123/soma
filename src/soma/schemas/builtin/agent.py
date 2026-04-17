"""Agent domain schemas — task lifecycle, tool calls, observations, decisions."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["TaskState", "ToolCall", "Observation", "Decision"]


@schema("agent.task_state")
class TaskState:
    """Tracks the lifecycle of an agent task."""

    task_id: str = field(filterable=True)
    status: str = field(
        filterable=True,
        choices=["pending", "active", "blocked", "done", "failed"],
    )
    step: int = field(default=0)
    plan_summary: str = field(searchable=True, default="")
    blocked_by: str | None = field(default=None, filterable=True)
    parent_task_id: str | None = field(default=None, filterable=True)


@schema("agent.tool_call")
class ToolCall:
    """Records a single tool invocation and its outcome."""

    task_id: str = field(filterable=True)
    tool_name: str = field(filterable=True)
    args: str = field(default="")
    result_summary: str = field(searchable=True, default="")
    success: bool = field(filterable=True, default=True)
    latency_ms: int = field(default=0)
    error_type: str | None = field(default=None, filterable=True)


@schema("agent.observation")
class Observation:
    """An observation gathered during task execution."""

    task_id: str = field(filterable=True)
    source: str = field(
        filterable=True,
        choices=["tool", "user", "env", "internal"],
    )
    content: str = field(searchable=True)
    confidence: float = field(default=1.0)
    supersedes: str | None = field(default=None)


@schema("agent.decision")
class Decision:
    """A decision made by the agent, with rationale."""

    task_id: str = field(filterable=True)
    choice: str = field(searchable=True)
    alternatives: str = field(default="")
    rationale: str = field(searchable=True, default="")
