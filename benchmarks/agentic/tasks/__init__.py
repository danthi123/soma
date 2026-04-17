"""Agentic benchmark tasks.

Each task implements the Task protocol from harness.py.
"""

from __future__ import annotations

from benchmarks.agentic.tasks.context_overflow import ContextOverflowTask
from benchmarks.agentic.tasks.fact_recall import FactRecallTask
from benchmarks.agentic.tasks.multi_step_plan import MultiStepPlanTask
from benchmarks.agentic.tasks.session_resume import SessionResumeTask
from benchmarks.agentic.tasks.tool_learning import ToolLearningTask

TASK_REGISTRY: dict[str, type] = {
    "fact_recall": FactRecallTask,
    "tool_learning": ToolLearningTask,
    "context_overflow": ContextOverflowTask,
    "session_resume": SessionResumeTask,
    "multi_step_plan": MultiStepPlanTask,
}

__all__ = [
    "TASK_REGISTRY",
    "FactRecallTask",
    "ToolLearningTask",
    "ContextOverflowTask",
    "SessionResumeTask",
    "MultiStepPlanTask",
]
