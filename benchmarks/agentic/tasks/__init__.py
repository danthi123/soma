"""Agentic benchmark tasks.

Each task implements the Task protocol from harness.py.
"""

from __future__ import annotations

from benchmarks.agentic.tasks.codebase_debug import CodebaseDebugTask
from benchmarks.agentic.tasks.context_overflow import ContextOverflowTask
from benchmarks.agentic.tasks.customer_support import CustomerSupportTask
from benchmarks.agentic.tasks.fact_recall import FactRecallTask
from benchmarks.agentic.tasks.multi_step_plan import MultiStepPlanTask
from benchmarks.agentic.tasks.research_assistant import ResearchAssistantTask
from benchmarks.agentic.tasks.session_resume import SessionResumeTask
from benchmarks.agentic.tasks.tool_learning import ToolLearningTask

TASK_REGISTRY: dict[str, type] = {
    "fact_recall": FactRecallTask,
    "tool_learning": ToolLearningTask,
    "context_overflow": ContextOverflowTask,
    "session_resume": SessionResumeTask,
    "multi_step_plan": MultiStepPlanTask,
    "codebase_debug": CodebaseDebugTask,
    "customer_support": CustomerSupportTask,
    "research_assistant": ResearchAssistantTask,
}

__all__ = [
    "TASK_REGISTRY",
    "CodebaseDebugTask",
    "ContextOverflowTask",
    "CustomerSupportTask",
    "FactRecallTask",
    "MultiStepPlanTask",
    "ResearchAssistantTask",
    "SessionResumeTask",
    "ToolLearningTask",
]
