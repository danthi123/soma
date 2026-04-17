"""Agentic benchmark agents."""

from __future__ import annotations

from benchmarks.agentic.agents.baseline import BaselineAgent

AGENT_REGISTRY: dict[str, type] = {
    "baseline": BaselineAgent,
}

__all__ = ["AGENT_REGISTRY", "BaselineAgent"]
