"""Agentic benchmark agents."""

from __future__ import annotations

from benchmarks.agentic.agents.baseline import BaselineAgent
from benchmarks.agentic.agents.soma_agent import SomaAgent

AGENT_REGISTRY: dict[str, type] = {
    "baseline": BaselineAgent,
    "soma": SomaAgent,
}

__all__ = ["AGENT_REGISTRY", "BaselineAgent", "SomaAgent"]
