"""Conversational domain schemas — facts, preferences, contradictions."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["Fact", "Preference", "Contradiction"]


@schema("conv.fact")
class Fact:
    """A single subject-predicate-object triple extracted from conversation."""

    subject: str = field(filterable=True, searchable=True)
    predicate: str = field(searchable=True)
    object: str = field(searchable=True)
    confidence: float = field(default=1.0)
    source_turn_id: str | None = field(default=None, filterable=True)
    category: str = field(
        filterable=True,
        choices=[
            "identity",
            "preference",
            "relationship",
            "goal",
            "location",
            "other",
        ],
        default="other",
    )


@schema("conv.preference")
class Preference:
    """A user preference learned from conversation."""

    user_id: str = field(filterable=True)
    domain: str = field(filterable=True, searchable=True)
    preference: str = field(searchable=True)
    strength: float = field(default=0.5)
    last_confirmed: str = field(default="")


@schema("conv.contradiction")
class Contradiction:
    """Records two facts that contradict each other, with optional resolution."""

    fact_a_id: str = field(filterable=True)
    fact_b_id: str = field(filterable=True)
    resolution: str | None = field(default=None, searchable=True)
    resolved_at: str | None = field(default=None)
