"""Collaboration domain schemas — action items, decisions, follow-ups, positions."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["ActionItem", "CollabDecision", "FollowUp", "StakeholderPosition"]


@schema("collab.action_item")
class ActionItem:
    """A task assigned in a meeting or async discussion."""

    assignee: str = field(filterable=True, searchable=True)
    description: str = field(searchable=True)
    status: str = field(
        filterable=True,
        choices=["open", "done", "dropped"],
        default="open",
    )
    due_date: str = field(default="")
    meeting_id: str = field(filterable=True, default="")
    context: str = field(searchable=True, default="")


@schema("collab.decision")
class CollabDecision:
    """A decision reached by a group, with optional dissent and revisit date."""

    topic: str = field(searchable=True)
    decision: str = field(searchable=True)
    participants: str = field(default="")  # comma-separated
    meeting_id: str = field(filterable=True, default="")
    dissent: str = field(searchable=True, default="")
    revisit_date: str | None = field(default=None)


@schema("collab.follow_up")
class FollowUp:
    """Links two meetings via a topic that carries forward."""

    from_meeting: str = field(filterable=True)
    to_meeting: str = field(filterable=True)
    topic: str = field(searchable=True)
    status: str = field(
        filterable=True,
        choices=["pending", "addressed", "dropped"],
        default="pending",
    )


@schema("collab.stakeholder_position")
class StakeholderPosition:
    """A person's stance on a topic, useful for tracking alignment."""

    person: str = field(filterable=True, searchable=True)
    topic: str = field(filterable=True, searchable=True)
    position: str = field(searchable=True)
    last_expressed: str = field(default="")
    confidence: float = field(default=0.5)
