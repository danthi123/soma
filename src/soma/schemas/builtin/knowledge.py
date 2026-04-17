"""Knowledge management domain schemas — notes, connections, questions, insights."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["Note", "Connection", "Question", "Insight"]


@schema("km.note")
class Note:
    """A knowledge note — manual, extracted, or imported."""

    title: str = field(searchable=True)
    source: str = field(
        filterable=True,
        choices=["manual", "extracted", "imported"],
        default="manual",
    )
    tags: str = field(filterable=True, default="")  # comma-separated
    project: str | None = field(default=None, filterable=True)
    url: str | None = field(default=None)


@schema("km.connection")
class Connection:
    """A typed link between two knowledge entries."""

    from_id: str = field(filterable=True)
    to_id: str = field(filterable=True)
    relationship: str = field(
        filterable=True,
        choices=["supports", "contradicts", "extends", "depends_on", "related"],
    )
    strength: float = field(default=0.5)


@schema("km.question")
class Question:
    """An open question, optionally linked to its answer."""

    question_text: str = field(searchable=True)
    status: str = field(
        filterable=True,
        choices=["open", "answered", "stale"],
        default="open",
    )
    answer_id: str | None = field(default=None, filterable=True)
    asked_by: str = field(default="")
    context: str = field(searchable=True, default="")


@schema("km.insight")
class Insight:
    """A synthesised claim with supporting evidence and confidence."""

    claim: str = field(searchable=True)
    supporting_evidence: str = field(default="")  # comma-separated node_ids
    confidence: float = field(default=0.5)
    domain: str = field(filterable=True, default="")
    derived_from: str = field(
        filterable=True,
        choices=["synthesis", "observation", "external"],
        default="observation",
    )
