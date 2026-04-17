"""Code intelligence domain schemas — decisions, patterns, incidents, dependency notes."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["CodeDecision", "Pattern", "Incident", "DependencyNote"]


@schema("code.decision")
class CodeDecision:
    """An architectural or implementation decision with rationale."""

    scope: str = field(filterable=True, searchable=True)
    decision: str = field(searchable=True)
    rationale: str = field(searchable=True, default="")
    alternatives: str = field(default="")
    files_affected: str = field(default="")  # comma-separated
    decided_by: str = field(filterable=True, default="")
    decided_at: str = field(default="")
    revisit_after: str | None = field(default=None)


@schema("code.pattern")
class Pattern:
    """A reusable code pattern with usage guidance."""

    name: str = field(filterable=True, searchable=True)
    language: str = field(filterable=True, default="")
    example_code: str = field(default="")
    when_to_use: str = field(searchable=True, default="")
    when_not_to_use: str = field(searchable=True, default="")
    files_using: str = field(default="")  # comma-separated


@schema("code.incident")
class Incident:
    """A production incident or significant bug with post-mortem details."""

    title: str = field(searchable=True)
    severity: str = field(
        filterable=True,
        choices=["p0", "p1", "p2", "p3"],
    )
    root_cause: str = field(searchable=True, default="")
    fix_summary: str = field(searchable=True, default="")
    commit: str = field(filterable=True, default="")
    files_affected: str = field(default="")  # comma-separated
    prevention: str = field(default="")


@schema("code.dependency_note")
class DependencyNote:
    """A note about a project dependency — risks, version pins, blockers."""

    package: str = field(filterable=True, searchable=True)
    note: str = field(searchable=True)
    version: str = field(filterable=True, default="")
    risk: str = field(
        filterable=True,
        choices=["low", "medium", "high", "critical"],
        default="low",
    )
    upgrade_blocked_by: str | None = field(default=None, searchable=True)
