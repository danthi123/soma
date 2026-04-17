"""Research domain schemas — hypotheses, experiments, results, literature."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["Hypothesis", "Experiment", "Result", "Literature"]


@schema("research.hypothesis")
class Hypothesis:
    """A research hypothesis with lifecycle status and confidence."""

    claim: str = field(searchable=True)
    status: str = field(
        filterable=True,
        choices=["proposed", "testing", "confirmed", "refuted", "inconclusive"],
        default="proposed",
    )
    confidence: float = field(default=0.5)
    domain: str = field(filterable=True, searchable=True, default="")
    depends_on: str | None = field(default=None)  # hypothesis id


@schema("research.experiment")
class Experiment:
    """A planned or running experiment tied to a hypothesis."""

    hypothesis_id: str = field(filterable=True)
    method: str = field(searchable=True)
    parameters: str = field(default="")  # JSON or comma-separated key=value
    status: str = field(
        filterable=True,
        choices=["planned", "running", "complete", "failed"],
        default="planned",
    )
    started_at: str = field(default="")
    completed_at: str | None = field(default=None)


@schema("research.result")
class Result:
    """The outcome of an experiment."""

    experiment_id: str = field(filterable=True)
    outcome: str = field(
        filterable=True,
        choices=["pass", "fail", "ambiguous"],
    )
    key_metrics: str = field(searchable=True, default="")
    interpretation: str = field(searchable=True, default="")
    surprises: str = field(searchable=True, default="")


@schema("research.literature")
class Literature:
    """A literature reference with relevance annotation."""

    title: str = field(searchable=True)
    authors: str = field(searchable=True, default="")
    year: int = field(default=0)
    key_claim: str = field(searchable=True, default="")
    relevance: str = field(searchable=True, default="")
    doi: str | None = field(default=None, filterable=True)
