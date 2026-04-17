"""Customer domain schemas — profiles, issues, sentiment, preferences."""

from __future__ import annotations

from soma.schemas import field, schema

__all__ = ["Profile", "CustomerIssue", "Sentiment", "CustomerPreference"]


@schema("customer.profile")
class Profile:
    """A customer profile with tier and contact history."""

    customer_id: str = field(filterable=True)
    name: str = field(searchable=True)
    company: str = field(filterable=True, searchable=True, default="")
    tier: str = field(
        filterable=True,
        choices=["free", "pro", "enterprise"],
        default="free",
    )
    first_seen: str = field(default="")
    last_contact: str = field(default="")


@schema("customer.issue")
class CustomerIssue:
    """A customer-reported issue with resolution tracking."""

    customer_id: str = field(filterable=True)
    category: str = field(filterable=True, searchable=True)
    description: str = field(searchable=True)
    status: str = field(
        filterable=True,
        choices=["open", "resolved", "escalated"],
        default="open",
    )
    resolution: str = field(searchable=True, default="")
    resolved_at: str | None = field(default=None)


@schema("customer.sentiment")
class Sentiment:
    """A sentiment signal from a customer interaction."""

    customer_id: str = field(filterable=True)
    score: float = field()  # -1.0 to 1.0
    signal: str = field(searchable=True)
    channel: str = field(
        filterable=True,
        choices=["support", "email", "social", "call"],
    )


@schema("customer.preference")
class CustomerPreference:
    """A customer preference, either explicitly stated or inferred."""

    customer_id: str = field(filterable=True)
    domain: str = field(filterable=True, searchable=True)
    preference: str = field(searchable=True)
    source: str = field(
        filterable=True,
        choices=["explicit", "inferred"],
        default="explicit",
    )
