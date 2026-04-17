"""Tests for soma.schemas.builtin.customer — customer domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.customer import (
    CustomerIssue,
    CustomerPreference,
    Profile,
    Sentiment,
)

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_profile_registered(self) -> None:
        assert get_schema("customer.profile") is Profile

    def test_customer_issue_registered(self) -> None:
        assert get_schema("customer.issue") is CustomerIssue

    def test_sentiment_registered(self) -> None:
        assert get_schema("customer.sentiment") is Sentiment

    def test_customer_preference_registered(self) -> None:
        assert get_schema("customer.preference") is CustomerPreference

    def test_all_customer_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in [
            "customer.profile",
            "customer.issue",
            "customer.sentiment",
            "customer.preference",
        ]:
            assert name in names


# ── Round-trip: to_metadata / from_metadata ─────────────────────────


class TestRoundTrip:
    def test_profile_round_trip(self) -> None:
        p = Profile(
            customer_id="c-42",
            name="Acme Corp",
            company="Acme",
            tier="enterprise",
            first_seen="2025-01-01",
            last_contact="2026-04-16",
        )
        meta = p.to_metadata()
        assert meta["type"] == "customer.profile"
        assert Profile.from_metadata(meta) == p

    def test_profile_minimal(self) -> None:
        p = Profile(customer_id="c-1", name="Alice")
        assert p.tier == "free"
        assert Profile.from_metadata(p.to_metadata()) == p

    def test_customer_issue_round_trip(self) -> None:
        ci = CustomerIssue(
            customer_id="c-42",
            category="billing",
            description="double charged for March",
            status="resolved",
            resolution="refund issued",
            resolved_at="2026-04-10",
        )
        assert CustomerIssue.from_metadata(ci.to_metadata()) == ci

    def test_customer_issue_none_fields_omitted(self) -> None:
        ci = CustomerIssue(
            customer_id="c-1", category="auth", description="cannot login"
        )
        meta = ci.to_metadata()
        assert "resolved_at" not in meta

    def test_sentiment_round_trip(self) -> None:
        s = Sentiment(
            customer_id="c-42",
            score=0.8,
            signal="praised onboarding flow",
            channel="support",
        )
        assert Sentiment.from_metadata(s.to_metadata()) == s

    def test_sentiment_negative(self) -> None:
        s = Sentiment(
            customer_id="c-1",
            score=-0.7,
            signal="frustrated with latency",
            channel="call",
        )
        assert Sentiment.from_metadata(s.to_metadata()) == s

    def test_customer_preference_round_trip(self) -> None:
        cp = CustomerPreference(
            customer_id="c-42",
            domain="communication",
            preference="prefers email over phone",
            source="explicit",
        )
        assert CustomerPreference.from_metadata(cp.to_metadata()) == cp


# ── Validation: choices ─────────────────────────────────────────────


class TestChoicesValidation:
    def test_profile_bad_tier(self) -> None:
        with pytest.raises(ValueError, match="tier"):
            Profile(customer_id="c-1", name="X", tier="premium")

    def test_profile_good_tiers(self) -> None:
        for t in ["free", "pro", "enterprise"]:
            Profile(customer_id="c-1", name="X", tier=t)

    def test_customer_issue_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            CustomerIssue(
                customer_id="c-1", category="x", description="y", status="closed"
            )

    def test_customer_issue_good_statuses(self) -> None:
        for s in ["open", "resolved", "escalated"]:
            CustomerIssue(customer_id="c-1", category="x", description="y", status=s)

    def test_sentiment_bad_channel(self) -> None:
        with pytest.raises(ValueError, match="channel"):
            Sentiment(customer_id="c-1", score=0.5, signal="x", channel="slack")

    def test_sentiment_good_channels(self) -> None:
        for ch in ["support", "email", "social", "call"]:
            Sentiment(customer_id="c-1", score=0.5, signal="x", channel=ch)

    def test_customer_preference_bad_source(self) -> None:
        with pytest.raises(ValueError, match="source"):
            CustomerPreference(
                customer_id="c-1", domain="x", preference="y", source="guessed"
            )

    def test_customer_preference_good_sources(self) -> None:
        for s in ["explicit", "inferred"]:
            CustomerPreference(customer_id="c-1", domain="x", preference="y", source=s)


# ── Searchable text extraction ──────────────────────────────────────


class TestSearchableText:
    def test_profile_search_text(self) -> None:
        p = Profile(customer_id="c-1", name="Alice", company="Acme")
        assert p._search_text() == "Alice Acme"

    def test_customer_issue_search_text(self) -> None:
        ci = CustomerIssue(
            customer_id="c-1",
            category="billing",
            description="overcharged",
            resolution="refunded",
        )
        assert ci._search_text() == "billing overcharged refunded"

    def test_sentiment_search_text(self) -> None:
        s = Sentiment(
            customer_id="c-1", score=0.9, signal="loves the product", channel="email"
        )
        assert s._search_text() == "loves the product"

    def test_customer_preference_search_text(self) -> None:
        cp = CustomerPreference(
            customer_id="c-1", domain="UI", preference="dark mode"
        )
        assert cp._search_text() == "UI dark mode"


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_profile_filterable(self) -> None:
        expected = {"customer_id", "company", "tier"}
        assert Profile._filterable_fields() == expected

    def test_customer_issue_filterable(self) -> None:
        expected = {"customer_id", "category", "status"}
        assert CustomerIssue._filterable_fields() == expected

    def test_sentiment_filterable(self) -> None:
        expected = {"customer_id", "channel"}
        assert Sentiment._filterable_fields() == expected

    def test_customer_preference_filterable(self) -> None:
        expected = {"customer_id", "domain", "source"}
        assert CustomerPreference._filterable_fields() == expected
