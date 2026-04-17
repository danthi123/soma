"""Tests for soma.schemas.builtin.conv — conversational domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.conv import Contradiction, Fact, Preference

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_fact_registered(self) -> None:
        assert get_schema("conv.fact") is Fact

    def test_preference_registered(self) -> None:
        assert get_schema("conv.preference") is Preference

    def test_contradiction_registered(self) -> None:
        assert get_schema("conv.contradiction") is Contradiction

    def test_all_conv_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in ["conv.fact", "conv.preference", "conv.contradiction"]:
            assert name in names


# ── Round-trip ──────────────────────────────────────────────────────


class TestRoundTrip:
    def test_fact_round_trip(self) -> None:
        f = Fact(subject="Alice", predicate="works at", object="Acme")
        meta = f.to_metadata()
        assert meta["type"] == "conv.fact"
        assert Fact.from_metadata(meta) == f

    def test_fact_with_category(self) -> None:
        f = Fact(
            subject="Bob",
            predicate="likes",
            object="pizza",
            category="preference",
            confidence=0.9,
        )
        assert Fact.from_metadata(f.to_metadata()) == f

    def test_preference_round_trip(self) -> None:
        p = Preference(
            user_id="u1",
            domain="code style",
            preference="tabs over spaces",
            strength=0.8,
        )
        assert Preference.from_metadata(p.to_metadata()) == p

    def test_contradiction_round_trip(self) -> None:
        c = Contradiction(fact_a_id="f1", fact_b_id="f2")
        assert Contradiction.from_metadata(c.to_metadata()) == c

    def test_contradiction_with_resolution(self) -> None:
        c = Contradiction(
            fact_a_id="f1",
            fact_b_id="f2",
            resolution="f2 supersedes f1",
            resolved_at="2026-04-16",
        )
        assert Contradiction.from_metadata(c.to_metadata()) == c

    def test_fact_none_fields_omitted(self) -> None:
        f = Fact(subject="X", predicate="is", object="Y")
        meta = f.to_metadata()
        assert "source_turn_id" not in meta


# ── Validation ──────────────────────────────────────────────────────


class TestChoicesValidation:
    def test_fact_bad_category(self) -> None:
        with pytest.raises(ValueError, match="category"):
            Fact(subject="X", predicate="is", object="Y", category="wrong")

    def test_fact_good_categories(self) -> None:
        for cat in ["identity", "preference", "relationship", "goal", "location", "other"]:
            Fact(subject="X", predicate="is", object="Y", category=cat)


# ── Searchable text ─────────────────────────────────────────────────


class TestSearchableText:
    def test_fact_search_text(self) -> None:
        f = Fact(subject="Alice", predicate="works at", object="Acme")
        assert f._search_text() == "Alice works at Acme"

    def test_preference_search_text(self) -> None:
        p = Preference(user_id="u1", domain="food", preference="vegan")
        assert p._search_text() == "food vegan"

    def test_contradiction_search_text_none_resolution(self) -> None:
        c = Contradiction(fact_a_id="f1", fact_b_id="f2")
        # resolution is None, so no searchable text
        assert c._search_text() == ""

    def test_contradiction_search_text_with_resolution(self) -> None:
        c = Contradiction(fact_a_id="f1", fact_b_id="f2", resolution="f2 wins")
        assert c._search_text() == "f2 wins"


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_fact_filterable(self) -> None:
        expected = {"subject", "source_turn_id", "category"}
        assert Fact._filterable_fields() == expected

    def test_preference_filterable(self) -> None:
        expected = {"user_id", "domain"}
        assert Preference._filterable_fields() == expected

    def test_contradiction_filterable(self) -> None:
        expected = {"fact_a_id", "fact_b_id"}
        assert Contradiction._filterable_fields() == expected
