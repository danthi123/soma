"""Tests for soma.schemas.builtin.knowledge — knowledge management domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.knowledge import Connection, Insight, Note, Question

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_note_registered(self) -> None:
        assert get_schema("km.note") is Note

    def test_connection_registered(self) -> None:
        assert get_schema("km.connection") is Connection

    def test_question_registered(self) -> None:
        assert get_schema("km.question") is Question

    def test_insight_registered(self) -> None:
        assert get_schema("km.insight") is Insight

    def test_all_km_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in ["km.note", "km.connection", "km.question", "km.insight"]:
            assert name in names


# ── Round-trip ──────────────────────────────────────────────────────


class TestRoundTrip:
    def test_note_round_trip(self) -> None:
        n = Note(title="How to deploy SOMA")
        meta = n.to_metadata()
        assert meta["type"] == "km.note"
        assert Note.from_metadata(meta) == n

    def test_note_with_all_fields(self) -> None:
        n = Note(
            title="API design",
            source="extracted",
            tags="api,design",
            project="soma",
            url="https://example.com",
        )
        assert Note.from_metadata(n.to_metadata()) == n

    def test_connection_round_trip(self) -> None:
        c = Connection(from_id="n1", to_id="n2", relationship="supports", strength=0.9)
        assert Connection.from_metadata(c.to_metadata()) == c

    def test_question_round_trip(self) -> None:
        q = Question(question_text="What is SOMA?", context="reading docs")
        assert Question.from_metadata(q.to_metadata()) == q

    def test_question_answered(self) -> None:
        q = Question(
            question_text="How to prune?",
            status="answered",
            answer_id="a1",
            asked_by="alice",
        )
        assert Question.from_metadata(q.to_metadata()) == q

    def test_insight_round_trip(self) -> None:
        i = Insight(claim="pruning improves latency", confidence=0.8, domain="perf")
        assert Insight.from_metadata(i.to_metadata()) == i

    def test_note_none_fields_omitted(self) -> None:
        n = Note(title="Quick note")
        meta = n.to_metadata()
        assert "project" not in meta
        assert "url" not in meta


# ── Validation ──────────────────────────────────────────────────────


class TestChoicesValidation:
    def test_note_bad_source(self) -> None:
        with pytest.raises(ValueError, match="source"):
            Note(title="X", source="invented")

    def test_note_good_sources(self) -> None:
        for s in ["manual", "extracted", "imported"]:
            Note(title="X", source=s)

    def test_connection_bad_relationship(self) -> None:
        with pytest.raises(ValueError, match="relationship"):
            Connection(from_id="a", to_id="b", relationship="destroys")

    def test_connection_good_relationships(self) -> None:
        for r in ["supports", "contradicts", "extends", "depends_on", "related"]:
            Connection(from_id="a", to_id="b", relationship=r)

    def test_question_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            Question(question_text="Q?", status="unknown")

    def test_insight_bad_derived_from(self) -> None:
        with pytest.raises(ValueError, match="derived_from"):
            Insight(claim="X", derived_from="magic")

    def test_insight_good_derived_from(self) -> None:
        for d in ["synthesis", "observation", "external"]:
            Insight(claim="X", derived_from=d)


# ── Searchable text ─────────────────────────────────────────────────


class TestSearchableText:
    def test_note_search_text(self) -> None:
        n = Note(title="How to deploy SOMA")
        assert n._search_text() == "How to deploy SOMA"

    def test_question_search_text(self) -> None:
        q = Question(question_text="What is X?", context="reading about X")
        assert q._search_text() == "What is X? reading about X"

    def test_insight_search_text(self) -> None:
        i = Insight(claim="pruning helps")
        assert i._search_text() == "pruning helps"

    def test_connection_no_searchable(self) -> None:
        c = Connection(from_id="a", to_id="b", relationship="related")
        assert c._search_text() == ""


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_note_filterable(self) -> None:
        expected = {"source", "tags", "project"}
        assert Note._filterable_fields() == expected

    def test_connection_filterable(self) -> None:
        expected = {"from_id", "to_id", "relationship"}
        assert Connection._filterable_fields() == expected

    def test_question_filterable(self) -> None:
        expected = {"status", "answer_id"}
        assert Question._filterable_fields() == expected

    def test_insight_filterable(self) -> None:
        expected = {"domain", "derived_from"}
        assert Insight._filterable_fields() == expected
