"""Tests for soma.schemas.builtin.code — code intelligence domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.code import CodeDecision, DependencyNote, Incident, Pattern

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_code_decision_registered(self) -> None:
        assert get_schema("code.decision") is CodeDecision

    def test_pattern_registered(self) -> None:
        assert get_schema("code.pattern") is Pattern

    def test_incident_registered(self) -> None:
        assert get_schema("code.incident") is Incident

    def test_dependency_note_registered(self) -> None:
        assert get_schema("code.dependency_note") is DependencyNote

    def test_all_code_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in [
            "code.decision",
            "code.pattern",
            "code.incident",
            "code.dependency_note",
        ]:
            assert name in names


# ── Round-trip: to_metadata / from_metadata ─────────────────────────


class TestRoundTrip:
    def test_code_decision_round_trip(self) -> None:
        d = CodeDecision(
            scope="api",
            decision="use REST not gRPC",
            rationale="simpler client story",
            alternatives="gRPC, GraphQL",
            files_affected="src/serve.py",
            decided_by="alice",
            decided_at="2026-04-16",
        )
        meta = d.to_metadata()
        assert meta["type"] == "code.decision"
        assert CodeDecision.from_metadata(meta) == d

    def test_code_decision_minimal(self) -> None:
        d = CodeDecision(scope="auth", decision="JWT over API key")
        meta = d.to_metadata()
        assert "revisit_after" not in meta  # None fields omitted
        assert CodeDecision.from_metadata(meta) == d

    def test_pattern_round_trip(self) -> None:
        p = Pattern(
            name="Repository",
            language="python",
            example_code="class UserRepo: ...",
            when_to_use="data access abstraction",
            when_not_to_use="simple scripts",
            files_using="src/repo.py,src/user_repo.py",
        )
        assert Pattern.from_metadata(p.to_metadata()) == p

    def test_incident_round_trip(self) -> None:
        inc = Incident(
            title="Login 500s after deploy",
            severity="p1",
            root_cause="missing migration",
            fix_summary="ran migrate, added CI check",
            commit="abc123",
            files_affected="auth.py,migrations/",
            prevention="pre-deploy migration lint",
        )
        assert Incident.from_metadata(inc.to_metadata()) == inc

    def test_dependency_note_round_trip(self) -> None:
        dn = DependencyNote(
            package="numpy",
            version="1.26.4",
            note="pinned due to ABI break in 2.0",
            risk="high",
            upgrade_blocked_by="scipy < 1.13",
        )
        assert DependencyNote.from_metadata(dn.to_metadata()) == dn

    def test_dependency_note_none_fields_omitted(self) -> None:
        dn = DependencyNote(package="requests", note="stable")
        meta = dn.to_metadata()
        assert "upgrade_blocked_by" not in meta


# ── Validation: choices ─────────────────────────────────────────────


class TestChoicesValidation:
    def test_incident_bad_severity(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            Incident(title="X", severity="p4")

    def test_incident_good_severities(self) -> None:
        for s in ["p0", "p1", "p2", "p3"]:
            Incident(title="X", severity=s)

    def test_dependency_note_bad_risk(self) -> None:
        with pytest.raises(ValueError, match="risk"):
            DependencyNote(package="x", note="y", risk="extreme")

    def test_dependency_note_good_risks(self) -> None:
        for r in ["low", "medium", "high", "critical"]:
            DependencyNote(package="x", note="y", risk=r)


# ── Searchable text extraction ──────────────────────────────────────


class TestSearchableText:
    def test_code_decision_search_text(self) -> None:
        d = CodeDecision(
            scope="api",
            decision="use REST",
            rationale="simpler",
        )
        assert d._search_text() == "api use REST simpler"

    def test_pattern_search_text(self) -> None:
        p = Pattern(
            name="Singleton",
            when_to_use="global state",
            when_not_to_use="stateless services",
        )
        assert p._search_text() == "Singleton global state stateless services"

    def test_incident_search_text(self) -> None:
        inc = Incident(
            title="OOM crash",
            severity="p0",
            root_cause="memory leak",
            fix_summary="added limit",
        )
        assert inc._search_text() == "OOM crash memory leak added limit"

    def test_dependency_note_search_text(self) -> None:
        dn = DependencyNote(
            package="torch",
            note="CUDA 12 only",
            upgrade_blocked_by="driver",
        )
        assert dn._search_text() == "torch CUDA 12 only driver"


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_code_decision_filterable(self) -> None:
        expected = {"scope", "decided_by"}
        assert CodeDecision._filterable_fields() == expected

    def test_pattern_filterable(self) -> None:
        expected = {"name", "language"}
        assert Pattern._filterable_fields() == expected

    def test_incident_filterable(self) -> None:
        expected = {"severity", "commit"}
        assert Incident._filterable_fields() == expected

    def test_dependency_note_filterable(self) -> None:
        expected = {"package", "version", "risk"}
        assert DependencyNote._filterable_fields() == expected
