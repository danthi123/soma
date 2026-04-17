"""Tests for soma.schemas.builtin.research — research domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.research import Experiment, Hypothesis, Literature, Result

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_hypothesis_registered(self) -> None:
        assert get_schema("research.hypothesis") is Hypothesis

    def test_experiment_registered(self) -> None:
        assert get_schema("research.experiment") is Experiment

    def test_result_registered(self) -> None:
        assert get_schema("research.result") is Result

    def test_literature_registered(self) -> None:
        assert get_schema("research.literature") is Literature

    def test_all_research_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in [
            "research.hypothesis",
            "research.experiment",
            "research.result",
            "research.literature",
        ]:
            assert name in names


# ── Round-trip: to_metadata / from_metadata ─────────────────────────


class TestRoundTrip:
    def test_hypothesis_round_trip(self) -> None:
        h = Hypothesis(
            claim="pruning improves latency by 30%",
            status="testing",
            confidence=0.7,
            domain="performance",
        )
        meta = h.to_metadata()
        assert meta["type"] == "research.hypothesis"
        assert Hypothesis.from_metadata(meta) == h

    def test_hypothesis_minimal(self) -> None:
        h = Hypothesis(claim="X causes Y")
        assert h.status == "proposed"
        assert h.confidence == 0.5
        assert Hypothesis.from_metadata(h.to_metadata()) == h

    def test_experiment_round_trip(self) -> None:
        e = Experiment(
            hypothesis_id="h1",
            method="A/B test with 1K users",
            parameters="n=1000,alpha=0.05",
            status="running",
            started_at="2026-04-16",
        )
        assert Experiment.from_metadata(e.to_metadata()) == e

    def test_experiment_none_fields_omitted(self) -> None:
        e = Experiment(hypothesis_id="h1", method="survey")
        meta = e.to_metadata()
        assert "completed_at" not in meta

    def test_result_round_trip(self) -> None:
        r = Result(
            experiment_id="e1",
            outcome="pass",
            key_metrics="p50=12ms, p95=34ms",
            interpretation="significant improvement",
            surprises="cache hit rate also rose",
        )
        assert Result.from_metadata(r.to_metadata()) == r

    def test_literature_round_trip(self) -> None:
        lit = Literature(
            title="Attention Is All You Need",
            authors="Vaswani et al.",
            year=2017,
            key_claim="self-attention replaces recurrence",
            relevance="foundational for transformer architectures",
            doi="10.48550/arXiv.1706.03762",
        )
        assert Literature.from_metadata(lit.to_metadata()) == lit

    def test_literature_none_doi_omitted(self) -> None:
        lit = Literature(title="Unpublished draft")
        meta = lit.to_metadata()
        assert "doi" not in meta


# ── Validation: choices ─────────────────────────────────────────────


class TestChoicesValidation:
    def test_hypothesis_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            Hypothesis(claim="X", status="wishful_thinking")

    def test_hypothesis_good_statuses(self) -> None:
        for s in ["proposed", "testing", "confirmed", "refuted", "inconclusive"]:
            Hypothesis(claim="X", status=s)

    def test_experiment_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            Experiment(hypothesis_id="h1", method="m", status="paused")

    def test_experiment_good_statuses(self) -> None:
        for s in ["planned", "running", "complete", "failed"]:
            Experiment(hypothesis_id="h1", method="m", status=s)

    def test_result_bad_outcome(self) -> None:
        with pytest.raises(ValueError, match="outcome"):
            Result(experiment_id="e1", outcome="maybe")

    def test_result_good_outcomes(self) -> None:
        for o in ["pass", "fail", "ambiguous"]:
            Result(experiment_id="e1", outcome=o)


# ── Searchable text extraction ──────────────────────────────────────


class TestSearchableText:
    def test_hypothesis_search_text(self) -> None:
        h = Hypothesis(claim="pruning helps", domain="perf")
        assert h._search_text() == "pruning helps perf"

    def test_experiment_search_text(self) -> None:
        e = Experiment(hypothesis_id="h1", method="benchmark suite")
        assert e._search_text() == "benchmark suite"

    def test_result_search_text(self) -> None:
        r = Result(
            experiment_id="e1",
            outcome="pass",
            key_metrics="p50=5ms",
            interpretation="fast enough",
            surprises="none",
        )
        assert r._search_text() == "p50=5ms fast enough none"

    def test_literature_search_text(self) -> None:
        lit = Literature(
            title="SOMA paper",
            authors="Smith",
            key_claim="local is better",
            relevance="core",
        )
        assert lit._search_text() == "SOMA paper Smith local is better core"


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_hypothesis_filterable(self) -> None:
        expected = {"status", "domain"}
        assert Hypothesis._filterable_fields() == expected

    def test_experiment_filterable(self) -> None:
        expected = {"hypothesis_id", "status"}
        assert Experiment._filterable_fields() == expected

    def test_result_filterable(self) -> None:
        expected = {"experiment_id", "outcome"}
        assert Result._filterable_fields() == expected

    def test_literature_filterable(self) -> None:
        expected = {"doi"}
        assert Literature._filterable_fields() == expected
