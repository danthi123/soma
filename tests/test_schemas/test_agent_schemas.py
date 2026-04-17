"""Tests for soma.schemas.builtin.agent — agent domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.agent import Decision, Observation, TaskState, ToolCall

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_task_state_registered(self) -> None:
        assert get_schema("agent.task_state") is TaskState

    def test_tool_call_registered(self) -> None:
        assert get_schema("agent.tool_call") is ToolCall

    def test_observation_registered(self) -> None:
        assert get_schema("agent.observation") is Observation

    def test_decision_registered(self) -> None:
        assert get_schema("agent.decision") is Decision

    def test_all_agent_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in ["agent.task_state", "agent.tool_call", "agent.observation", "agent.decision"]:
            assert name in names


# ── Round-trip: to_metadata / from_metadata ─────────────────────────


class TestRoundTrip:
    def test_task_state_round_trip(self) -> None:
        ts = TaskState(task_id="t1", status="active", step=3, plan_summary="do stuff")
        meta = ts.to_metadata()
        assert meta["type"] == "agent.task_state"
        restored = TaskState.from_metadata(meta)
        assert restored == ts

    def test_tool_call_round_trip(self) -> None:
        tc = ToolCall(
            task_id="t1",
            tool_name="web_search",
            args='{"q": "hello"}',
            result_summary="found 3 results",
            success=True,
            latency_ms=120,
        )
        assert ToolCall.from_metadata(tc.to_metadata()) == tc

    def test_observation_round_trip(self) -> None:
        obs = Observation(task_id="t1", source="tool", content="file contains X")
        assert Observation.from_metadata(obs.to_metadata()) == obs

    def test_decision_round_trip(self) -> None:
        d = Decision(task_id="t1", choice="use approach A", rationale="faster")
        assert Decision.from_metadata(d.to_metadata()) == d

    def test_task_state_none_fields_omitted(self) -> None:
        ts = TaskState(task_id="t1", status="pending")
        meta = ts.to_metadata()
        assert "blocked_by" not in meta
        assert "parent_task_id" not in meta


# ── Validation: choices ─────────────────────────────────────────────


class TestChoicesValidation:
    def test_task_state_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            TaskState(task_id="t1", status="invalid")

    def test_task_state_good_statuses(self) -> None:
        for s in ["pending", "active", "blocked", "done", "failed"]:
            TaskState(task_id="t1", status=s)

    def test_observation_bad_source(self) -> None:
        with pytest.raises(ValueError, match="source"):
            Observation(task_id="t1", source="magic", content="x")

    def test_observation_good_sources(self) -> None:
        for s in ["tool", "user", "env", "internal"]:
            Observation(task_id="t1", source=s, content="x")


# ── Searchable text extraction ──────────────────────────────────────


class TestSearchableText:
    def test_task_state_search_text(self) -> None:
        ts = TaskState(task_id="t1", status="active", plan_summary="deploy the app")
        assert ts._search_text() == "deploy the app"

    def test_tool_call_search_text(self) -> None:
        tc = ToolCall(task_id="t1", tool_name="ls", result_summary="three files")
        assert tc._search_text() == "three files"

    def test_observation_search_text(self) -> None:
        obs = Observation(task_id="t1", source="user", content="I prefer dark mode")
        assert obs._search_text() == "I prefer dark mode"

    def test_decision_search_text(self) -> None:
        d = Decision(task_id="t1", choice="A", rationale="simpler")
        assert d._search_text() == "A simpler"

    def test_empty_searchable_defaults(self) -> None:
        ts = TaskState(task_id="t1", status="pending")
        # plan_summary defaults to "" -- empty but not None
        assert ts._search_text() == ""


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_task_state_filterable(self) -> None:
        expected = {"task_id", "status", "blocked_by", "parent_task_id"}
        assert TaskState._filterable_fields() == expected

    def test_tool_call_filterable(self) -> None:
        expected = {"task_id", "tool_name", "success", "error_type"}
        assert ToolCall._filterable_fields() == expected

    def test_observation_filterable(self) -> None:
        expected = {"task_id", "source"}
        assert Observation._filterable_fields() == expected

    def test_decision_filterable(self) -> None:
        assert Decision._filterable_fields() == {"task_id"}
