"""Tests for soma.schemas.builtin.collab — collaboration domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.collab import (
    ActionItem,
    CollabDecision,
    FollowUp,
    StakeholderPosition,
)

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_action_item_registered(self) -> None:
        assert get_schema("collab.action_item") is ActionItem

    def test_collab_decision_registered(self) -> None:
        assert get_schema("collab.decision") is CollabDecision

    def test_follow_up_registered(self) -> None:
        assert get_schema("collab.follow_up") is FollowUp

    def test_stakeholder_position_registered(self) -> None:
        assert get_schema("collab.stakeholder_position") is StakeholderPosition

    def test_all_collab_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in [
            "collab.action_item",
            "collab.decision",
            "collab.follow_up",
            "collab.stakeholder_position",
        ]:
            assert name in names


# ── Round-trip: to_metadata / from_metadata ─────────────────────────


class TestRoundTrip:
    def test_action_item_round_trip(self) -> None:
        ai = ActionItem(
            assignee="alice",
            description="write migration script",
            status="open",
            due_date="2026-04-20",
            meeting_id="standup-42",
            context="discussed in retro",
        )
        meta = ai.to_metadata()
        assert meta["type"] == "collab.action_item"
        assert ActionItem.from_metadata(meta) == ai

    def test_action_item_minimal(self) -> None:
        ai = ActionItem(assignee="bob", description="review PR")
        assert ai.status == "open"
        assert ActionItem.from_metadata(ai.to_metadata()) == ai

    def test_collab_decision_round_trip(self) -> None:
        cd = CollabDecision(
            topic="API versioning",
            decision="use URL-path versioning",
            participants="alice,bob,carol",
            meeting_id="arch-review-7",
            dissent="bob preferred header versioning",
        )
        assert CollabDecision.from_metadata(cd.to_metadata()) == cd

    def test_collab_decision_none_fields_omitted(self) -> None:
        cd = CollabDecision(topic="X", decision="Y")
        meta = cd.to_metadata()
        assert "revisit_date" not in meta

    def test_follow_up_round_trip(self) -> None:
        fu = FollowUp(
            from_meeting="standup-41",
            to_meeting="standup-42",
            topic="migration timeline",
            status="addressed",
        )
        assert FollowUp.from_metadata(fu.to_metadata()) == fu

    def test_stakeholder_position_round_trip(self) -> None:
        sp = StakeholderPosition(
            person="carol",
            topic="cloud provider",
            position="prefers AWS over GCP",
            last_expressed="2026-04-15",
            confidence=0.8,
        )
        assert StakeholderPosition.from_metadata(sp.to_metadata()) == sp


# ── Validation: choices ─────────────────────────────────────────────


class TestChoicesValidation:
    def test_action_item_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            ActionItem(assignee="x", description="y", status="cancelled")

    def test_action_item_good_statuses(self) -> None:
        for s in ["open", "done", "dropped"]:
            ActionItem(assignee="x", description="y", status=s)

    def test_follow_up_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            FollowUp(from_meeting="m1", to_meeting="m2", topic="t", status="resolved")

    def test_follow_up_good_statuses(self) -> None:
        for s in ["pending", "addressed", "dropped"]:
            FollowUp(from_meeting="m1", to_meeting="m2", topic="t", status=s)


# ── Searchable text extraction ──────────────────────────────────────


class TestSearchableText:
    def test_action_item_search_text(self) -> None:
        ai = ActionItem(
            assignee="alice",
            description="fix the build",
            context="CI is red",
        )
        assert ai._search_text() == "alice fix the build CI is red"

    def test_collab_decision_search_text(self) -> None:
        cd = CollabDecision(
            topic="deploy strategy",
            decision="blue-green",
            dissent="canary was considered",
        )
        assert cd._search_text() == "deploy strategy blue-green canary was considered"

    def test_follow_up_search_text(self) -> None:
        fu = FollowUp(from_meeting="m1", to_meeting="m2", topic="latency budget")
        assert fu._search_text() == "latency budget"

    def test_stakeholder_position_search_text(self) -> None:
        sp = StakeholderPosition(
            person="dave", topic="pricing", position="should be usage-based"
        )
        assert sp._search_text() == "dave pricing should be usage-based"


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_action_item_filterable(self) -> None:
        expected = {"assignee", "status", "meeting_id"}
        assert ActionItem._filterable_fields() == expected

    def test_collab_decision_filterable(self) -> None:
        expected = {"meeting_id"}
        assert CollabDecision._filterable_fields() == expected

    def test_follow_up_filterable(self) -> None:
        expected = {"from_meeting", "to_meeting", "status"}
        assert FollowUp._filterable_fields() == expected

    def test_stakeholder_position_filterable(self) -> None:
        expected = {"person", "topic"}
        assert StakeholderPosition._filterable_fields() == expected
