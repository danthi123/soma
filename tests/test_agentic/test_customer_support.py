"""Tests for the customer support task."""

from __future__ import annotations

import json

from benchmarks.agentic.tasks.customer_support import CustomerSupportTask


class TestCustomerSupport:
    def test_setup_returns_issue(self) -> None:
        task = CustomerSupportTask(seed=0)
        obs = task.setup()
        assert "customer" in obs.lower() or "issue" in obs.lower()
        assert "TK-" in obs
        assert not task.is_complete()

    def test_search_tickets(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "search_tickets", "arguments": {"query": "auth"}}'
        )
        data = json.loads(result)
        assert data["total"] >= 1
        ids = [r["id"] for r in data["results"]]
        assert "TK-1001" in ids

    def test_search_tickets_no_match(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "search_tickets", "arguments": {"query": "xyznotfound"}}'
        )
        data = json.loads(result)
        assert data["total"] == 0

    def test_get_ticket(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "get_ticket", "arguments": {"id": "TK-1001"}}'
        )
        data = json.loads(result)
        assert data["id"] == "TK-1001"
        assert "password" in data["body"].lower()
        # Should be tracked
        assert "TK-1001" in task._referenced_tickets

    def test_get_ticket_not_found(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "get_ticket", "arguments": {"id": "TK-9999"}}'
        )
        data = json.loads(result)
        assert "error" in data

    def test_reply_to_customer(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        result = task.execute_action(json.dumps({
            "tool": "reply_to_customer",
            "arguments": {
                "message": "Looking into TK-1003 for rate limit issue.",
            },
        }))
        assert "thank you" in result.lower() or "customer" in result.lower()
        assert "TK-1003" in task._referenced_tickets

    def test_escalate(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        result = task.execute_action(json.dumps({
            "tool": "escalate",
            "arguments": {"reason": "Complex multi-system issue"},
        }))
        assert task._escalated
        assert task.is_complete()

    def test_resolve(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        result = task.execute_action(json.dumps({
            "tool": "resolve",
            "arguments": {
                "solution": (
                    "Synced plan tier to rate limiter per TK-1003. "
                    "SSO SAML clock skew issue per TK-1004."
                ),
            },
        }))
        assert task._resolution_submitted
        assert task.is_complete()
        assert "TK-1003" in task._referenced_tickets
        assert "TK-1004" in task._referenced_tickets

    def test_scoring_defaults(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        score = task.score()
        assert score.accuracy == 0.0
        assert not score.completion

    def test_scoring_with_correct_refs(self) -> None:
        task = CustomerSupportTask(seed=0)
        task.setup()
        # Get all 3 expected tickets
        for tid in ["TK-1003", "TK-1004", "TK-1005"]:
            task.execute_action(json.dumps({
                "tool": "get_ticket",
                "arguments": {"id": tid},
            }))
        # Resolve with keywords
        task.execute_action(json.dumps({
            "tool": "resolve",
            "arguments": {
                "solution": (
                    "Sync plan tier and rate limit cache per TK-1003 "
                    "and TK-1005. SSO SAML NTP clock skew per TK-1004."
                ),
            },
        }))
        score = task.score()
        assert score.completion is True
        assert score.accuracy > 50.0
        assert score.extra["correct_refs"] == 3

    def test_scenario_variant_seed_1(self) -> None:
        task = CustomerSupportTask(seed=1)
        obs = task.setup()
        assert "TK-2" in obs
        result = task.execute_action(
            '{"tool": "search_tickets", "arguments": {"query": "export"}}'
        )
        data = json.loads(result)
        assert data["total"] >= 1

    def test_max_steps_terminates(self) -> None:
        task = CustomerSupportTask(seed=0)
        task._max_steps = 3
        task.setup()
        for _ in range(5):
            task.execute_action(
                '{"tool": "search_tickets", "arguments": {"query": "auth"}}'
            )
        assert task.is_complete()
