"""Tests for the research assistant task."""

from __future__ import annotations

import json

from benchmarks.agentic.tasks.research_assistant import ResearchAssistantTask


class TestResearchAssistant:
    def test_setup_returns_question(self) -> None:
        task = ResearchAssistantTask(seed=0)
        obs = task.setup()
        assert "Question 1" in obs
        assert "papers" in obs.lower()
        assert not task.is_complete()

    def test_search_papers(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "search_papers", "arguments": {"query": "scaling"}}'
        )
        data = json.loads(result)
        assert data["total"] >= 3
        ids = [r["id"] for r in data["results"]]
        assert "P001" in ids

    def test_search_papers_no_match(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "search_papers", "arguments": {"query": "xyznotfound"}}'
        )
        data = json.loads(result)
        assert data["total"] == 0

    def test_read_abstract(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "read_abstract", "arguments": {"paper_id": "P001"}}'
        )
        data = json.loads(result)
        assert data["id"] == "P001"
        assert "abstract" in data
        assert "P001" in task._abstracts_read

    def test_read_full(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "read_full", "arguments": {"paper_id": "P001"}}'
        )
        data = json.loads(result)
        assert "full_text" in data
        assert "findings" in data
        assert "P001" in task._papers_read

    def test_read_paper_not_found(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "read_abstract", "arguments": {"paper_id": "P999"}}'
        )
        data = json.loads(result)
        assert "error" in data

    def test_submit_finding_advances_question(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        result = task.execute_action(json.dumps({
            "tool": "submit_finding",
            "arguments": {
                "claim": (
                    "P001 recommends power-law scaling model size "
                    "faster, but P002 contradicts this saying scale "
                    "equally and compute-optimal."
                ),
                "evidence": ["P001", "P002"],
            },
        }))
        assert "F1" in result
        assert "Question 2" in result
        assert task._current_question_idx == 1

    def test_revise_finding(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        # Submit initial finding
        task.execute_action(json.dumps({
            "tool": "submit_finding",
            "arguments": {
                "claim": "Initial claim",
                "evidence": ["P001"],
            },
        }))
        # Revise it
        result = task.execute_action(json.dumps({
            "tool": "revise_finding",
            "arguments": {
                "finding_id": "F1",
                "new_claim": "Revised claim about scaling",
                "new_evidence": ["P001", "P002"],
            },
        }))
        assert "revised" in result.lower()
        assert task._revision_count == 1
        assert task._findings["F1"]["claim"] == "Revised claim about scaling"

    def test_revise_nonexistent_finding(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        result = task.execute_action(json.dumps({
            "tool": "revise_finding",
            "arguments": {
                "finding_id": "F999",
                "new_claim": "nope",
                "new_evidence": [],
            },
        }))
        assert "not found" in result.lower()

    def test_scoring_defaults(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        score = task.score()
        assert score.accuracy == 0.0
        assert not score.completion

    def test_complete_all_questions(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task.setup()
        # Submit 5 findings with correct keywords and evidence
        answers = [
            {
                "claim": (
                    "P001 says power-law scaling equally but P002 "
                    "contradicts saying compute-optimal scale equally"
                ),
                "evidence": ["P001", "P002"],
            },
            {
                "claim": (
                    "P003 claims emergent abilities at thresholds "
                    "but P004 shows metric artifact with continuous "
                    "metrics - contradicts the emergence claim"
                ),
                "evidence": ["P003", "P004"],
            },
            {
                "claim": (
                    "RLHF from P005 works, DPO from P009 needs no "
                    "reward model with less compute, constitutional "
                    "AI from P008 reduces annotation"
                ),
                "evidence": ["P005", "P008", "P009"],
            },
            {
                "claim": (
                    "Distillation P007, RAG P010, tool-use P011, and "
                    "test-time compute P016 enable augmentation of "
                    "small models to match large"
                ),
                "evidence": ["P007", "P010", "P011", "P016"],
            },
            {
                "claim": (
                    "LoRA P013 with merging P018 for efficient "
                    "deployment, MoE P006 for sparse scaling, "
                    "alignment tax minimal per P019"
                ),
                "evidence": ["P006", "P013", "P018", "P019"],
            },
        ]
        for ans in answers:
            task.execute_action(json.dumps({
                "tool": "submit_finding",
                "arguments": ans,
            }))
        assert task.is_complete()
        score = task.score()
        assert score.completion is True
        assert score.accuracy > 50.0
        assert score.extra["findings_submitted"] == 5

    def test_scenario_variant_seed_1(self) -> None:
        task = ResearchAssistantTask(seed=1)
        obs = task.setup()
        assert "Question 1" in obs
        # Scenario 1 has medical papers
        result = task.execute_action(
            '{"tool": "search_papers", "arguments": {"query": "vitamin"}}'
        )
        data = json.loads(result)
        assert data["total"] >= 1

    def test_max_steps_terminates(self) -> None:
        task = ResearchAssistantTask(seed=0)
        task._max_steps = 3
        task.setup()
        for _ in range(5):
            task.execute_action(
                '{"tool": "search_papers", "arguments": {"query": "scaling"}}'
            )
        assert task.is_complete()
