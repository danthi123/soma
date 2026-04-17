"""Tests for all 5 agentic benchmark tasks.

Each test exercises scoring logic with known-good actions.
"""

from __future__ import annotations

import json

from benchmarks.agentic.tasks.context_overflow import ContextOverflowTask
from benchmarks.agentic.tasks.fact_recall import FactRecallTask
from benchmarks.agentic.tasks.multi_step_plan import MultiStepPlanTask
from benchmarks.agentic.tasks.session_resume import SessionResumeTask
from benchmarks.agentic.tasks.tool_learning import ToolLearningTask


# -----------------------------------------------------------------------
# Task 1: Fact recall
# -----------------------------------------------------------------------
class TestFactRecall:
    def test_setup_returns_observation(self) -> None:
        task = FactRecallTask(seed=0)
        obs = task.setup()
        assert "Alice Chen" in obs  # first fact for seed 0

    def test_scoring_perfect(self) -> None:
        task = FactRecallTask(seed=0)
        task.setup()

        # Walk through all turns answering anything
        while not task.is_complete():
            task.execute_action("Sure, I remember!")
        # The last 5 turns are recall questions -- we answered generically
        # so 0 correct is expected
        result = task.score()
        assert result.accuracy == 0.0
        assert result.steps > 0

    def test_scoring_correct_answers(self) -> None:
        task = FactRecallTask(seed=0)
        task.setup()

        # Walk through filler turns
        turn = 0
        total_turns = len(task._script)
        recall_start = total_turns - 5  # last 5 are recall

        while turn < total_turns:
            if task.is_complete():
                break
            if turn >= recall_start:
                # Answer with the correct fact
                q_idx = turn - recall_start
                facts = list(task._facts.values())
                action = f"The answer is {facts[q_idx]}"
            else:
                action = "That's interesting, tell me more."
            task.execute_action(action)
            turn += 1

        result = task.score()
        assert result.accuracy == 100.0
        assert result.completion is True

    def test_is_complete_at_end(self) -> None:
        task = FactRecallTask(seed=0)
        task.setup()
        assert not task.is_complete()


# -----------------------------------------------------------------------
# Task 2: Tool learning
# -----------------------------------------------------------------------
class TestToolLearning:
    def test_valid_search(self) -> None:
        task = ToolLearningTask(seed=0)
        task.setup()
        action = json.dumps({"tool": "search_database", "arguments": {"query": "quantum"}})
        result = task.execute_action(action)
        data = json.loads(result)
        assert "results" in data
        assert data["total"] >= 1

    def test_uppercase_error(self) -> None:
        task = ToolLearningTask(seed=0)
        task.setup()
        action = json.dumps({"tool": "search_database", "arguments": {"query": "Quantum"}})
        result = task.execute_action(action)
        data = json.loads(result)
        assert data["error"] == "QUERY_FORMAT_ERROR"
        assert "lowercase" in data["message"]

    def test_too_many_words_error(self) -> None:
        task = ToolLearningTask(seed=0)
        task.setup()
        action = json.dumps({
            "tool": "search_database",
            "arguments": {"query": "this has too many words"},
        })
        result = task.execute_action(action)
        data = json.loads(result)
        assert data["error"] == "QUERY_FORMAT_ERROR"
        assert "3 words" in data["message"]

    def test_submit_results(self) -> None:
        task = ToolLearningTask(seed=0)
        task.setup()
        action = json.dumps({
            "tool": "submit_results",
            "arguments": {"results": ["quantum entanglement basics"]},
        })
        task.execute_action(action)
        assert task.is_complete()

    def test_scoring(self) -> None:
        task = ToolLearningTask(seed=0)
        task.setup()
        # One valid search, then submit
        search = {"tool": "search_database", "arguments": {"query": "quantum"}}
        task.execute_action(json.dumps(search))
        task.execute_action(json.dumps({"tool": "submit_results", "arguments": {"results": []}}))
        result = task.score()
        assert result.steps == 2
        assert result.tool_errors == 0


# -----------------------------------------------------------------------
# Task 3: Context overflow
# -----------------------------------------------------------------------
class TestContextOverflow:
    def test_read_document(self) -> None:
        task = ContextOverflowTask(seed=0)
        task.setup()
        action = json.dumps({"tool": "read_document", "arguments": {"id": "0"}})
        result = task.execute_action(action)
        assert "Document 0" in result
        assert "VALUE_0_" in result

    def test_submit_correct_fact(self) -> None:
        task = ContextOverflowTask(seed=0)
        task.setup()
        # Read doc 0
        read_action = json.dumps({"tool": "read_document", "arguments": {"id": "0"}})
        doc_content = task.execute_action(read_action)
        # Extract the VALUE_ token
        for word in doc_content.split():
            if word.startswith("VALUE_0_"):
                value = word.rstrip(".")
                break
        submit = {
            "tool": "submit_fact",
            "arguments": {"doc_id": "0", "fact": f"The finding is {value}"},
        }
        task.execute_action(json.dumps(submit))
        assert task._extraction_correct == 1

    def test_scoring_defaults(self) -> None:
        task = ContextOverflowTask(seed=0)
        task.setup()
        result = task.score()
        assert result.accuracy == 0.0


# -----------------------------------------------------------------------
# Task 4: Session resume
# -----------------------------------------------------------------------
class TestSessionResume:
    def test_check_status_empty(self) -> None:
        task = SessionResumeTask(seed=0)
        task.setup()
        action = json.dumps({"tool": "check_status", "arguments": {}})
        result = task.execute_action(action)
        data = json.loads(result)
        assert len(data["completed_steps"]) == 0
        assert len(data["remaining"]) == 10

    def test_create_file(self) -> None:
        task = SessionResumeTask(seed=0)
        task.setup()
        action = json.dumps({
            "tool": "create_file",
            "arguments": {"name": "README.md", "content": "# MyProject"},
        })
        result = task.execute_action(action)
        assert "README.md" in result
        assert "1" in task._completed_step_ids

    def test_session_break(self) -> None:
        task = SessionResumeTask(seed=0)
        task.setup()
        # Complete steps 1-5
        steps = [
            ("create_file", {"name": "README.md", "content": "# MyProject"}),
            ("create_file", {"name": "setup.py", "content": "setup()"}),
            ("run_command", {"cmd": "git init"}),
            ("create_file", {"name": ".gitignore", "content": "__pycache__/"}),
            ("create_file", {"name": "src/__init__.py", "content": ""}),
        ]
        for tool, args in steps:
            task.execute_action(json.dumps({"tool": tool, "arguments": args}))

        # Next action should trigger session break
        result = task.execute_action(json.dumps({"tool": "check_status", "arguments": {}}))
        assert "SESSION RESTART" in result
        assert task._session_broken

    def test_redundant_detection(self) -> None:
        task = SessionResumeTask(seed=0)
        task.setup()
        action = json.dumps({
            "tool": "create_file",
            "arguments": {"name": "README.md", "content": "# MyProject"},
        })
        task.execute_action(action)
        result = task.execute_action(action)
        assert "redundant" in result.lower()
        assert task._redundant_steps == 1


# -----------------------------------------------------------------------
# Task 5: Multi-step plan
# -----------------------------------------------------------------------
class TestMultiStepPlan:
    def test_read_file(self) -> None:
        task = MultiStepPlanTask(seed=0)
        task.setup()
        action = json.dumps({"tool": "read_file", "arguments": {"path": "config.py"}})
        result = task.execute_action(action)
        assert "TIMEOUT = 30" in result

    def test_write_correct_fix(self) -> None:
        task = MultiStepPlanTask(seed=0)
        task.setup()
        fixed_content = (
            '# Configuration\n'
            'DATABASE_URL = "postgres://localhost:5432/mydb"\n'
            'MAX_CONNECTIONS = 10\n'
            'TIMEOUT = 60  # production timeout\n'
            '# Note: connection pool settings are in pool.py\n'
        )
        action = json.dumps({
            "tool": "write_file",
            "arguments": {"path": "config.py", "content": fixed_content},
        })
        task.execute_action(action)
        assert "config.py" in task._fixes_applied

    def test_run_tests_fail(self) -> None:
        task = MultiStepPlanTask(seed=0)
        task.setup()
        action = json.dumps({"tool": "run_tests", "arguments": {}})
        result = task.execute_action(action)
        assert "FAILED" in result

    def test_run_tests_pass_after_all_fixes(self) -> None:
        task = MultiStepPlanTask(seed=0)
        task.setup()
        # Apply all fixes
        for f in task._scenario:
            action = json.dumps({
                "tool": "write_file",
                "arguments": {"path": f["path"], "content": f["fixed"]},
            })
            task.execute_action(action)
        # Run tests
        action = json.dumps({"tool": "run_tests", "arguments": {}})
        result = task.execute_action(action)
        assert "PASSED" in result
        assert task._tests_passed
        assert task.is_complete()

    def test_scoring(self) -> None:
        task = MultiStepPlanTask(seed=0)
        task.setup()
        result = task.score()
        assert result.completion is False
        assert result.accuracy == 0.0

    def test_file_not_found(self) -> None:
        task = MultiStepPlanTask(seed=0)
        task.setup()
        action = json.dumps({"tool": "read_file", "arguments": {"path": "nonexistent.py"}})
        result = task.execute_action(action)
        assert "not found" in result.lower()
