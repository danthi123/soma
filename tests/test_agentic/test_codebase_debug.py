"""Tests for the codebase debugging task."""

from __future__ import annotations

import json

from benchmarks.agentic.tasks.codebase_debug import CodebaseDebugTask


class TestCodebaseDebug:
    def test_setup_returns_bug_report(self) -> None:
        task = CodebaseDebugTask(seed=0)
        obs = task.setup()
        assert "bug" in obs.lower() or "error" in obs.lower()
        assert not task.is_complete()

    def test_list_files(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "list_files", "arguments": {}}'
        )
        assert ".py" in result
        assert "main.py" in result

    def test_read_file(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "read_file", "arguments": {"path": "main.py"}}'
        )
        assert "main.py" in result
        assert "def " in result or "from " in result

    def test_read_file_not_found(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "read_file", "arguments": {"path": "nope.py"}}'
        )
        assert "not found" in result.lower()

    def test_search_codebase(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "search_codebase", "arguments": {"query": "import"}}'
        )
        assert "match" in result.lower() or ":" in result

    def test_grep_tool(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "grep", "arguments": {"pattern": "def "}}'
        )
        assert "match" in result.lower() or ":" in result

    def test_grep_invalid_regex(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "grep", "arguments": {"pattern": "[invalid"}}'
        )
        assert "error" in result.lower()

    def test_run_tests_initially_fails(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        result = task.execute_action(
            '{"tool": "run_tests", "arguments": {}}'
        )
        assert "FAIL" in result

    def test_scoring_defaults(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        score = task.score()
        assert score.accuracy == 0.0
        assert not score.completion

    def test_write_correct_fix(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        # Fix config.py
        fixed = task._files["config.py"]["fixed"]
        action = json.dumps({
            "tool": "write_file",
            "arguments": {"path": "config.py", "content": fixed},
        })
        task.execute_action(action)
        assert task._fixes_applied["config.py"] is True

    def test_full_fix_and_tests_pass(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        buggy_files = [
            name
            for name, info in task._files.items()
            if info["buggy"]
        ]
        for name in buggy_files:
            action = json.dumps({
                "tool": "write_file",
                "arguments": {
                    "path": name,
                    "content": task._files[name]["fixed"],
                },
            })
            task.execute_action(action)
        # Run tests
        result = task.execute_action(
            '{"tool": "run_tests", "arguments": {}}'
        )
        assert "PASS" in result
        assert task.is_complete()
        score = task.score()
        assert score.completion is True
        assert score.accuracy == 100.0

    def test_scenario_variant_seed_1(self) -> None:
        task = CodebaseDebugTask(seed=1)
        obs = task.setup()
        assert "bug" in obs.lower() or "error" in obs.lower()
        # Scenario 1 has different files
        result = task.execute_action(
            '{"tool": "list_files", "arguments": {}}'
        )
        assert "serializer.py" in result

    def test_whitespace_tolerant_fix(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task.setup()
        # Add trailing whitespace -- should still match
        fixed = task._files["config.py"]["fixed"]
        fixed_with_ws = fixed.rstrip() + "   \n"
        action = json.dumps({
            "tool": "write_file",
            "arguments": {
                "path": "config.py",
                "content": fixed_with_ws,
            },
        })
        task.execute_action(action)
        assert task._fixes_applied["config.py"] is True

    def test_max_steps_terminates(self) -> None:
        task = CodebaseDebugTask(seed=0)
        task._max_steps = 3
        task.setup()
        for _ in range(5):
            task.execute_action(
                '{"tool": "list_files", "arguments": {}}'
            )
        assert task.is_complete()
