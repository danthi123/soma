"""Tests for the BaselineAgent with mocked LLM responses."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from benchmarks.agentic.agents.baseline import BaselineAgent
from benchmarks.agentic.models import ModelConfig


def _make_agent(**kwargs) -> BaselineAgent:
    cfg = ModelConfig(
        name="test-model:latest",
        display_name="Test Model",
        vram_estimate_gb=1.0,
        context_window=1024,
    )
    return BaselineAgent(model_config=cfg, **kwargs)


def _openai_response(
    content: str = "",
    tool_calls: list | None = None,
) -> dict:
    """Build a fake OpenAI-compatible chat/completions response."""
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


class TestBaselineAgentParsing:
    """Test tool-call parsing from OpenAI-compatible responses."""

    def test_parse_text_response(self) -> None:
        agent = _make_agent()
        resp = _openai_response(content="Hello, world!")
        action = agent._parse_response(resp)
        assert action == "Hello, world!"

    def test_parse_tool_call(self) -> None:
        agent = _make_agent()
        resp = _openai_response(
            tool_calls=[{
                "function": {
                    "name": "search_database",
                    "arguments": {"query": "quantum"},
                }
            }]
        )
        action = agent._parse_response(resp)
        parsed = json.loads(action)
        assert parsed["tool"] == "search_database"
        assert parsed["arguments"]["query"] == "quantum"

    def test_parse_tool_call_string_args(self) -> None:
        """Ollama sometimes returns arguments as a JSON string."""
        agent = _make_agent()
        resp = _openai_response(
            tool_calls=[{
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "config.py"}',
                }
            }]
        )
        action = agent._parse_response(resp)
        parsed = json.loads(action)
        assert parsed["tool"] == "read_file"
        assert parsed["arguments"]["path"] == "config.py"

    def test_parse_empty_response(self) -> None:
        agent = _make_agent()
        resp = _openai_response(content="")
        action = agent._parse_response(resp)
        assert "error" in action.lower()

    def test_parse_json_content_as_tool_call(self) -> None:
        """When model returns tool call as content JSON."""
        agent = _make_agent()
        tc = json.dumps({"tool": "run_tests", "arguments": {}})
        resp = _openai_response(content=tc)
        action = agent._parse_response(resp)
        parsed = json.loads(action)
        assert parsed["tool"] == "run_tests"


class TestBaselineAgentContext:
    """Test context truncation."""

    def test_truncation_drops_oldest(self) -> None:
        agent = _make_agent(max_context_tokens=50)
        # Each message is about 25 chars ~ 6 tokens
        agent._messages = [
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
            {"role": "user", "content": "C" * 100},
        ]
        agent._truncate_if_needed()
        # Should have dropped some messages
        assert len(agent._messages) < 3

    def test_truncation_keeps_at_least_one(self) -> None:
        agent = _make_agent(max_context_tokens=10)
        agent._messages = [
            {"role": "user", "content": "X" * 1000},
        ]
        agent._truncate_if_needed()
        assert len(agent._messages) == 1  # never drops below 1

    def test_reset_clears_messages(self) -> None:
        agent = _make_agent()
        agent._messages = [
            {"role": "user", "content": "test"},
        ]
        agent._total_steps = 5
        agent.reset()
        assert agent._messages == []
        assert agent._total_steps == 0


class TestBaselineAgentStep:
    """Test the step() method with mocked HTTP."""

    @patch("benchmarks.agentic.agents.baseline.requests.post")
    def test_step_calls_ollama(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(
            content="I will help you."
        )
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        agent = _make_agent()
        action = agent.step("Hello")

        assert action == "I will help you."
        assert agent._total_steps == 1
        assert len(agent._messages) == 2  # user + assistant

        # Verify Ollama was called correctly
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        assert payload["model"] == "test-model:latest"
        assert payload["stream"] is False

    @patch("benchmarks.agentic.agents.baseline.requests.post")
    def test_step_with_tool_call(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(
            tool_calls=[{
                "function": {
                    "name": "search_database",
                    "arguments": {"query": "test"},
                }
            }]
        )
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        agent = _make_agent(
            tools=[{
                "type": "function",
                "function": {
                    "name": "search_database",
                    "parameters": {},
                },
            }]
        )
        action = agent.step("Search for something")

        parsed = json.loads(action)
        assert parsed["tool"] == "search_database"

        # Verify tools were sent in payload
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        assert "tools" in payload

    @patch("benchmarks.agentic.agents.baseline.requests.post")
    def test_disable_thinking(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(content="ok")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = ModelConfig(
            name="test:latest",
            display_name="Test",
            vram_estimate_gb=1.0,
            disable_thinking=True,
        )
        agent = BaselineAgent(model_config=cfg)
        agent.step("test")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]

        # System prompt should start with /no_think
        sys_msg = payload["messages"][0]
        assert sys_msg["content"].startswith("/no_think")

    @patch("benchmarks.agentic.agents.baseline.requests.post")
    def test_thinking_enabled(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(content="ok")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cfg = ModelConfig(
            name="test:latest",
            display_name="Test",
            vram_estimate_gb=1.0,
            disable_thinking=False,
        )
        agent = BaselineAgent(model_config=cfg)
        agent.step("test")

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]

        # System prompt should NOT have /no_think
        sys_msg = payload["messages"][0]
        assert not sys_msg["content"].startswith("/no_think")


class TestBaselineAgentE2E:
    """End-to-end test: agent + task through harness, mocked LLM."""

    @patch("benchmarks.agentic.agents.baseline.requests.post")
    def test_harness_loop(self, mock_post: MagicMock) -> None:
        from benchmarks.agentic.harness import run
        from benchmarks.agentic.tasks.tool_learning import (
            ToolLearningTask,
        )

        # Mock Ollama to always return submit_results
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_response(
            tool_calls=[{
                "function": {
                    "name": "submit_results",
                    "arguments": {"results": []},
                }
            }]
        )
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        agent = _make_agent()
        task = ToolLearningTask(seed=0)
        agent.tools = task.TOOLS

        result = run(
            agent, task, task_name="tool_learning", max_steps=10
        )
        assert result.total_steps == 1  # submit on first step
        assert result.task_result.steps == 1
