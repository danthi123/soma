"""Tests for SomaAgent with mocked Ollama responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from benchmarks.agentic.agents.soma_agent import SomaAgent
from benchmarks.agentic.models import ModelConfig


def _make_agent(**kwargs) -> SomaAgent:
    cfg = ModelConfig(
        name="test-model:latest",
        display_name="Test Model",
        vram_estimate_gb=1.0,
        context_window=1024,
    )
    return SomaAgent(model_config=cfg, **kwargs)


def _ollama_response(
    content: str = "",
    tool_calls: list | None = None,
) -> dict:
    """Build a fake Ollama /api/chat response."""
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"message": msg}


class TestSomaAgentStoresObservations:
    """After one step, MemoryLayer has a stored Observation."""

    @patch("benchmarks.agentic.agents.soma_agent.requests.post")
    def test_observation_stored(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_response(
            content="I understand."
        )
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        agent = _make_agent()
        agent.step("The sky is blue.")

        # MemoryLayer should have at least one entry
        assert len(agent._mem._ids) >= 1

        # Check that the observation text was stored
        stored_texts = agent._mem._texts
        assert any("sky is blue" in t for t in stored_texts)


class TestSomaAgentStoresToolCalls:
    """After a tool-call step, MemoryLayer has a stored ToolCall."""

    @patch("benchmarks.agentic.agents.soma_agent.requests.post")
    def test_tool_call_stored(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_response(
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
        agent.step("Search for something")

        # Should have observation + tool call entries
        assert len(agent._mem._ids) >= 2

        # Check that tool call metadata was stored
        tool_entries = [
            md for md in agent._mem._metadatas
            if md.get("type") == "agent.tool_call"
        ]
        assert len(tool_entries) >= 1
        assert tool_entries[0]["tool_name"] == "search_database"


class TestSomaAgentUsesPackContext:
    """The messages sent to Ollama include packed context from memory."""

    @patch("benchmarks.agentic.agents.soma_agent.requests.post")
    def test_context_included_in_prompt(
        self, mock_post: MagicMock
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_response(content="ok")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        agent = _make_agent()
        # First step stores an observation
        agent.step("The capital of France is Paris.")
        # Second step should include context from memory
        agent.step("What is the capital?")

        # Check the last Ollama call's system message
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        sys_msg = payload["messages"][0]["content"]

        # The system message should contain "Context from memory"
        assert "Context from memory" in sys_msg


class TestSomaAgentPersistsAcrossReset:
    """reset(persist=True) saves; subsequent step() retrieves from saved."""

    @patch("benchmarks.agentic.agents.soma_agent.requests.post")
    def test_persist_and_reload(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_response(content="ok")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        agent = _make_agent()
        # Store some data
        agent.step("Important fact: project uses Python 3.11")

        entries_before = len(agent._mem._ids)
        assert entries_before >= 1

        # Persist and reset
        agent.reset(persist=True)

        # Memory should survive the reset
        entries_after = len(agent._mem._ids)
        assert entries_after == entries_before


class TestSomaAgentFreshResetClearsMemory:
    """reset(persist=False) starts with empty MemoryLayer."""

    @patch("benchmarks.agentic.agents.soma_agent.requests.post")
    def test_fresh_reset(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_response(content="ok")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        agent = _make_agent()
        agent.step("Some data")
        assert len(agent._mem._ids) >= 1

        agent.reset(persist=False)
        assert len(agent._mem._ids) == 0


class TestSomaAgentContextBounded:
    """After many observations, context size stays under max_tokens."""

    @patch("benchmarks.agentic.agents.soma_agent.requests.post")
    def test_bounded_context(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_response(content="ok")
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        max_tokens = 200
        agent = _make_agent(max_context_tokens=max_tokens)

        # Feed 100 observations
        for i in range(100):
            agent.step(f"Observation number {i}: " + "x" * 50)

        # Check the last Ollama call's system message size
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1]["json"]
        sys_msg = payload["messages"][0]["content"]

        # Extract the context portion after "Context from memory:"
        if "Context from memory:" in sys_msg:
            context_part = sys_msg.split("Context from memory:\n", 1)[1]
            # Context should be bounded (roughly max_tokens * 4 chars)
            assert len(context_part) <= max_tokens * 4 + 200  # some slack


class TestSomaAgentRegisteredInHarness:
    """SomaAgent is registered in AGENT_REGISTRY."""

    def test_registered(self) -> None:
        from benchmarks.agentic.agents import AGENT_REGISTRY

        assert "soma" in AGENT_REGISTRY
        assert AGENT_REGISTRY["soma"] is SomaAgent


class TestSomaAgentRecordToolResult:
    """record_tool_result stores ToolCall with result info."""

    def test_record_tool_result(self) -> None:
        agent = _make_agent()
        agent.record_tool_result(
            tool_name="search_database",
            args={"query": "test"},
            result="Found 3 results",
            success=True,
        )

        assert len(agent._mem._ids) >= 1
        tool_entries = [
            md for md in agent._mem._metadatas
            if md.get("type") == "agent.tool_call"
        ]
        assert len(tool_entries) == 1
        assert tool_entries[0]["tool_name"] == "search_database"
        assert tool_entries[0]["success"] is True


class TestSomaAgentE2E:
    """End-to-end test: agent + task through harness, mocked LLM."""

    @patch("benchmarks.agentic.agents.soma_agent.requests.post")
    def test_harness_loop(self, mock_post: MagicMock) -> None:
        from benchmarks.agentic.harness import run
        from benchmarks.agentic.tasks.tool_learning import (
            ToolLearningTask,
        )

        # Mock Ollama to always return submit_results
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_response(
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

        # Verify SOMA stored entries during the run
        assert len(agent._mem._ids) >= 1
