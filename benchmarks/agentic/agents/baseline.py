"""Baseline agent: standalone LLM with context-window truncation.

Talks to Ollama ``/api/chat`` with tool definitions.  When the
conversation exceeds ``max_context_tokens``, the oldest messages
(after the system prompt) are dropped.

No cross-session persistence -- ``reset()`` clears everything.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

from benchmarks.agentic.models import ModelConfig

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ~ 4 chars (conservative)
_CHARS_PER_TOKEN = 4


@dataclass
class BaselineAgent:
    """Pure-LLM agent backed by Ollama."""

    model_config: ModelConfig
    system_prompt: str = "You are a helpful assistant."
    tools: list[dict[str, Any]] = field(default_factory=list)
    ollama_url: str = "http://localhost:11434"
    max_context_tokens: int | None = None  # None = use model default

    # Internal state
    _messages: list[dict[str, str]] = field(
        default_factory=list, repr=False
    )
    _total_steps: int = 0
    _tool_errors: int = 0

    def __post_init__(self) -> None:
        if self.max_context_tokens is None:
            self.max_context_tokens = self.model_config.context_window

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        self._total_steps = 0
        self._tool_errors = 0

    def step(self, observation: str) -> str:
        """Receive an observation, call the LLM, return an action string.

        The action is either a tool-call JSON or plain text.
        """
        self._messages.append({"role": "user", "content": observation})
        self._truncate_if_needed()

        response = self._call_ollama()
        action = self._parse_response(response)

        self._messages.append({"role": "assistant", "content": action})
        self._total_steps += 1
        return action

    def get_metrics(self) -> dict[str, Any]:
        return {
            "total_steps": self._total_steps,
            "tool_errors": self._tool_errors,
            "context_messages": len(self._messages),
        }

    # ------------------------------------------------------------------
    # Ollama integration
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        prompt = self.system_prompt
        if self.model_config.disable_thinking:
            prompt = "/no_think\n" + prompt
        return prompt

    def _call_ollama(self) -> dict:
        """Call Ollama /api/chat and return the response JSON."""
        payload: dict[str, Any] = {
            "model": self.model_config.name,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                *self._messages,
            ],
            "stream": False,
            "options": {
                **self.model_config.ollama_options,
            },
        }

        if self.model_config.disable_thinking:
            payload["options"]["think"] = False

        if self.tools:
            payload["tools"] = self.tools

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Ollama call failed: %s", exc)
            self._tool_errors += 1
            return {"message": {"content": f"ERROR: LLM call failed: {exc}"}}

    def _parse_response(self, response: dict) -> str:
        """Extract tool calls or text from Ollama response.

        Ollama returns tool calls in the message.tool_calls field:
        ```json
        {
          "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
              {
                "function": {
                  "name": "search_database",
                  "arguments": {"query": "quantum"}
                }
              }
            ]
          }
        }
        ```

        We convert the first tool call into a JSON action string
        that the task can parse.
        """
        message = response.get("message", {})
        tool_calls = message.get("tool_calls")

        if tool_calls and len(tool_calls) > 0:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", {})

            # Normalize arguments if they came as a string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    self._tool_errors += 1
                    arguments = {}

            return json.dumps({
                "tool": name,
                "arguments": arguments,
            })

        # No tool call -- return raw content
        content = message.get("content", "")
        if not content:
            self._tool_errors += 1
            return '{"error": "empty response"}'

        # Try to detect if the content IS a tool call in JSON format
        if content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
                if "tool" in parsed or "name" in parsed:
                    return content.strip()
            except json.JSONDecodeError:
                pass

        return content

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def _estimate_tokens(self) -> int:
        """Rough token count of current conversation."""
        total_chars = len(self._build_system_prompt())
        for msg in self._messages:
            total_chars += len(msg.get("content", ""))
        return total_chars // _CHARS_PER_TOKEN

    def _truncate_if_needed(self) -> None:
        """Drop oldest messages (keeping system prompt separate)."""
        assert self.max_context_tokens is not None
        while (
            self._estimate_tokens() > self.max_context_tokens
            and len(self._messages) > 1
        ):
            self._messages.pop(0)
