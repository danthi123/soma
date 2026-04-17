"""SomaAgent: LLM + SOMA agent for the agentic benchmark.

Every tool-call result, observation, and decision is stored via
``store_typed()`` using the ``agent.*`` schemas.  Before each LLM call,
``pack_context()`` builds a bounded prompt from:

- recency (recent turns)
- relevant facts (semantic retrieval)
- task state (active ``agent.task_state``)
- tool examples (past successful tool calls)
- decisions (rationale trail)

Cross-session: ``save()``/``load()`` the MemoryLayer.  The harness
calls ``reset()`` between sessions (Task 4: session resume).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import torch

from benchmarks.agentic.models import DEFAULT_API_BASE, ModelConfig
from soma.memory.api import MemoryLayer
from soma.schemas.builtin.agent import Observation, ToolCall
from soma.schemas.packing import pack_context

logger = logging.getLogger(__name__)

# Rough token estimate: 1 token ~ 4 chars (conservative).
_CHARS_PER_TOKEN = 4

# Default embed dimension for the stub embedder.
_STUB_EMBED_DIM = 16


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic but non-trivially-overlapping vectors."""
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(_STUB_EMBED_DIM)


EmbedFn = Any  # Callable[[str], torch.Tensor]


def _make_memory_layer(
    embed_fn: EmbedFn | None = None,
    embed_dim: int | None = None,
) -> MemoryLayer:
    """Build a fresh MemoryLayer with the given (or stub) embedder."""
    fn = embed_fn or _stub_embed
    dim = embed_dim or _STUB_EMBED_DIM
    return MemoryLayer(embed_fn=fn, embed_dim=dim)


@dataclass
class SomaAgent:
    """LLM + SOMA agent backed by MemoryLayer and typed schemas."""

    model_config: ModelConfig
    system_prompt: str = "You are a helpful assistant."
    tools: list[dict[str, Any]] = field(default_factory=list)
    api_base: str = DEFAULT_API_BASE
    max_context_tokens: int = 3800
    embed_fn: EmbedFn | None = field(default=None, repr=False)
    embed_dim: int | None = None

    # Internal state
    _mem: MemoryLayer = field(init=False, repr=False)
    _total_steps: int = field(default=0, init=False)
    _tool_errors: int = field(default=0, init=False)
    _current_task_id: str = field(default="task-0", init=False)
    _persist_path: Path | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._mem = _make_memory_layer(self.embed_fn, self.embed_dim)
        self._persist_path = None

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------
    def reset(self, *, persist: bool = False) -> None:
        """Reset for a new task.

        If *persist* is ``True`` (Task 4: session resume), save the
        MemoryLayer to a temp dir and reload it on the next ``step()``
        call -- simulating a process restart with SOMA persistence.

        If *persist* is ``False``, start fresh with an empty MemoryLayer.
        """
        if persist:
            tmp = Path(tempfile.mkdtemp(prefix="soma_persist_"))
            self._mem.save(tmp)
            self._persist_path = tmp
            # Rebuild a fresh MemoryLayer and immediately reload
            self._mem = MemoryLayer.load(
                self._persist_path,
                embed_fn=self.embed_fn or _stub_embed,
            )
        else:
            self._persist_path = None
            self._mem = _make_memory_layer(self.embed_fn, self.embed_dim)

        self._total_steps = 0
        self._tool_errors = 0

    def step(self, observation: str) -> str:
        """Receive an observation, call the LLM, return an action string."""
        # 1. Store the observation
        self._mem.store_typed(
            Observation(
                task_id=self._current_task_id,
                source="env",
                content=observation,
            )
        )

        # 2. Build context via pack_context
        context = pack_context(
            self._mem,
            query=observation,
            max_tokens=self.max_context_tokens,
        )

        # 3. Build messages: system prompt + packed context + observation
        sys_content = self._build_system_prompt()
        if context.strip():
            sys_content += "\n\nContext from memory:\n" + context

        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": observation},
        ]

        # 4. Call Ollama
        response = self._call_llm(messages)

        # 5. Parse the response
        action = self._parse_response(response)
        self._total_steps += 1

        # 6. If it is a tool call, store it
        try:
            parsed = json.loads(action)
            if isinstance(parsed, dict) and parsed.get("tool"):
                args_str = json.dumps(parsed.get("arguments", {}))
                summary = (
                    f"Called {parsed['tool']} with {args_str}"
                )
                self._mem.store_typed(
                    ToolCall(
                        task_id=self._current_task_id,
                        tool_name=parsed["tool"],
                        args=args_str,
                        result_summary=summary,
                        success=True,
                    )
                )
        except (json.JSONDecodeError, TypeError):
            pass

        return action

    def record_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        success: bool,
    ) -> None:
        """Called by the harness after tool execution to record outcome."""
        summary = str(result)[:500] if result else f"Executed {tool_name}"
        self._mem.store_typed(
            ToolCall(
                task_id=self._current_task_id,
                tool_name=tool_name,
                args=json.dumps(args),
                result_summary=summary,
                success=success,
            )
        )

    def get_metrics(self) -> dict[str, Any]:
        return {
            "total_steps": self._total_steps,
            "tool_errors": self._tool_errors,
            "context_tokens": self.max_context_tokens,
            "memory_entries": len(self._mem._ids),
        }

    # ------------------------------------------------------------------
    # LLM integration (OpenAI-compatible API, mirrors BaselineAgent)
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        prompt = self.system_prompt
        if self.model_config.disable_thinking:
            prompt = "/no_think\n" + prompt
        return prompt

    def _call_llm(self, messages: list[dict[str, str]]) -> dict:
        """Call the OpenAI-compatible chat/completions endpoint."""
        payload: dict[str, Any] = {
            "model": self.model_config.name,
            "messages": messages,
            "stream": False,
            "max_tokens": 1024,
        }

        if self.tools:
            wrapped = []
            for t in self.tools:
                if "type" in t and "function" in t:
                    wrapped.append(t)
                else:
                    wrapped.append({"type": "function", "function": t})
            payload["tools"] = wrapped

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("LLM call failed: %s", exc)
            self._tool_errors += 1
            return {
                "choices": [{
                    "message": {"content": f"ERROR: LLM call failed: {exc}"}
                }]
            }

    def _parse_response(self, response: dict) -> str:
        """Extract tool calls or text from OpenAI-format response.

        Handles both OpenAI-compatible (choices[0].message) and
        Ollama native (message at top level) formats.
        """
        choices = response.get("choices")
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
        else:
            message = response.get("message", {})

        tool_calls = message.get("tool_calls")

        if tool_calls and len(tool_calls) > 0:
            tc = tool_calls[0]
            func = tc.get("function", tc)
            name = func.get("name", "")
            arguments = func.get("arguments", {})

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

        content = message.get("content", "")
        if not content:
            self._tool_errors += 1
            return '{"error": "empty response"}'

        if content.strip().startswith("{"):
            try:
                parsed = json.loads(content)
                if "tool" in parsed or "name" in parsed:
                    return content.strip()
            except json.JSONDecodeError:
                pass

        return content

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_memory(self, path: str | Path) -> None:
        """Save the MemoryLayer to disk."""
        self._mem.save(Path(path))

    def load_memory(self, path: str | Path) -> None:
        """Load a previously saved MemoryLayer from disk."""
        self._mem = MemoryLayer.load(
            Path(path),
            embed_fn=self.embed_fn or _stub_embed,
        )
