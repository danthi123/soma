"""LLM backends for benchmarks.

Each backend exposes a minimal ``complete(prompt, system_prompt=None)``
interface returning the LLM's stripped text response, abstracting away
the transport (HTTP, SSH+docker, etc.).
"""

from benchmarks.industry.llm_backends.claude_runner_client import (
    ClaudeRunnerClient,
    ClaudeRunnerError,
)

__all__ = ["ClaudeRunnerClient", "ClaudeRunnerError"]
