"""Client for the Unraid claude-code-runner Docker image.

Uses the user's Claude Max subscription via the long-lived OAuth token
(``CLAUDE_CODE_OAUTH_TOKEN``) that the stocks-runner / claude-runner
infrastructure on Unraid already maintains. No per-request API charges.

Transport: SSH into Unraid, invoke ``docker run`` with the token pulled
from a secrets file, pipe the prompt via stdin. The container's
``claude -p`` flag turns it into a single-shot, non-interactive call.

Example::

    client = ClaudeRunnerClient()
    answer = client.complete(
        "What is the capital of France?",
        system_prompt="Answer with the city name only.",
    )
    # answer == "Paris"
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class ClaudeRunnerError(RuntimeError):
    """Raised when the remote claude-code-runner invocation fails."""


@dataclass
class ClaudeRunnerClient:
    """Thin SSH+docker wrapper around Unraid claude-code-runner.

    Attributes:
        ssh_host: SSH target, e.g. ``root@192.168.0.10``.
        token_path: Path on the remote host to the OAuth token file.
            The default matches the stocks-runner / n8n convention.
        image: Docker image to run.
        timeout: Per-call timeout in seconds.
        extra_claude_args: Additional flags to pass to ``claude``.
    """

    ssh_host: str = "root@192.168.0.10"
    token_path: str = "/mnt/cache/appdata/claude-runner/auth-token"
    image: str = "claude-code-runner:latest"
    timeout: int = 180
    extra_claude_args: list[str] = field(default_factory=list)

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call Claude with ``prompt`` and return its stripped response.

        Args:
            prompt: User message / question for Claude.
            system_prompt: Optional system prompt. If given, prepended to
                the user prompt with a blank line in between. (We can't
                use ``--append-system-prompt`` without exposing the
                prompt in the shell command — stdin is safer.)

        Returns:
            Claude's reply as a stripped string. Empty string if Claude
            returned nothing.

        Raises:
            ClaudeRunnerError: On SSH failure, docker failure, or timeout.
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Remote command: cat prompt from stdin into a shell variable,
        # then invoke claude. Redirect stderr to /dev/null to keep stdout
        # clean for the benchmark caller.
        remote_cmd = (
            f'TOKEN=$(cat {self.token_path}); '
            f'docker run --rm -i '
            f'-e CLAUDE_CODE_OAUTH_TOKEN="$TOKEN" '
            f'--entrypoint bash '
            f'{self.image} '
            f'-c "PROMPT=\\$(cat); '
            f'claude -p \\"\\$PROMPT\\" '
            f'--dangerously-skip-permissions '
            f'--no-session-persistence '
            f'{" ".join(self.extra_claude_args)} '
            f'2>/dev/null"'
        )

        cmd = ["ssh", self.ssh_host, remote_cmd]

        try:
            # encoding="utf-8" is explicit because Windows defaults to
            # cp1252, which crashes on emoji / non-BMP chars sometimes
            # present in LongMemEval transcripts. errors="replace"
            # guarantees no crash if the remote ever emits a byte we
            # can't decode — benchmarks score on semantic equivalence,
            # not exact bytes.
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeRunnerError(
                f"Claude runner timed out after {self.timeout}s"
            ) from exc

        if result.returncode != 0:
            raise ClaudeRunnerError(
                f"Claude runner failed (exit={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        return result.stdout.strip()
