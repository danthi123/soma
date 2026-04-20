"""Tests for ClaudeRunnerClient — calls Claude via Unraid docker image.

These tests mock subprocess.run to validate the command construction and
output parsing without hitting the network. A separate integration test
(marked ``@pytest.mark.integration``) can exercise the real Unraid box.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from benchmarks.industry.llm_backends.claude_runner_client import (
    ClaudeRunnerClient,
    ClaudeRunnerError,
)


def _mock_run(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock(spec=subprocess.CompletedProcess)
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


def test_complete_returns_stripped_stdout():
    """Happy path: client returns cleaned stdout from subprocess."""
    client = ClaudeRunnerClient(ssh_host="root@host")
    with patch("subprocess.run", return_value=_mock_run("  Paris\n")) as sp:
        result = client.complete("capital of France?")
    assert result == "Paris"
    assert sp.called


def test_complete_builds_ssh_docker_command():
    """Command should invoke ssh, then docker run on the remote host."""
    client = ClaudeRunnerClient(
        ssh_host="root@unraid",
        token_path="/path/to/token",
        image="claude-code-runner:latest",
    )
    captured_cmd = []

    def fake_run(cmd, **_kwargs):
        captured_cmd.append(cmd)
        return _mock_run("answer\n")

    with patch("subprocess.run", side_effect=fake_run):
        client.complete("hello")

    cmd = captured_cmd[0]
    assert cmd[0] == "ssh"
    assert "root@unraid" in cmd
    remote = " ".join(cmd[2:]) if len(cmd) > 2 else cmd[-1]
    assert "docker run" in remote
    assert "claude-code-runner:latest" in remote
    assert "/path/to/token" in remote


def test_complete_passes_prompt_via_stdin():
    """Prompt goes in via stdin to survive special chars/newlines."""
    client = ClaudeRunnerClient(ssh_host="root@host")
    captured_input = []

    def fake_run(cmd, input=None, **_kwargs):
        captured_input.append(input)
        return _mock_run("ok")

    with patch("subprocess.run", side_effect=fake_run):
        client.complete("A tricky prompt with 'quotes' and\nnewlines and $vars")

    assert captured_input[0] == "A tricky prompt with 'quotes' and\nnewlines and $vars"


def test_complete_combines_system_and_user():
    """When system_prompt is given, it is prepended to the user message."""
    client = ClaudeRunnerClient(ssh_host="root@host")
    captured_input = []

    def fake_run(cmd, input=None, **_kwargs):
        captured_input.append(input)
        return _mock_run("ok")

    with patch("subprocess.run", side_effect=fake_run):
        client.complete(
            "What is 2+2?",
            system_prompt="You only answer with numbers.",
        )

    assert captured_input[0].startswith("You only answer with numbers.")
    assert "What is 2+2?" in captured_input[0]


def test_complete_raises_on_nonzero_exit():
    """Non-zero exit code raises ClaudeRunnerError with stderr included."""
    client = ClaudeRunnerClient(ssh_host="root@host")
    with patch(
        "subprocess.run",
        return_value=_mock_run("", returncode=1, stderr="auth failed"),
    ):
        with pytest.raises(ClaudeRunnerError) as exc_info:
            client.complete("hello")
    assert "auth failed" in str(exc_info.value)


def test_complete_returns_empty_on_empty_stdout_and_zero_exit():
    """Zero exit + empty stdout = empty string (no exception).

    Claude sometimes returns empty on trivial/meta prompts; benchmark
    code should see an empty string rather than an error.
    """
    client = ClaudeRunnerClient(ssh_host="root@host")
    with patch("subprocess.run", return_value=_mock_run("")):
        assert client.complete("hello") == ""


def test_complete_timeout_passed_to_subprocess():
    """Timeout value is forwarded to subprocess.run."""
    client = ClaudeRunnerClient(ssh_host="root@host", timeout=60)
    captured_kwargs = {}

    def fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        return _mock_run("ok")

    with patch("subprocess.run", side_effect=fake_run):
        client.complete("hello")
    assert captured_kwargs.get("timeout") == 60


def test_complete_handles_subprocess_timeout_as_error():
    """subprocess.TimeoutExpired surfaces as ClaudeRunnerError."""
    client = ClaudeRunnerClient(ssh_host="root@host", timeout=1)
    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="ssh ...", timeout=1),
    ):
        with pytest.raises(ClaudeRunnerError) as exc_info:
            client.complete("hello")
    assert "timed out" in str(exc_info.value).lower()


def test_complete_forces_utf8_encoding():
    """subprocess.run must be called with encoding='utf-8'.

    Windows defaults to cp1252 which crashes on emoji/non-BMP chars
    present in real LongMemEval transcripts.
    """
    client = ClaudeRunnerClient(ssh_host="root@host")
    captured_kwargs = {}

    def fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        return _mock_run("ok")

    with patch("subprocess.run", side_effect=fake_run):
        # Prompt with an emoji that would crash under cp1252.
        client.complete("Hello \U0001f60a world")

    assert captured_kwargs.get("encoding") == "utf-8"
    # errors="replace" prevents crashes on any odd byte sequence.
    assert captured_kwargs.get("errors") == "replace"
