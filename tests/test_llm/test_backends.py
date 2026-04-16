"""Tests for the LLM backend layer (mocked — never hits real APIs)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any
from unittest import mock

import pytest

from soma.llm import (
    AnthropicBackend,
    DryRunBackend,
    LLMBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenAICompatibleBackend,
    backend_from_env,
)


@contextmanager
def _env(**kwargs: str | None) -> Any:
    """Context manager: set env vars (None deletes), restore on exit."""
    before: dict[str, str | None] = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_protocol_runtime_check_recognizes_backends() -> None:
    assert isinstance(DryRunBackend(), LLMBackend)
    assert isinstance(OllamaBackend(), LLMBackend)


def test_dry_run_backend_echoes_context() -> None:
    b = DryRunBackend()
    out = b.generate("context block:\n[1] foo\n    body line\n\nquestion: x", max_tokens=10)
    assert "[dry-run]" in out
    assert "1 chunks" in out


def test_dry_run_backend_handles_no_context() -> None:
    out = DryRunBackend().generate("plain prompt with no brackets")
    assert "no context" in out


def test_ollama_posts_to_generate_endpoint() -> None:
    captured = {}

    fake_payload = json.dumps({"response": "hello world"}).encode("utf-8")
    fake_resp = mock.MagicMock()
    fake_resp.read.return_value = fake_payload
    fake_resp.__enter__.return_value = fake_resp
    fake_resp.__exit__.return_value = False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return fake_resp

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        b = OllamaBackend(model="llama3.2", host="http://example:11434")
        out = b.generate("hello", max_tokens=42)
    assert out == "hello world"
    assert captured["url"] == "http://example:11434/api/generate"
    assert captured["data"]["model"] == "llama3.2"
    assert captured["data"]["prompt"] == "hello"
    assert captured["data"]["options"]["num_predict"] == 42
    assert captured["data"]["stream"] is False


def test_ollama_raises_clear_error_on_unreachable() -> None:
    import urllib.error

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("Connection refused")

    with mock.patch("urllib.request.urlopen", fake_urlopen):
        b = OllamaBackend()
        with pytest.raises(RuntimeError, match="Ollama server unreachable"):
            b.generate("hi")


def test_ollama_name_includes_model() -> None:
    assert OllamaBackend(model="qwen2").name == "ollama:qwen2"


def test_openai_compat_uses_sdk_with_custom_base_url() -> None:
    fake_msg = mock.MagicMock()
    fake_msg.content = "compat reply"
    fake_choice = mock.MagicMock()
    fake_choice.message = fake_msg
    fake_resp = mock.MagicMock()
    fake_resp.choices = [fake_choice]

    fake_client = mock.MagicMock()
    fake_client.chat.completions.create.return_value = fake_resp

    fake_openai_module = mock.MagicMock()
    fake_openai_module.OpenAI.return_value = fake_client

    with mock.patch.dict("sys.modules", {"openai": fake_openai_module}):
        b = OpenAICompatibleBackend(
            model="my-local",
            base_url="http://vllm:8000/v1",
            api_key="x",
        )
        out = b.generate("question", max_tokens=99)
    assert out == "compat reply"
    fake_openai_module.OpenAI.assert_called_once_with(
        api_key="x", base_url="http://vllm:8000/v1", timeout=120.0
    )
    create_call = fake_client.chat.completions.create.call_args
    assert create_call.kwargs["model"] == "my-local"
    assert create_call.kwargs["max_tokens"] == 99
    assert create_call.kwargs["messages"][0]["content"] == "question"


def test_openai_backend_requires_key() -> None:
    with _env(OPENAI_API_KEY=None), pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIBackend()


def test_openai_backend_picks_up_env_key() -> None:
    with _env(OPENAI_API_KEY="sk-test"):
        b = OpenAIBackend(model="gpt-4o-mini")
        assert b.api_key == "sk-test"
        assert b.name == "openai:gpt-4o-mini"
        assert b.base_url == "https://api.openai.com/v1"


def test_anthropic_backend_requires_key() -> None:
    with _env(ANTHROPIC_API_KEY=None), pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()


def test_anthropic_calls_messages_create() -> None:
    block = mock.MagicMock()
    block.text = "anthropic reply"
    fake_msg = mock.MagicMock()
    fake_msg.content = [block]
    fake_client = mock.MagicMock()
    fake_client.messages.create.return_value = fake_msg

    fake_anthropic_module = mock.MagicMock()
    fake_anthropic_module.Anthropic.return_value = fake_client

    with (
        mock.patch.dict("sys.modules", {"anthropic": fake_anthropic_module}),
        _env(ANTHROPIC_API_KEY="a-test"),
    ):
        b = AnthropicBackend(model="claude-test")
        out = b.generate("hi", max_tokens=50)
    assert out == "anthropic reply"
    create_call = fake_client.messages.create.call_args
    assert create_call.kwargs["model"] == "claude-test"
    assert create_call.kwargs["max_tokens"] == 50


def test_backend_from_env_prefers_explicit_choice() -> None:
    with _env(SOMA_LLM_BACKEND=None, OPENAI_API_KEY=None, ANTHROPIC_API_KEY=None):
        with mock.patch("soma.llm.backends._ollama_alive", return_value=False):
            b = backend_from_env(prefer="ollama")
        assert isinstance(b, OllamaBackend)


def test_backend_from_env_uses_openai_when_key_set() -> None:
    with _env(
        SOMA_LLM_BACKEND=None,
        OPENAI_API_KEY="sk-test",
        ANTHROPIC_API_KEY=None,
    ):
        b = backend_from_env()
    assert isinstance(b, OpenAIBackend)


def test_backend_from_env_uses_anthropic_when_only_key_set() -> None:
    with _env(
        SOMA_LLM_BACKEND=None,
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY="a-test",
    ):
        b = backend_from_env()
    assert isinstance(b, AnthropicBackend)


def test_backend_from_env_falls_back_to_ollama_when_alive() -> None:
    with (
        _env(
            SOMA_LLM_BACKEND=None,
            OPENAI_API_KEY=None,
            ANTHROPIC_API_KEY=None,
        ),
        mock.patch("soma.llm.backends._ollama_alive", return_value=True),
    ):
        b = backend_from_env()
    assert isinstance(b, OllamaBackend)


def test_backend_from_env_falls_back_to_hf_when_nothing_else() -> None:
    with (
        _env(
            SOMA_LLM_BACKEND=None,
            OPENAI_API_KEY=None,
            ANTHROPIC_API_KEY=None,
        ),
        mock.patch("soma.llm.backends._ollama_alive", return_value=False),
    ):
        b = backend_from_env()
    from soma.llm import HuggingFaceBackend

    assert isinstance(b, HuggingFaceBackend)


def test_backend_from_env_openai_compat_needs_base_url() -> None:
    with (
        _env(
            SOMA_LLM_BACKEND="openai-compat",
            SOMA_LLM_BASE_URL=None,
            SOMA_LLM_MODEL=None,
        ),
        pytest.raises(ValueError, match="SOMA_LLM_BASE_URL"),
    ):
        backend_from_env()


def test_backend_from_env_rejects_unknown_choice() -> None:
    with (
        _env(SOMA_LLM_BACKEND="totally-fake"),
        pytest.raises(ValueError, match="unknown SOMA_LLM_BACKEND"),
    ):
        backend_from_env()


def test_ollama_alive_returns_false_on_url_error() -> None:
    import urllib.error

    from soma.llm.backends import _ollama_alive

    def boom(_url, timeout):
        raise urllib.error.URLError("nope")

    with mock.patch("urllib.request.urlopen", boom):
        assert _ollama_alive("http://nowhere:11434") is False
