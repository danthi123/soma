"""Concrete LLM backends + the backend protocol.

Every backend exposes one method: ``generate(prompt) -> str``. Callers
can swap backends without touching the prompt-building or retrieval
sides.

All optional SDK imports are lazy so installing SOMA doesn't pull in
``openai`` / ``anthropic`` / ``transformers`` etc. unless you need them.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal protocol any backend must satisfy.

    Backends MAY additionally implement an optional streaming method::

        def stream_generate(
            self, prompt: str, *, max_tokens: int = 256
        ) -> Iterator[str]:
            ...

    When present, callers that care about live token output (``soma
    chat``) detect it via ``hasattr(backend, "stream_generate")`` and
    pipe chunks to stdout as they arrive. Backends without it keep
    working identically — the optional method is additive.
    """

    name: str

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        """Return the model's completion for ``prompt`` (greedy/default)."""
        ...


# ----------------------------------------------------------------------
# Dry-run / testing backend
# ----------------------------------------------------------------------


@dataclass
class DryRunBackend:
    """No-op backend that echoes the prompt's context section.

    Useful in tests + ``--dry-run`` modes — exercises the full RAG
    pipeline (retrieve, prompt build) without loading or calling a real
    model.
    """

    name: str = "dry-run"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        ctx_lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("[")]
        if not ctx_lines:
            return "[dry-run] no context blocks in prompt"
        head = ctx_lines[0][:140]
        return f"[dry-run] would answer using {len(ctx_lines)} chunks; first: {head}"


# ----------------------------------------------------------------------
# Ollama (local HTTP, no API key)
# ----------------------------------------------------------------------


@dataclass
class OllamaBackend:
    """Talks to a locally-running Ollama server (default ``localhost:11434``).

    Install Ollama (https://ollama.com), pull a model
    (``ollama pull llama3.2``), and you're done — no API key, no extra
    Python deps.
    """

    model: str = "llama3.2"
    host: str = "http://localhost:11434"
    timeout: float = 120.0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"ollama:{self.model}"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        import json

        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.0},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama server unreachable at {self.host}: {exc}. "
                "Install Ollama (https://ollama.com) and run `ollama serve`."
            ) from exc
        return str(payload.get("response", "")).strip()

    def stream_generate(
        self, prompt: str, *, max_tokens: int = 256
    ) -> Iterator[str]:
        """Stream tokens from Ollama's ``/api/generate`` (stream=True).

        Ollama returns newline-delimited JSON on the response body: one
        object per incremental token (or small token group), each with a
        ``response`` field carrying the delta, and a final object with
        ``done: true`` (and an empty ``response``).
        """
        import json

        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens, "temperature": 0.0},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = str(payload.get("response", ""))
                    if chunk:
                        yield chunk
                    if payload.get("done"):
                        return
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama server unreachable at {self.host}: {exc}. "
                "Install Ollama (https://ollama.com) and run `ollama serve`."
            ) from exc


# ----------------------------------------------------------------------
# OpenAI-compatible (OpenAI cloud + vLLM / LM Studio / LiteLLM / etc.)
# ----------------------------------------------------------------------


@dataclass
class OpenAICompatibleBackend:
    """Generic OpenAI-compatible chat-completions backend.

    Works against any server speaking OpenAI's HTTP API: vLLM, LM Studio,
    LiteLLM proxy, llama.cpp's ``--server``, or the real OpenAI cloud.
    Requires the ``openai`` Python SDK.
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    timeout: float = 120.0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"openai-compat:{self.model}"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleBackend needs the openai SDK. Install: pip install openai"
            ) from exc
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream_generate(
        self, prompt: str, *, max_tokens: int = 256
    ) -> Iterator[str]:
        """Stream chunks from an OpenAI-compatible chat-completions API.

        Works against OpenAI cloud, LM Studio, vLLM, LiteLLM, llama.cpp's
        built-in server — anything that speaks the OpenAI chat protocol.
        Role-only / stop-reason chunks have ``delta.content`` of ``None``
        or ``""`` and are skipped.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleBackend needs the openai SDK. Install: pip install openai"
            ) from exc
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        stream = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        for event in stream:
            choices = getattr(event, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content


@dataclass
class OpenAIBackend(OpenAICompatibleBackend):
    """OpenAI cloud (uses ``OPENAI_API_KEY`` env by default)."""

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "OpenAIBackend needs an api key. Set OPENAI_API_KEY or pass api_key=..."
            )
        self.name = f"openai:{self.model}"


# ----------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------


@dataclass
class AnthropicBackend:
    """Anthropic Claude API (uses ``ANTHROPIC_API_KEY`` env by default)."""

    model: str = "claude-haiku-4-5-20251001"
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    timeout: float = 120.0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "AnthropicBackend needs an api key. Set ANTHROPIC_API_KEY or pass api_key=..."
            )
        self.name = f"anthropic:{self.model}"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicBackend needs the anthropic SDK. Install: pip install anthropic"
            ) from exc
        client = Anthropic(api_key=self.api_key, timeout=self.timeout)
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # messages.content is a list of blocks; concatenate text blocks.
        parts = [getattr(b, "text", "") for b in msg.content]
        return "".join(parts).strip()

    def stream_generate(
        self, prompt: str, *, max_tokens: int = 256
    ) -> Iterator[str]:
        """Stream text deltas from Anthropic's ``messages.stream`` API.

        ``client.messages.stream(...)`` returns a context manager whose
        ``text_stream`` attribute yields the incremental text as str.
        We swallow empty deltas — Anthropic occasionally emits them at
        block boundaries.
        """
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicBackend needs the anthropic SDK. Install: pip install anthropic"
            ) from exc
        client = Anthropic(api_key=self.api_key, timeout=self.timeout)
        with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


# ----------------------------------------------------------------------
# HuggingFace local (wraps soma.deploy.chat_head_factory)
# ----------------------------------------------------------------------


@dataclass
class HuggingFaceBackend:
    """Local HuggingFace model via :mod:`soma.deploy.chat_head_factory`.

    Loads on first ``generate`` call (lazy). Tier ``auto`` lets the
    deploy module pick a model that fits available VRAM/RAM.
    """

    tier: str = "auto"
    device: str | None = None
    dtype: str | None = None
    quantization: str = "none"
    name: str = field(init=False)
    _chat_head: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"hf:{self.tier}"

    def _load(self) -> Any:
        if self._chat_head is not None:
            return self._chat_head
        import argparse as _ap

        from soma.deploy.chat_head_factory import build_chat_head
        from soma.deploy.cli import resolve_device_dtype_tier

        ns = _ap.Namespace(
            tier=self.tier,
            llm_name=None,
            device=self.device,
            dtype=self.dtype,
            quantization=self.quantization,
        )
        device, dtype, _llm_name, resolved_tier = resolve_device_dtype_tier(ns)
        assert resolved_tier is not None
        self._chat_head = build_chat_head(tier=resolved_tier, device=device, dtype=dtype)
        return self._chat_head

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        import torch

        ch = self._load()
        hf_tok = ch.tokenizer
        device = ch.model.get_input_embeddings().weight.device
        inputs = hf_tok(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            out = ch.model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False)
        new_ids = out[0][input_ids.shape[1] :]
        text: str = hf_tok.decode(new_ids, skip_special_tokens=True)
        return text.strip()


# ----------------------------------------------------------------------
# Auto-selection from environment
# ----------------------------------------------------------------------


def backend_from_env(*, prefer: str | None = None) -> LLMBackend:
    """Pick a backend based on env vars / availability.

    Order of preference (override with ``prefer=...``):

    1. ``SOMA_LLM_BACKEND`` env var if set: one of
       ``ollama``, ``openai``, ``anthropic``, ``openai-compat``, ``hf``.
    2. ``OPENAI_API_KEY`` set → :class:`OpenAIBackend`.
    3. ``ANTHROPIC_API_KEY`` set → :class:`AnthropicBackend`.
    4. Ollama reachable at ``localhost:11434`` → :class:`OllamaBackend`.
    5. Fallback: :class:`HuggingFaceBackend` (loads a local HF model).

    Per-backend env knobs (only read when that backend is chosen):

    - ``SOMA_LLM_MODEL``: model name to pass through.
    - ``SOMA_LLM_BASE_URL``: base URL (openai-compat / ollama).
    """
    choice = (prefer or os.environ.get("SOMA_LLM_BACKEND") or "").lower()
    model = os.environ.get("SOMA_LLM_MODEL")
    base_url = os.environ.get("SOMA_LLM_BASE_URL")

    if not choice:
        if os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif _ollama_alive(base_url or "http://localhost:11434"):
            choice = "ollama"
        else:
            choice = "hf"

    if choice == "ollama":
        return OllamaBackend(
            model=model or "llama3.2",
            host=base_url or "http://localhost:11434",
        )
    if choice == "openai":
        return OpenAIBackend(model=model or "gpt-4o-mini")
    if choice == "anthropic":
        return AnthropicBackend(model=model or "claude-haiku-4-5-20251001")
    if choice == "openai-compat":
        if not base_url:
            raise ValueError("openai-compat backend needs SOMA_LLM_BASE_URL or base_url= ...")
        return OpenAICompatibleBackend(
            model=model or "local",
            base_url=base_url,
        )
    if choice == "hf":
        return HuggingFaceBackend()
    raise ValueError(f"unknown SOMA_LLM_BACKEND: {choice!r}")


def _ollama_alive(host: str) -> bool:
    """Quick reachability probe — 1s timeout, doesn't block longer."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1.0) as r:
            status = int(r.status)
            return 200 <= status < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False
