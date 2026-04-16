"""LLM backend abstraction for SOMA's RAG layer.

SOMA's MemoryLayer is LLM-agnostic by design — ``store(text)`` /
``retrieve(query)`` return text and metadata, and what you do with
them is up to you. Most callers, though, want the obvious next step:
take the retrieved chunks, hand them to an LLM with a prompt, and get
a grounded answer back. This module ships that wiring as a small
pluggable layer so any LLM can drive the conversation without each
caller writing the same glue.

Usage::

    from soma.llm import OllamaBackend, RAGSession
    from soma.memory import MemoryLayer

    mem = MemoryLayer.load("brain/")
    chat = RAGSession(memory=mem, llm=OllamaBackend(model="llama3"))
    answer = chat.ask("where does the user live?")

The :class:`LLMBackend` protocol is one method (``generate``) so any
HTTP / SDK / local-model backend can be plugged in. We ship five:

- :class:`OllamaBackend` — talks to a local Ollama server (no API key).
- :class:`OpenAIBackend` — uses ``openai`` SDK + ``OPENAI_API_KEY``.
- :class:`AnthropicBackend` — uses ``anthropic`` SDK + ``ANTHROPIC_API_KEY``.
- :class:`OpenAICompatibleBackend` — generic OpenAI-compatible server
  (vLLM / LM Studio / llama.cpp / LiteLLM proxy / etc.).
- :class:`HuggingFaceBackend` — wraps :mod:`soma.deploy.chat_head_factory`
  for a fully-local HF model.

:func:`backend_from_env` picks one based on env vars so callers can
swap LLMs without code changes.
"""

from __future__ import annotations

from soma.llm.backends import (
    AnthropicBackend,
    DryRunBackend,
    HuggingFaceBackend,
    LLMBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenAICompatibleBackend,
    backend_from_env,
)
from soma.llm.rag import RAGSession

__all__ = [
    "AnthropicBackend",
    "DryRunBackend",
    "HuggingFaceBackend",
    "LLMBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "RAGSession",
    "backend_from_env",
]
