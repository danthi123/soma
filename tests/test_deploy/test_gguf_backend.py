"""Tests for the GGUF inference-only backend.

These tests mock the ``llama-cpp-python`` boundary entirely. We never
load a real GGUF file -- production GGUFs are 1-30 GB and live in the
operator's LM Studio cache, not in the repo. The factory's only job is
to wire ``gguf_path`` -> ``Llama(model_path=...)`` and surface a
narrow inference-only API; the tests assert exactly that contract,
including the explicit-rejection ``__getattr__`` that prevents callers
from accidentally treating a GGUF backend as trainable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import soma.deploy.gguf_backend as gguf_mod
from soma.deploy.gguf_backend import GGUFChatHead, build_gguf_chat_head


class _FakeLlama:
    """Minimal stand-in for ``llama_cpp.Llama``.

    Captures construction kwargs so tests can assert correct plumbing,
    and returns a deterministic completion dict shaped like the real
    ``Llama.__call__`` output.
    """

    last_init_kwargs: dict[str, Any] = {}
    last_call_args: tuple[Any, ...] = ()
    last_call_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_init_kwargs = kwargs

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        type(self).last_call_args = (prompt,)
        type(self).last_call_kwargs = kwargs
        return {"choices": [{"text": "fake-completion"}]}

    def tokenize(self, data: bytes) -> list[int]:
        # Trivial 1-byte-per-token stand-in; real llama.cpp does BPE.
        return list(data)

    def detokenize(self, tokens: list[int]) -> bytes:
        return bytes(tokens)


@pytest.fixture
def fake_llama(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLlama]:
    """Patch ``Llama`` and ``_HAS_LLAMA_CPP`` so the module thinks it's installed."""
    monkeypatch.setattr(gguf_mod, "Llama", _FakeLlama)
    monkeypatch.setattr(gguf_mod, "_HAS_LLAMA_CPP", True)
    # Reset shared class-level state between tests.
    _FakeLlama.last_init_kwargs = {}
    _FakeLlama.last_call_args = ()
    _FakeLlama.last_call_kwargs = {}
    return _FakeLlama


@pytest.fixture
def gguf_file(tmp_path: Path) -> Path:
    """Create an empty file on disk that passes the ``exists()`` check."""
    p = tmp_path / "fake.gguf"
    p.write_bytes(b"\x00")  # contents irrelevant; mock never reads it
    return p


# ----- import-availability gate --------------------------------------------


def test_build_gguf_chat_head_without_llama_cpp_raises(
    monkeypatch: pytest.MonkeyPatch, gguf_file: Path
) -> None:
    """When llama-cpp-python is missing, construction must raise ImportError
    with a hint pointing at the optional extras install command."""
    monkeypatch.setattr(gguf_mod, "_HAS_LLAMA_CPP", False)
    with pytest.raises(ImportError, match=r"llama-cpp-python.*\.\[gguf\]"):
        build_gguf_chat_head(gguf_path=gguf_file)


# ----- file existence gate -------------------------------------------------


def test_build_gguf_chat_head_missing_file_raises(
    fake_llama: type[_FakeLlama], tmp_path: Path
) -> None:
    """Nonexistent GGUF path -> FileNotFoundError, before Llama() runs."""
    missing = tmp_path / "no-such.gguf"
    with pytest.raises(FileNotFoundError, match="GGUF file not found"):
        build_gguf_chat_head(gguf_path=missing)


# ----- happy-path construction --------------------------------------------


def test_build_gguf_chat_head_passes_kwargs_to_llama(
    fake_llama: type[_FakeLlama], gguf_file: Path
) -> None:
    """Factory plumbs ``n_ctx`` / ``n_gpu_layers`` to the underlying Llama init."""
    head = build_gguf_chat_head(gguf_path=gguf_file, n_ctx=2048, n_gpu_layers=20)
    assert isinstance(head, GGUFChatHead)
    assert fake_llama.last_init_kwargs["model_path"] == str(gguf_file)
    assert fake_llama.last_init_kwargs["n_ctx"] == 2048
    assert fake_llama.last_init_kwargs["n_gpu_layers"] == 20
    # Defaults: verbose=False is the operator-friendly choice.
    assert fake_llama.last_init_kwargs["verbose"] is False


def test_build_gguf_chat_head_default_offloads_all_layers(
    fake_llama: type[_FakeLlama], gguf_file: Path
) -> None:
    """Default ``n_gpu_layers=-1`` means full GPU offload."""
    build_gguf_chat_head(gguf_path=gguf_file)
    assert fake_llama.last_init_kwargs["n_gpu_layers"] == -1
    assert fake_llama.last_init_kwargs["n_ctx"] == 4096


# ----- training-API rejection ---------------------------------------------


def test_gguf_supports_gradients_is_false(fake_llama: type[_FakeLlama], gguf_file: Path) -> None:
    """Explicit assertion of the inference-only invariant."""
    head = build_gguf_chat_head(gguf_path=gguf_file)
    assert head.supports_gradients is False


@pytest.mark.parametrize("attr", ["model", "get_input_embeddings", "parameters"])
def test_gguf_rejects_chathead_training_api(
    fake_llama: type[_FakeLlama], gguf_file: Path, attr: str
) -> None:
    """Accessing trainable-LLM attributes must raise AttributeError with hint."""
    head = build_gguf_chat_head(gguf_path=gguf_file)
    with pytest.raises(AttributeError, match=r"GGUF is inference-only"):
        getattr(head, attr)


def test_gguf_unknown_attribute_raises_plain_attribute_error(
    fake_llama: type[_FakeLlama], gguf_file: Path
) -> None:
    """Other attribute typos must still raise AttributeError, just without
    the training-API hint, so the standard ``hasattr`` protocol works."""
    head = build_gguf_chat_head(gguf_path=gguf_file)
    with pytest.raises(AttributeError):
        head.totally_unrelated_attr  # noqa: B018  -- intentional access for the raise


# ----- generate_text plumbing ---------------------------------------------


def test_generate_text_returns_llama_output(fake_llama: type[_FakeLlama], gguf_file: Path) -> None:
    """Factory plumbs the prompt + generation kwargs to Llama() correctly."""
    head = build_gguf_chat_head(gguf_path=gguf_file)
    out = head.generate_text(
        prompt="Hello there",
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        stop=["\n"],
    )
    assert out == "fake-completion"
    assert fake_llama.last_call_args == ("Hello there",)
    assert fake_llama.last_call_kwargs["max_tokens"] == 32
    assert fake_llama.last_call_kwargs["temperature"] == 0.7
    assert fake_llama.last_call_kwargs["top_p"] == 0.9
    assert fake_llama.last_call_kwargs["stop"] == ["\n"]


def test_generate_text_silently_ignores_hf_kwargs(
    fake_llama: type[_FakeLlama], gguf_file: Path
) -> None:
    """Unknown HF-style kwargs (attention_mask, inputs_embeds) must not crash --
    callers that re-use HF call sites may pass them by accident."""
    head = build_gguf_chat_head(gguf_path=gguf_file)
    out = head.generate_text(
        prompt="Hi",
        max_new_tokens=4,
        attention_mask="ignored",
        inputs_embeds="also ignored",
    )
    assert out == "fake-completion"


def test_generate_text_default_stop_is_empty_list(
    fake_llama: type[_FakeLlama], gguf_file: Path
) -> None:
    """``stop=None`` resolves to an empty list, never a ``None`` leak to llama.cpp."""
    head = build_gguf_chat_head(gguf_path=gguf_file)
    head.generate_text(prompt="Hi", max_new_tokens=1)
    assert fake_llama.last_call_kwargs["stop"] == []


# ----- tokenize / detokenize roundtrip ------------------------------------


def test_tokenize_returns_int_list(fake_llama: type[_FakeLlama], gguf_file: Path) -> None:
    head = build_gguf_chat_head(gguf_path=gguf_file)
    tokens = head.tokenize("ABC")
    # Our fake tokenizer uses raw bytes; "ABC" -> [65, 66, 67].
    assert tokens == [65, 66, 67]
    assert all(isinstance(t, int) for t in tokens)


def test_detokenize_returns_str(fake_llama: type[_FakeLlama], gguf_file: Path) -> None:
    head = build_gguf_chat_head(gguf_path=gguf_file)
    text = head.detokenize([72, 105])
    assert text == "Hi"


def test_tokenize_detokenize_roundtrip_through_fake(
    fake_llama: type[_FakeLlama], gguf_file: Path
) -> None:
    """The fake is byte-for-byte invertible; this asserts the wrapper
    doesn't accidentally double-encode or strip the result."""
    head = build_gguf_chat_head(gguf_path=gguf_file)
    text = "soma"
    assert head.detokenize(head.tokenize(text)) == text


# ----- gguf_path property -------------------------------------------------


def test_gguf_path_property_exposes_loaded_file(
    fake_llama: type[_FakeLlama], gguf_file: Path
) -> None:
    head = build_gguf_chat_head(gguf_path=gguf_file)
    assert head.gguf_path == gguf_file
