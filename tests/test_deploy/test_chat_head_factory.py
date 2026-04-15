"""Tests for the tier-selected ChatHead factory.

These tests monkeypatch ``AutoModelForCausalLM`` and ``AutoTokenizer`` so no
real HuggingFace download is ever triggered. The factory's job is to wire
a tier name -> registered hidden_dim -> loaded model.config.hidden_size
check, and hand the result off to the existing ``ChatHead`` wrapper.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from soma.io.chat_head import ChatHead


class _MockAutoModel:
    """Minimal stand-in for a HF causal LM with trainable params."""

    def __init__(self, *, hidden_size: int, vocab_size: int = 32):
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = type("Cfg", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def parameters(self) -> Any:
        yield from self.embed.parameters()
        yield from self.lm_head.parameters()

    def to(self, device_or_dtype: Any) -> _MockAutoModel:
        return self

    def train(self, mode: bool = True) -> _MockAutoModel:
        return self


class _MockAutoTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "hi"


def _patch_hf(monkeypatch: pytest.MonkeyPatch, *, hidden_size: int) -> None:
    """Replace the factory's HF classes with mocks returning our shapes."""

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kw: Any) -> _MockAutoModel:
            return _MockAutoModel(hidden_size=hidden_size)

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kw: Any) -> _MockAutoTokenizer:
            return _MockAutoTokenizer()

    import soma.deploy.chat_head_factory as mod

    monkeypatch.setattr(mod, "AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(mod, "AutoTokenizer", _FakeAutoTokenizer)


def test_build_chat_head_tier_unknown_raises() -> None:
    """Unknown tier name yields a clear ValueError, not a KeyError."""
    from soma.deploy.chat_head_factory import build_chat_head

    with pytest.raises(ValueError, match="unknown tier"):
        build_chat_head(
            tier="nonsense",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_build_chat_head_hidden_dim_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded model with wrong hidden_size raises, guarding registry drift."""
    from soma.deploy.chat_head_factory import build_chat_head

    _patch_hf(monkeypatch, hidden_size=999)  # registry says 1536 for small
    with pytest.raises(ValueError, match="hidden_dim mismatch"):
        build_chat_head(
            tier="small",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_build_chat_head_returns_chathead_on_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path returns a ChatHead with frozen weights and correct hidden_size."""
    from soma.deploy.chat_head_factory import build_chat_head

    _patch_hf(monkeypatch, hidden_size=1536)  # matches tier="small"
    head = build_chat_head(
        tier="small",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert isinstance(head, ChatHead)
    assert head.hidden_size == 1536
    for p in head.model.parameters():
        assert not p.requires_grad, "ChatHead must freeze all base-model params"
