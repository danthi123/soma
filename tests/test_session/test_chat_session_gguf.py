"""Track D: ChatSession GGUF-backend branch tests.

Covers the inference-only path where ``gguf_head`` replaces ``chat_head``
and the verbalizer is deliberately skipped. Uses a fake
:class:`_FakeGGUFHead` so we don't drag in ``llama-cpp-python``.

Reuses the ``_soma_cfg`` + ``_shared_tokenizer`` module-private helpers
from ``test_chat_session`` rather than duplicating the 20-line config
block.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from soma.io.chat_head import ChatHead
from soma.io.text_encoder import TextEncoder
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession
from soma.system import SOMA
from tests.test_session.test_chat_session import _shared_tokenizer, _soma_cfg


class _FakeGGUFHead:
    """Minimal stand-in: ``supports_gradients`` + ``generate_text(prompt=..)``."""

    def __init__(self, canned: str = "canned-gguf-response") -> None:
        self.canned = canned
        self.calls: list[dict[str, Any]] = []

    @property
    def supports_gradients(self) -> bool:
        return False

    def generate_text(self, *, prompt: str, **kw: Any) -> str:
        self.calls.append({"prompt": prompt, **kw})
        return self.canned


class _DummyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 16)
        self.config = type("Cfg", (), {"hidden_size": 16, "vocab_size": 8})()

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed


def _encoder() -> TextEncoder:
    cfg = _soma_cfg()
    return TextEncoder(
        _shared_tokenizer, embed_dim=cfg.text_embed_dim, max_seq_len=cfg.max_input_tokens
    )


def _fresh_soma() -> SOMA:
    return SOMA(_soma_cfg(), device=torch.device("cpu"))


def _fresh_verbalizer() -> SomaVerbalizer:
    spec = VerbalizerSpec(
        soma_output_dim=_soma_cfg().sensor_output_dim,
        llm_name="mock",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    return SomaVerbalizer(spec)


def _build_gguf_session(canned: str = "mock-response") -> tuple[ChatSession, _FakeGGUFHead]:
    gguf = _FakeGGUFHead(canned=canned)
    session = ChatSession(
        soma=_fresh_soma(),
        chat_head=None,
        gguf_head=gguf,
        tokenizer=_shared_tokenizer,
        encoder=_encoder(),
    )
    return session, gguf


# -- construction -----------------------------------------------------------


def test_gguf_session_constructs_cleanly() -> None:
    session, gguf = _build_gguf_session()
    assert session.gguf_head is gguf
    assert session.chat_head is None
    assert session.verbalizer is None
    assert session.online_trainer is None
    assert session.history == []


# -- respond() flow ---------------------------------------------------------


def test_gguf_respond_returns_fake_canned_output() -> None:
    session, gguf = _build_gguf_session(canned="hi-from-gguf")
    out = session.respond(user_text="hello there")
    assert out == "hi-from-gguf"
    assert len(gguf.calls) == 1
    assert gguf.calls[0]["prompt"] == "hello there"


def test_gguf_respond_feeds_soma_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """user_text + response both flow through SOMA."""
    session, _gguf = _build_gguf_session()
    calls: list[str] = []
    original = session._feed_text_through_soma

    def _spy(text: str) -> None:
        calls.append(text)
        original(text)

    monkeypatch.setattr(session, "_feed_text_through_soma", _spy)
    out = session.respond(user_text="user-prompt")
    assert calls == ["user-prompt", out]


def test_gguf_respond_skips_verbalizer_and_hf_aggregator() -> None:
    """GGUF mode must not take the HF branch. ``SomaAggregator.collapse``
    is also called inside ``text_to_state`` (2x, one per feed); the HF
    branch would add a 3rd. Counting distinguishes the paths.
    """
    from soma.io import verbalizer as verb_mod

    session, _gguf = _build_gguf_session()
    assert session.verbalizer is None
    collapse_calls = 0
    original = verb_mod.SomaAggregator.collapse

    def _spy(*a: Any, **kw: Any) -> Any:
        nonlocal collapse_calls
        collapse_calls += 1
        return original(*a, **kw)

    verb_mod.SomaAggregator.collapse = staticmethod(_spy)  # type: ignore[method-assign]
    try:
        out = session.respond(user_text="hello")
    finally:
        verb_mod.SomaAggregator.collapse = staticmethod(original)  # type: ignore[method-assign]
    assert isinstance(out, str)
    assert collapse_calls == 2, f"expected 2, got {collapse_calls}"


def test_gguf_respond_advances_global_step() -> None:
    session, _gguf = _build_gguf_session()
    before = session.soma.global_step
    session.respond(user_text="alpha beta gamma")
    assert session.soma.global_step > before


def test_gguf_respond_appends_history_turns() -> None:
    session, _gguf = _build_gguf_session(canned="R1")
    session.respond(user_text="Q1")
    assert len(session.history) == 2
    assert session.history[0].role == "user"
    assert session.history[0].text == "Q1"
    assert session.history[1].role == "assistant"
    assert session.history[1].text == "R1"


def test_gguf_respond_forwards_unknown_gen_kwargs() -> None:
    """HF-specific kwargs flow into generate_text; real GGUFChatHead ignores them."""
    session, gguf = _build_gguf_session()
    session.respond(user_text="hi", max_new_tokens=8, do_sample=False, min_new_tokens=2)
    call = gguf.calls[0]
    assert call["max_new_tokens"] == 8
    assert call["do_sample"] is False
    assert call["min_new_tokens"] == 2


# -- __init__ validation ----------------------------------------------------


def test_session_neither_backend_raises() -> None:
    with pytest.raises(ValueError, match="exactly one of chat_head"):
        ChatSession(
            soma=_fresh_soma(),
            chat_head=None,
            gguf_head=None,
            tokenizer=_shared_tokenizer,
            encoder=_encoder(),
        )


def test_session_both_backends_raises() -> None:
    with pytest.raises(ValueError, match="both were provided"):
        ChatSession(
            soma=_fresh_soma(),
            verbalizer=_fresh_verbalizer(),
            chat_head=ChatHead(model=_DummyLM(), tokenizer=object()),
            gguf_head=_FakeGGUFHead(),
            tokenizer=_shared_tokenizer,
            encoder=_encoder(),
        )


def test_session_gguf_head_plus_online_trainer_raises() -> None:
    """GGUF + online_trainer is refused (no gradients through llama.cpp)."""
    with pytest.raises(ValueError, match="online verbalizer training"):
        ChatSession(
            soma=_fresh_soma(),
            chat_head=None,
            gguf_head=_FakeGGUFHead(),
            tokenizer=_shared_tokenizer,
            encoder=_encoder(),
            online_trainer=object(),
        )


def test_session_gguf_head_plus_verbalizer_raises() -> None:
    with pytest.raises(ValueError, match="gguf_head \\+ verbalizer"):
        ChatSession(
            soma=_fresh_soma(),
            verbalizer=_fresh_verbalizer(),
            chat_head=None,
            gguf_head=_FakeGGUFHead(),
            tokenizer=_shared_tokenizer,
            encoder=_encoder(),
        )


def test_session_chat_head_requires_verbalizer() -> None:
    with pytest.raises(ValueError, match="chat_head requires a verbalizer"):
        ChatSession(
            soma=_fresh_soma(),
            verbalizer=None,
            chat_head=ChatHead(model=_DummyLM(), tokenizer=object()),
            tokenizer=_shared_tokenizer,
            encoder=_encoder(),
        )
