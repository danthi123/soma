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


# ----- quantization rejection paths ----------------------------------------


def test_build_chat_head_quantization_rejects_cpu() -> None:
    """int4 / int8 on CPU is a ValueError before any HF download is attempted."""
    from soma.deploy.chat_head_factory import build_chat_head

    for q in ("int4", "int8"):
        with pytest.raises(ValueError, match="CUDA"):
            build_chat_head(
                tier="small",
                device=torch.device("cpu"),
                dtype=torch.float16,
                quantization=q,  # type: ignore[arg-type]
            )


def test_build_chat_head_quantization_without_bitsandbytes_raises_int4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When bnb is missing, int4 raises ImportError with the install hint."""
    import soma.deploy.chat_head_factory as mod
    from soma.deploy.chat_head_factory import build_chat_head

    monkeypatch.setattr(mod, "_HAS_BITSANDBYTES", False)
    with pytest.raises(ImportError, match=r"bitsandbytes.*\[quant\]"):
        build_chat_head(
            tier="small",
            device=torch.device("cuda"),
            dtype=torch.float16,
            quantization="int4",
        )


def test_build_chat_head_quantization_without_bitsandbytes_raises_int8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When bnb is missing, int8 raises ImportError with the install hint."""
    import soma.deploy.chat_head_factory as mod
    from soma.deploy.chat_head_factory import build_chat_head

    monkeypatch.setattr(mod, "_HAS_BITSANDBYTES", False)
    with pytest.raises(ImportError, match=r"bitsandbytes.*\[quant\]"):
        build_chat_head(
            tier="small",
            device=torch.device("cuda"),
            dtype=torch.float16,
            quantization="int8",
        )


# ----- quantization config-passing -----------------------------------------


def _patch_hf_capturing(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hidden_size: int,
    captured_kwargs: dict[str, Any],
) -> None:
    """Patch HF + presence flag; capture kwargs the factory passes to from_pretrained.

    Lets quantisation tests inspect the ``quantization_config`` and
    ``device_map`` that the factory hands to ``AutoModelForCausalLM`` without
    requiring real bitsandbytes wheels at test time.
    """

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name: str, **kw: Any) -> _MockAutoModel:
            captured_kwargs.update(kw)
            return _MockAutoModel(hidden_size=hidden_size)

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kw: Any) -> _MockAutoTokenizer:
            return _MockAutoTokenizer()

    import soma.deploy.chat_head_factory as mod

    monkeypatch.setattr(mod, "AutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(mod, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(mod, "_HAS_BITSANDBYTES", True)


def test_build_chat_head_quantization_int4_passes_config_to_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """int4 path constructs an nf4 BitsAndBytesConfig and passes it via kwargs."""
    pytest.importorskip("transformers")
    from transformers import BitsAndBytesConfig

    from soma.deploy.chat_head_factory import build_chat_head

    captured: dict[str, Any] = {}
    _patch_hf_capturing(monkeypatch, hidden_size=1536, captured_kwargs=captured)

    head = build_chat_head(
        tier="small",
        device=torch.device("cuda"),
        dtype=torch.float16,
        quantization="int4",
    )

    assert isinstance(head, ChatHead)
    qcfg = captured.get("quantization_config")
    assert isinstance(qcfg, BitsAndBytesConfig)
    assert qcfg.load_in_4bit is True
    assert qcfg.bnb_4bit_quant_type == "nf4"
    assert qcfg.bnb_4bit_use_double_quant is True
    assert qcfg.bnb_4bit_compute_dtype == torch.float16
    # device_map pins to single CUDA device (not "auto" sharding).
    assert captured.get("device_map") == {"": "cuda"}


def test_build_chat_head_quantization_int8_passes_config_to_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """int8 path constructs a load_in_8bit BitsAndBytesConfig and passes it."""
    pytest.importorskip("transformers")
    from transformers import BitsAndBytesConfig

    from soma.deploy.chat_head_factory import build_chat_head

    captured: dict[str, Any] = {}
    _patch_hf_capturing(monkeypatch, hidden_size=1536, captured_kwargs=captured)

    build_chat_head(
        tier="small",
        device=torch.device("cuda"),
        dtype=torch.float16,
        quantization="int8",
    )

    qcfg = captured.get("quantization_config")
    assert isinstance(qcfg, BitsAndBytesConfig)
    assert qcfg.load_in_8bit is True
    assert captured.get("device_map") == {"": "cuda"}


def test_build_chat_head_quantization_none_does_not_set_quant_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default quantization='none' must NOT inject quantization_config or device_map.

    Guards against a future regression where the bnb branch leaks into the
    fp16 path -- the existing fp16 ChatHead behaviour must stay byte-identical.
    """
    from soma.deploy.chat_head_factory import build_chat_head

    captured: dict[str, Any] = {}
    _patch_hf_capturing(monkeypatch, hidden_size=1536, captured_kwargs=captured)

    build_chat_head(
        tier="small",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert "quantization_config" not in captured
    assert "device_map" not in captured
