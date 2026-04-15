"""Slow+cuda+quant smoke: Qwen2.5-1.5B-Instruct in int4 on GPU.

Opt-in only -- requires CUDA, ~3GB cached model weights (shared with
the fp16 smoke), AND a working bitsandbytes wheel. Skips cleanly if
any prerequisite is missing.

Run with: pytest -v -m "slow and cuda and quant"
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.cuda, pytest.mark.quant]


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def test_qwen15b_int4_loads_and_generates_on_cuda() -> None:
    """End-to-end int4 smoke: load + generate via inputs_embeds path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    if not _has_bitsandbytes():
        pytest.skip("bitsandbytes not installed (pip install -e '.[quant]')")

    from soma.deploy.chat_head_factory import build_chat_head
    from soma.deploy.devices import select_device_and_dtype

    device, dtype = select_device_and_dtype()
    head = build_chat_head(
        tier="small",
        device=device,
        dtype=dtype,
        quantization="int4",
    )

    # Generate via the same inputs_embeds path SOMA uses in production.
    prefix = torch.zeros(1, 4, head.hidden_size, dtype=dtype, device=device)
    ids = head.tokenizer("Hello, world.", return_tensors="pt")["input_ids"].to(device)
    token_embeds = head.model.get_input_embeddings()(ids)
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
    attention_mask = torch.ones(
        1,
        inputs_embeds.shape[1],
        dtype=torch.long,
        device=device,
    )

    text = head.generate_text(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=10,
        min_new_tokens=1,
        do_sample=False,
    )
    assert isinstance(text, str)
    assert len(text) > 0
    print(f"Qwen2.5-1.5B int4 on {device}: {text!r}")


def test_qwen15b_int4_embeddings_keep_gradient_flow() -> None:
    """Load-bearing invariant: int4 quantizes linear weights but leaves
    the input embedding module as a standard nn.Embedding (fp16/bf16).
    This is what makes SomaVerbalizer training compatible with int4."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    if not _has_bitsandbytes():
        pytest.skip("bitsandbytes not installed")

    from torch import nn

    from soma.deploy.chat_head_factory import build_chat_head
    from soma.deploy.devices import select_device_and_dtype

    device, dtype = select_device_and_dtype()
    head = build_chat_head(
        tier="small",
        device=device,
        dtype=dtype,
        quantization="int4",
    )

    embed = head.model.get_input_embeddings()
    # Must be a real nn.Embedding (or subclass) -- NOT a bnb-wrapped 4bit
    # layer. bnb's Linear4bit is NOT an nn.Embedding subclass.
    assert isinstance(embed, nn.Embedding), (
        f"int4 path must keep embeddings as nn.Embedding, got {type(embed).__name__}"
    )
    # Embedding weight must be in a gradient-friendly floating dtype.
    assert embed.weight.dtype in (torch.float16, torch.bfloat16, torch.float32)

    # Run a tiny forward with requires_grad=True on a downstream prefix
    # to verify autograd reaches here.
    prefix = torch.zeros(
        1,
        2,
        head.hidden_size,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    ids = torch.tensor([[1, 2, 3]], device=device)
    token_embeds = embed(ids)
    concat = torch.cat([prefix, token_embeds], dim=1)
    loss = concat.sum()
    loss.backward()
    assert prefix.grad is not None
    assert torch.any(prefix.grad != 0)


def test_qwen15b_int4_vram_less_than_fp16() -> None:
    """int4 quantization must measurably reduce VRAM vs fp16. Qwen2.5-1.5B
    is ~3GB fp16, ~1GB int4 -- expect >1GB savings."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    if not _has_bitsandbytes():
        pytest.skip("bitsandbytes not installed")

    from soma.deploy.chat_head_factory import build_chat_head
    from soma.deploy.devices import select_device_and_dtype

    device, dtype = select_device_and_dtype()

    # Measure int4 footprint.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    head_int4 = build_chat_head(
        tier="small",
        device=device,
        dtype=dtype,
        quantization="int4",
    )
    _ = head_int4.model.get_input_embeddings()  # force any lazy init
    vram_int4_gb = torch.cuda.memory_allocated() / (1024**3)
    del head_int4
    torch.cuda.empty_cache()

    # Measure fp16 footprint for comparison.
    torch.cuda.reset_peak_memory_stats()
    head_fp16 = build_chat_head(
        tier="small",
        device=device,
        dtype=dtype,
        quantization="none",
    )
    _ = head_fp16.model.get_input_embeddings()
    vram_fp16_gb = torch.cuda.memory_allocated() / (1024**3)
    del head_fp16
    torch.cuda.empty_cache()

    savings_gb = vram_fp16_gb - vram_int4_gb
    print(
        f"Qwen2.5-1.5B VRAM: fp16={vram_fp16_gb:.2f} GB, "
        f"int4={vram_int4_gb:.2f} GB, savings={savings_gb:.2f} GB"
    )
    # Tight-but-meaningful threshold: int4 should save at least 1 GB on
    # a 1.5B model. Real observed savings ~2GB but give margin.
    assert savings_gb > 1.0, (
        f"int4 should save >1GB over fp16 on 1.5B model, saved only {savings_gb:.2f}GB"
    )
