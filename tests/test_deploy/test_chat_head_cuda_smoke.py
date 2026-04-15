"""Slow+cuda smoke: Qwen2.5-1.5B-Instruct on GPU, fp16.

Opt-in only -- requires real HF model download (~3GB in fp16) and a
CUDA device. Run with ``pytest -v -m "slow and cuda"``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


def test_qwen15b_loads_and_generates_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")

    from soma.deploy.chat_head_factory import build_chat_head
    from soma.deploy.devices import select_device_and_dtype

    device, dtype = select_device_and_dtype()
    head = build_chat_head(tier="small", device=device, dtype=dtype)

    # Smoke: can the model produce a coherent response via the
    # inputs_embeds path SOMA uses in production?
    prefix = torch.zeros(1, 4, head.hidden_size, dtype=dtype, device=device)
    ids = head.tokenizer("Hello, world.", return_tensors="pt")["input_ids"].to(device)
    token_embeds = head.model.get_input_embeddings()(ids)
    # This concat mirrors SOMA.chat / ChatSession.respond; both were fixed
    # in T3 to cast prefix to token_embeds.dtype. Since we constructed
    # ``prefix`` already in ``dtype`` we don't need the cast here, but
    # verify anyway by asserting no RuntimeError.
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
    attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)

    text = head.generate_text(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=10,
        min_new_tokens=1,
        do_sample=False,
    )
    assert isinstance(text, str)
    assert len(text) > 0
    print(f"Qwen2.5-1.5B on {device} {dtype}: {text!r}")
