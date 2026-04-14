"""Smoke test exercising a real SmolLM2-360M-Instruct download.

Marked ``slow`` -- skipped from default CI. Run manually:

    pip install -e ".[dev-chat]"   # ensure transformers is installed
    pytest tests/test_io/test_chat_head_smoke.py -v -m slow

Skips automatically when:
- ``transformers`` is not installed (via importorskip)
- ``HF_HUB_OFFLINE=1`` is set and the model isn't cached locally
"""

from __future__ import annotations

import os
from typing import Any, cast

import pytest
import torch

pytestmark = pytest.mark.slow


def _hub_is_reachable() -> bool:
    """Best-effort: skip gracefully if HF hub is unreachable (air-gapped CI)."""
    return os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true")


def test_smolm2_360m_generates_with_soma_prefix() -> None:
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soma.core.config import SOMAConfig
    from soma.io.chat_head import ChatHead
    from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
    from soma.system import SOMA

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Pin to CPU: the GPU may be in use by the training loop, and we only
    # need correctness, not speed, for this smoke test.
    device = torch.device("cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=not _hub_is_reachable(),
        )
        # transformers 5.x has inline types that confuse mypy on .to(device);
        # cast to Any keeps type-check clean without masking a real bug.
        model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=not _hub_is_reachable(),
            ),
        )
        model.to(device)
    except (OSError, ValueError) as exc:
        pytest.skip(f"SmolLM2-360M not downloadable in this environment: {exc}")

    chat_head = ChatHead(model=model, tokenizer=tokenizer)

    cfg = SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=2,
        wm_dim=8,
        episodic_capacity=4,
        key_dim=8,
        value_dim=8,
        vocab_size=16,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=device)

    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name=model_name,
        llm_hidden_dim=chat_head.hidden_size,  # 960 for SmolLM2-360M
        num_prefix_tokens=8,
    )
    verbalizer = SomaVerbalizer(spec).to(device)

    response = soma.chat(
        user_text="Hello, how are you?",
        verbalizer=verbalizer,
        chat_head=chat_head,
        max_new_tokens=20,
        do_sample=False,  # deterministic for assertion stability
    )
    assert isinstance(response, str)
    assert len(response) > 0
    # Near-null init (see test_verbalizer:test_untrained_verbalizer_emits_near_null_prefix)
    # means the prefix is ~zero, so the LLM behaves roughly vanilla. Just
    # print the output for human inspection rather than assert exact string.
    print(f"\nSmolLM2 response (near-null prefix): {response!r}")
