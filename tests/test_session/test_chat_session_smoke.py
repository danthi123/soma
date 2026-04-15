"""Slow-marked smoke: ChatSession with real SmolLM2-360M-Instruct, 3 turns.

Exercises the full Phase 5 pipeline:
    user text -> SOMA -> state -> verbalizer -> soft-prompt prefix ->
    frozen SmolLM2 -> response -> response-back-into-SOMA -> next turn

Run manually:
    pytest tests/test_session/test_chat_session_smoke.py -v -m slow
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow


def _hub_is_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true")


def test_three_turn_chat_with_smolm2(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from typing import Any, cast

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soma.core.config import SOMAConfig
    from soma.io.chat_head import ChatHead
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
    from soma.session.chat_session import ChatSession
    from soma.system import SOMA

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device = torch.device("cpu")  # GPU reserved for training service

    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=not _hub_is_reachable(),
        )
        hf_model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=not _hub_is_reachable(),
            ),
        ).to(device)
    except (OSError, ValueError) as exc:
        pytest.skip(f"SmolLM2-360M not available: {exc}")

    chat_head = ChatHead(model=hf_model, tokenizer=hf_tokenizer)

    # Tiny CPU-friendly SOMA (same shape as Phase 4 smoke).
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
        vocab_size=128,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=device)
    tokenizer = train_bpe_tokenizer(
        iter(
            [
                "hello world",
                "the quick brown fox",
                "lorem ipsum dolor sit amet",
            ]
        ),
        vocab_size=cfg.vocab_size,
    )
    encoder = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=model_name,
        llm_hidden_dim=chat_head.hidden_size,  # 960 for SmolLM2-360M
        num_prefix_tokens=4,  # small for speed
    )
    verbalizer = SomaVerbalizer(spec).to(device)

    session = ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
        system_prompt=None,
    )

    # Three distinct turns. ``min_new_tokens=1`` forces >=1 non-EOS token
    # per turn: with a near-null verbalizer prefix (untrained projector)
    # and no chat-template framing, SmolLM2-Instruct will sometimes emit
    # <|im_end|> at t=0 for certain prompts, which decodes to an empty
    # string under skip_special_tokens. The smoke verifies the pipeline
    # produces text every turn, not that the untrained model is coherent.
    gen_kwargs = dict(max_new_tokens=10, do_sample=False, min_new_tokens=1)
    r1 = session.respond(user_text="Hello!", **gen_kwargs)
    r2 = session.respond(user_text="What's your favorite color?", **gen_kwargs)
    r3 = session.respond(user_text="Tell me a joke.", **gen_kwargs)

    print(f"\nTurn 1: {r1!r}")
    print(f"Turn 2: {r2!r}")
    print(f"Turn 3: {r3!r}")

    # Each response must be a non-empty string.
    for idx, r in enumerate((r1, r2, r3), start=1):
        assert isinstance(r, str), f"turn {idx} response is not str: {type(r)}"
        assert len(r) > 0, f"turn {idx} response is empty"

    # History shape check.
    assert len(session.history) == 6
    assert [t.role for t in session.history] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    # Save round-trip: session.save writes brain + history.
    session.save(out_dir=tmp_path)
    assert (tmp_path / "chat_history.json").exists()
    assert (tmp_path / "brain.pt").exists()

    # Fresh session can load the history back.
    soma2 = SOMA(cfg, device=device)
    verbalizer2 = SomaVerbalizer(spec).to(device)
    encoder2 = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    session2 = ChatSession(
        soma=soma2,
        verbalizer=verbalizer2,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder2,
        system_prompt=None,
    )
    session2.load_history(out_dir=tmp_path)
    assert len(session2.history) == 6
    assert session2.history[0].text == "Hello!"
    assert session2.history[2].text == "What's your favorite color?"
    assert session2.history[4].text == "Tell me a joke."
