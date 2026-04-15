"""Slow-marked smoke: online verbalizer training across a 10-turn chat.

Exercises the full Phase 6 pipeline:
    user text -> SOMA state -> verbalizer prefix -> SmolLM2 response ->
    response back into SOMA -> record (user, response) in replay ->
    sample batch -> per-sample teacher-forced LM loss -> Adam step ->
    divergence check

Run manually:
    pytest tests/test_training/test_online_verbalizer_smoke.py -v -m slow
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow


def _hub_is_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true")


def test_ten_turn_online_training_with_smolm2(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    import math
    from typing import Any, cast

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soma.core.config import SOMAConfig
    from soma.io.chat_head import ChatHead
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
    from soma.session.chat_session import ChatSession
    from soma.system import SOMA
    from soma.training.online_verbalizer import OnlineVerbalizerTrainer
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

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

    # Tiny CPU-friendly SOMA (same shape as Phase 5 T9).
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
                "time flies like an arrow",
                "knowledge is power",
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
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=4,
    )
    verbalizer = SomaVerbalizer(spec).to(device)

    # Wire up the online trainer.
    inner = VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=tokenizer,
        encoder=encoder,
    )
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)

    session = ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
        online_trainer=online,
    )

    # Ten distinct user turns -- mix topics to avoid trivial repetition.
    user_turns = [
        "Hello there!",
        "What's the weather like today?",
        "Tell me about Paris.",
        "What's 2 + 2?",
        "Describe a cat.",
        "Do you like music?",
        "What's your favorite book?",
        "Tell me a short story.",
        "Give me a recipe idea.",
        "What's the capital of Japan?",
    ]

    responses: list[str] = []
    for i, text in enumerate(user_turns):
        resp = session.respond(
            user_text=text,
            max_new_tokens=10,
            min_new_tokens=1,  # Phase 5 T9 workaround for near-null EOS
            do_sample=False,
        )
        responses.append(resp)
        print(f"turn {i:02d}: {text!r} -> {resp!r}")

    print(f"\nloss history: {list(online._loss_history)}")
    print(f"is_diverged: {online.is_diverged}")
    print(f"replay buffer size: {len(online.replay_buffer)}")

    # --- Load-bearing assertions ---
    # 1. All 10 responses are non-empty strings.
    for i, resp in enumerate(responses):
        assert isinstance(resp, str), f"turn {i} response is not str: {type(resp)}"
        assert len(resp) > 0, f"turn {i} response is empty"

    # 2. Replay buffer recorded all 10 turns.
    assert len(online.replay_buffer) == 10, (
        f"expected 10 entries in replay buffer, got {len(online.replay_buffer)}"
    )

    # 3. Loss history recorded 10 entries (one per turn, finite).
    assert len(online._loss_history) == 10, (
        f"expected 10 loss entries, got {len(online._loss_history)}"
    )
    for i, loss in enumerate(online._loss_history):
        assert math.isfinite(loss), f"loss at turn {i} is non-finite: {loss}"

    # 4. Training has not diverged -- the canonical "online training works"
    #    signal. If this flips True, the divergence monitor did its job
    #    but we have a real numerical stability problem to investigate.
    assert not online.is_diverged, (
        f"online training diverged. loss_history={list(online._loss_history)}"
    )

    # 5. History shape is correct -- 2 entries per turn = 20 total.
    assert len(session.history) == 20
