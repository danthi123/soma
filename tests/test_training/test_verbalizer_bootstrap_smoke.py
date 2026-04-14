"""Slow-marked smoke test: bootstrap verbalizer against real SmolLM2.

Exercises the full Phase 4 pipeline:
    text -> SOMA -> state -> verbalizer -> soft-prompt prefix ->
    frozen SmolLM2-360M -> LM loss -> backprop into verbalizer

Asserts held-out loss drops pre -> post training on tinyshakespeare.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow


def _hub_is_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true")


def test_bootstrap_loss_drops_on_smolm2_and_tinyshakespeare(tmp_path: Path) -> None:
    """The load-bearing Phase 4 assertion.

    After N steps of bootstrap training on a slice of tinyshakespeare,
    held-out LM loss should drop relative to the pre-training baseline.
    """
    pytest.importorskip("transformers")
    from dataclasses import replace
    from typing import Any, cast

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soma.core.config import SOMAConfig
    from soma.io.chat_head import ChatHead
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
    from soma.system import SOMA
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device = torch.device("cpu")  # GPU reserved for training service

    # --- Load the frozen LLM -------------------------------------------
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

    # --- Load tinyshakespeare slice ------------------------------------
    corpus_path = Path("data/tinyshakespeare.txt")
    if not corpus_path.exists():
        pytest.skip("data/tinyshakespeare.txt missing")
    # Take a small slice so the smoke stays fast but meaningful.
    text = corpus_path.read_text(encoding="utf-8")[:4000]

    # Slice into windows of ~256 characters each.
    eval_texts = [text[i : i + 256] for i in range(0, 1000, 256)]  # ~4 samples
    train_texts = [text[i : i + 256] for i in range(1000, 5000, 200)]  # ~20 samples

    # --- Tiny CPU-friendly SOMA ----------------------------------------
    # Use an elevated verbalizer_lr -- the default 1e-4 targets long training
    # runs; this smoke runs only 20 steps, so we need faster convergence.
    cfg = replace(
        SOMAConfig(
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
        ),
        verbalizer_lr=1e-3,
    )
    soma = SOMA(cfg, device=device)
    tokenizer = train_bpe_tokenizer(iter(train_texts), vocab_size=cfg.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )

    # --- Verbalizer sized to the real SmolLM2 d_model (960) ------------
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=model_name,
        llm_hidden_dim=chat_head.hidden_size,  # 960 for SmolLM2-360M
        num_prefix_tokens=4,  # small for speed
    )
    verbalizer = SomaVerbalizer(spec).to(device)

    trainer = VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=tokenizer,
        encoder=encoder,
    )

    # --- Baseline eval --------------------------------------------------
    pre_loss = trainer.eval_lm_loss(texts=eval_texts)
    print(f"\npre-training held-out loss: {pre_loss:.4f}")

    # --- Train ----------------------------------------------------------
    losses = trainer.train(
        corpus=iter(train_texts),
        max_steps=20,
        out_dir=tmp_path,
    )
    print(f"train losses: first={losses[0]:.4f}, last={losses[-1]:.4f}")

    # --- Post-training eval --------------------------------------------
    post_loss = trainer.eval_lm_loss(texts=eval_texts)
    print(f"post-training held-out loss: {post_loss:.4f}")

    # --- Load-bearing assertion ----------------------------------------
    assert post_loss < pre_loss, (
        f"Held-out loss did not drop -- pre={pre_loss:.4f} post={post_loss:.4f}. "
        "This breaks the core Phase 4 value prop (verbalizer learns something)."
    )

    # Final checkpoint must exist.
    assert (tmp_path / "verbalizer_final").exists()
    assert (tmp_path / "verbalizer_final" / "spec.json").exists()
    assert (tmp_path / "verbalizer_final" / "weights.pt").exists()
