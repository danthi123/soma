"""End-to-end SOMA chat demo.

Auto-detects the best tier for the available hardware and runs a short
4-turn demo conversation. No CLI args — this is the zero-config "hello
world" of SOMA chat: a new operator clones the repo, runs one command,
and sees the full SOMA + verbalizer + frozen LLM pipeline exchange four
turns end-to-end.

    python scripts/demo_chat.py

The first run downloads the tier-selected HF model (e.g. SmolLM2-360M
at ~720MB in fp16 for the CPU / low-VRAM "tiny" tier). Subsequent runs
read from the HF cache and start in a few seconds.

Because the SomaVerbalizer is freshly constructed with near-zero-init
weights (no bootstrap training inside the demo), responses are
grammatically-shaped but semantically empty. The point is to prove the
pipeline works, not that the untrained projector has anything to say.

GGUF opt-in
-----------
Setting ``SOMA_GGUF_PATH=/path/to/model.gguf`` switches the demo to the
inference-only GGUF backend (via llama-cpp-python). Verbalizer
construction is SKIPPED in that mode — user text is fed directly to the
llama.cpp runtime as a string prompt. The SOMA graph still runs for
state evolution across turns. Keeps demo_chat zero-arg; no CLI surface
change for operators who never set the env var.
"""

from __future__ import annotations

import os
from pathlib import Path

from soma.core.config import SOMAConfig
from soma.deploy.chat_head_factory import build_chat_head
from soma.deploy.cli import dtype_label
from soma.deploy.devices import (
    MODEL_TIERS,
    auto_select_tier,
    detect_cuda_vram,
    effective_vram_gb,
    select_device_and_dtype,
)
from soma.deploy.gguf_backend import build_gguf_chat_head
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession
from soma.system import SOMA

SEED_TEXTS: list[str] = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "knowledge is power",
    "time flies like an arrow",
    "machine learning is a subset of artificial intelligence",
]

DEMO_QUESTIONS: list[str] = [
    "Hello!",
    "What is a brain?",
    "Tell me a fun fact.",
    "What should I learn about next?",
]


def _build_demo_cfg() -> SOMAConfig:
    """Demo-shaped SOMA config. Small CPU-friendly dims."""
    return SOMAConfig(
        sensor_output_dim=16,
        associator_input_dim=16,
        associator_hidden_dim=32,
        associator_output_dim=16,
        integrator_input_dim=16,
        integrator_hidden_dim=32,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=4,
        wm_dim=16,
        episodic_capacity=16,
        key_dim=16,
        value_dim=16,
        vocab_size=256,
        text_embed_dim=16,
        max_nodes=64,
        initial_associator_count=4,
        initial_integrator_count=2,
        max_input_tokens=32,
        max_output_tokens=8,
        seed=0,
    )


def _read_gguf_env() -> Path | None:
    """Return the GGUF path from ``SOMA_GGUF_PATH`` if it points at a real file.

    Missing env var -> ``None`` (HF flow). Set-but-nonexistent -> raise
    so a typo surfaces immediately instead of silently falling back to
    the HF download path and redoing a several-GB pull.
    """
    raw = os.environ.get("SOMA_GGUF_PATH")
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(
            f"SOMA_GGUF_PATH points to a non-existent file: {path}. "
            "Unset the env var to fall back to the HF download path."
        )
    return path


def _build_session_hf(cfg: SOMAConfig) -> ChatSession:
    """Build the HF-backed demo session (downloads a tier model on first run)."""
    # 1. Detect hardware + pick a tier. Apply the headroom factor so an
    # RTX 3090 (raw 23 GB floor) gets 20 effective GB and lands in the
    # 'large' tier instead of bouncing on the 23/24 boundary.
    vram = detect_cuda_vram()
    effective = effective_vram_gb(vram_gb=vram, safety_factor=cfg.vram_safety_factor)
    tier = auto_select_tier(vram_gb=effective)
    device, dtype = select_device_and_dtype()
    spec = MODEL_TIERS[tier]

    # 2. Print the selection — the whole "is the auto-detect working" signal.
    if vram is None:
        print("Detected VRAM: none (no CUDA) -- running on CPU")
    else:
        print(
            f"Detected VRAM: {vram}GB raw -> {effective}GB effective "
            f"(safety_factor={cfg.vram_safety_factor})"
        )
    print(f"Selected tier: {tier}  ({spec['name']}, ~{spec['approx_fp16_gb']:.1f}GB)")
    print(f"Device: {device}  dtype: {dtype_label(dtype)}")
    print()

    # 3. Build the frozen LLM-backed ChatHead.
    chat_head = build_chat_head(tier=tier, device=device, dtype=dtype)

    # 4. SOMA + tokenizer + encoder + verbalizer (see individual comments).
    soma = SOMA(cfg, device=device)
    tokenizer = train_bpe_tokenizer(iter(SEED_TEXTS), vocab_size=cfg.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
        device=device,
    )
    verbalizer_spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=spec["name"],
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=8,
    )
    verbalizer = SomaVerbalizer(verbalizer_spec).to(device)

    return ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
    )


def _build_session_gguf(cfg: SOMAConfig, gguf_path: Path) -> ChatSession:
    """Build a GGUF-backed demo session. Skips verbalizer construction.

    Keeps the SOMA graph + tokenizer + encoder identical to the HF path
    so state evolution behaves the same across turns. Only the LLM
    surface swaps.
    """
    # GGUF manages its own device placement (n_gpu_layers=-1 offloads
    # everything). SOMA still wants a torch device — honour the same
    # auto-detection the HF path uses.
    device, _dtype = select_device_and_dtype()
    print(f"Using GGUF backend: {gguf_path}")
    print(f"Device (SOMA): {device}")
    print("Verbalizer: skipped (GGUF mode feeds user_text directly to llama.cpp)")
    print()

    gguf_head = build_gguf_chat_head(gguf_path=gguf_path)

    soma = SOMA(cfg, device=device)
    tokenizer = train_bpe_tokenizer(iter(SEED_TEXTS), vocab_size=cfg.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
        device=device,
    )

    return ChatSession(
        soma=soma,
        verbalizer=None,
        chat_head=None,
        gguf_head=gguf_head,
        tokenizer=tokenizer,
        encoder=encoder,
    )


def main() -> None:
    cfg = _build_demo_cfg()

    # GGUF opt-in via SOMA_GGUF_PATH env var. Missing env var -> HF flow.
    gguf_path = _read_gguf_env()
    if gguf_path is not None:
        session = _build_session_gguf(cfg, gguf_path)
    else:
        session = _build_session_hf(cfg)

    # Four-turn demo. ``min_new_tokens=3`` avoids the untrained-projector
    # edge case where SmolLM2-Instruct emits <|im_end|> at t=0 (decodes to
    # an empty string under skip_special_tokens). GGUFChatHead silently
    # ignores ``min_new_tokens`` / ``do_sample`` kwargs via **_ignored.
    for q in DEMO_QUESTIONS:
        response = session.respond(
            user_text=q,
            max_new_tokens=30,
            min_new_tokens=3,
            do_sample=False,
        )
        print(f"[user] {q}")
        print(f"[soma] {response}")
        print()


if __name__ == "__main__":
    main()
