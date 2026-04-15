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
"""

from __future__ import annotations

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


def main() -> None:
    # 0. Build the SOMA config first -- the demo SOMA uses small CPU-friendly
    # dims, but the vram_safety_factor is independent of those, so it's the
    # one knob worth surfacing up front.
    cfg = SOMAConfig(
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

    # 4. SOMA construction (cfg built above so vram_safety_factor was
    # available for the auto-detect step).
    soma = SOMA(cfg, device=device)

    # 5. Train a tiny BPE tokenizer on the bundled seed corpus.
    tokenizer = train_bpe_tokenizer(iter(SEED_TEXTS), vocab_size=cfg.vocab_size)

    # 6. Text encoder matching the tokenizer + config dims. ``device`` keeps
    # embedding tables on the same device as SOMA so per-token feeds avoid
    # CPU<->GPU roundtrips.
    encoder = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
        device=device,
    )

    # 7. Fresh verbalizer — near-zero init, no bootstrap in the demo.
    verbalizer_spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=spec["name"],
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=8,
    )
    verbalizer = SomaVerbalizer(verbalizer_spec).to(device)

    # 8. Wire everything into a ChatSession.
    session = ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
    )

    # 9. Four-turn demo. ``min_new_tokens=3`` avoids the untrained-projector
    # edge case where SmolLM2-Instruct emits <|im_end|> at t=0 (decodes to
    # an empty string under skip_special_tokens).
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
