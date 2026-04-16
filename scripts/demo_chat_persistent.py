"""Persistent-chat demo: MemoryLayer + frozen LLM.

Two-phase demo that proves the product story: chat → save the brain →
reload → the model "remembers" what you told it.

Phase 1 ("seed"): store a few personal facts into MemoryLayer, have a
short LLM-assisted chat that references them, save the memory bundle.

Phase 2 ("recall"): load the bundle in a fresh MemoryLayer, ask the
LLM a recall question whose answer lives only in memory, verify the
response cites the remembered fact.

    # Full LLM demo (uses deploy tier=auto):
    python scripts/demo_chat_persistent.py --bundle /tmp/my-brain

    # Dry-run (skips LLM, tests memory pipeline only):
    python scripts/demo_chat_persistent.py --bundle /tmp/my-brain --dry-run

The ``--dry-run`` flag replaces the LLM with a stub that echoes the
prompt's context section — useful for CI or machines without a GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory import MemoryLayer

SEED_FACTS = [
    "the user's name is Alex",
    "Alex lives in Portland, Oregon",
    "Alex is vegetarian and allergic to shellfish",
    "Alex's dog is named Luna",
    "Alex works at a robotics startup called ArcMotion",
]

PHASE1_PROMPTS = [
    "Tell me a bit about yourself.",
    "What should I cook for dinner tonight?",
]

RECALL_PROMPTS = [
    "What's my name and where do I live?",
    "I need a dinner recipe — what should I avoid?",
    "What's my dog's name?",
]


def _build_embedder(corpus: list[str]) -> tuple[Any, TextEncoder]:
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=512)
    encoder = TextEncoder(tokenizer, embed_dim=64, max_seq_len=128)
    return tokenizer, encoder


def _format_prompt(user_text: str, context_hits: list[Any]) -> str:
    if context_hits:
        ctx = "\n".join(f"- {h.text}" for h in context_hits)
        return (
            f"You are a helpful assistant. Use the following context about "
            f"the user to personalize your response:\n{ctx}\n\nUser: {user_text}\nAssistant:"
        )
    return f"User: {user_text}\nAssistant:"


def _generate(
    prompt: str,
    *,
    chat_head: Any | None,
    max_new_tokens: int = 80,
    dry_run: bool = False,
) -> str:
    if dry_run:
        lines = [
            ln for ln in prompt.splitlines() if ln.strip().startswith("- ")
        ]
        if lines:
            ctx = "; ".join(ln.strip("- ").strip() for ln in lines)
            return f"[dry-run] Context I would use: {ctx}"
        return "[dry-run] No context available."
    hf_tokenizer = chat_head.tokenizer
    inputs = hf_tokenizer(prompt, return_tensors="pt")
    device = chat_head.model.get_input_embeddings().weight.device
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        out = chat_head.model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    new_ids = out[0][input_ids.shape[1] :]
    return hf_tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _phase1(bundle_path: Path, *, chat_head: Any | None, dry_run: bool) -> None:
    print("=== Phase 1: seed facts + short chat + save ===\n")
    all_texts = SEED_FACTS + PHASE1_PROMPTS + RECALL_PROMPTS
    tokenizer, encoder = _build_embedder(all_texts)
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)

    for fact in SEED_FACTS:
        mem.store(fact, metadata={"source": "seed"})
        print(f"  [stored] {fact}")
    print()

    for prompt_text in PHASE1_PROMPTS:
        hits = mem.retrieve(prompt_text, k=3)
        full_prompt = _format_prompt(prompt_text, hits)
        reply = _generate(full_prompt, chat_head=chat_head, dry_run=dry_run)
        mem.store(prompt_text, metadata={"role": "user"})
        mem.store(reply, metadata={"role": "assistant"})
        print(f"  User: {prompt_text}")
        print(f"  Assistant: {reply}\n")

    mem.save(bundle_path)
    print(f"  Saved {len(mem)} entries to {bundle_path}\n")


def _phase2(bundle_path: Path, *, chat_head: Any | None, dry_run: bool) -> None:
    print("=== Phase 2: reload + recall (no re-seeding) ===\n")
    mem = MemoryLayer.load(bundle_path)
    print(f"  Loaded {len(mem)} entries.\n")

    for prompt_text in RECALL_PROMPTS:
        hits = mem.retrieve(prompt_text, k=3)
        full_prompt = _format_prompt(prompt_text, hits)
        reply = _generate(full_prompt, chat_head=chat_head, dry_run=dry_run)
        print(f"  User: {prompt_text}")
        if hits:
            top = hits[0]
            print(f"  Top memory hit: {top.text}  (score={top.score:.3f})")
        print(f"  Assistant: {reply}\n")

    print("Done. The model used persisted memory to answer recall questions.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/demo-persistent-chat"),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM loading; echo retrieved context instead.",
    )
    p.add_argument(
        "--tier",
        type=str,
        default="auto",
        help="Deploy tier for the LLM (default: auto). Ignored with --dry-run.",
    )
    args = p.parse_args()

    chat_head = None
    if not args.dry_run:
        from soma.deploy.chat_head_factory import build_chat_head
        from soma.deploy.cli import resolve_device_dtype_tier

        ns = argparse.Namespace(
            tier=args.tier,
            llm_name=None,
            device=None,
            dtype=None,
            quantization="none",
        )
        device, dtype, llm_name, tier = resolve_device_dtype_tier(ns)
        chat_head = build_chat_head(tier=tier, device=device, dtype=dtype)
        print(f"  LLM loaded: tier={tier}, device={device}, dtype={dtype}\n")

    _phase1(args.bundle, chat_head=chat_head, dry_run=args.dry_run)
    _phase2(args.bundle, chat_head=chat_head, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
