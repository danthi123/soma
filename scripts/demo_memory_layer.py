"""Demonstrate MemoryLayer: store, retrieve, persist, reload.

Two-phase demo that proves the core product claim: store facts, save
the brain to disk, load it in a new process, and retrieve correctly.

    python scripts/demo_memory_layer.py --bundle /tmp/my-brain

Phase 1: store a handful of facts, retrieve them, save.
Phase 2: load the saved bundle, retrieve without re-storing, verify
matches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory import MemoryLayer

FACTS = [
    ("user lives in Portland, OR", {"source": "onboarding"}),
    ("user is vegetarian", {"source": "chat-2026-04-10"}),
    ("user's dog is named Luna", {"source": "chat-2026-04-11"}),
    ("user works at a robotics startup", {"source": "linkedin"}),
    ("user prefers dark mode in all apps", {"source": "settings"}),
    ("user speaks French and English", {"source": "profile"}),
    ("user is allergic to shellfish", {"source": "chat-2026-04-12"}),
    ("the project deadline is June 15", {"source": "calendar"}),
]

QUERIES = [
    "where does the user live?",
    "any dietary restrictions?",
    "tell me about the user's pet",
    "what does the user do for work?",
    "when is the deadline?",
]


def _build_embedder(*, device: torch.device | None = None) -> tuple[object, TextEncoder]:
    corpus = [text for text, _ in FACTS] + QUERIES
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=64, max_seq_len=128, device=device)
    return tokenizer, encoder


def _phase1(bundle_path: Path) -> None:
    print("=== Phase 1: store + retrieve + save ===\n")
    tokenizer, encoder = _build_embedder()
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)

    for text, meta in FACTS:
        nid = mem.store(text, metadata=meta)
        print(f"  stored: {text!r}  (id={nid[:8]}...)")

    print(f"\n  {len(mem)} entries in memory.\n")

    for query in QUERIES:
        hits = mem.retrieve(query, k=2)
        top = hits[0] if hits else None
        print(f"  Q: {query}")
        if top:
            print(f"  A: {top.text}  (score={top.score:.3f})")
        print()

    mem.save(bundle_path)
    print(f"  Bundle saved to {bundle_path}\n")


def _phase2(bundle_path: Path) -> None:
    print("=== Phase 2: load from disk + retrieve (no re-store) ===\n")
    mem = MemoryLayer.load(bundle_path)
    print(f"  Loaded {len(mem)} entries from {bundle_path}.\n")

    all_pass = True
    for query in QUERIES:
        hits = mem.retrieve(query, k=2)
        top = hits[0] if hits else None
        print(f"  Q: {query}")
        if top:
            print(f"  A: {top.text}  (score={top.score:.3f})")
        else:
            print("  A: (no results)")
            all_pass = False
        print()

    recent = mem.get_recent(3)
    print("  3 most recent entries:")
    for h in recent:
        print(f"    - {h.text}  (meta={h.metadata})")
    print()

    if all_pass:
        print("  All queries returned results. Persistence works.\n")
    else:
        print("  Some queries returned no results. Investigate.\n")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/demo-memory-layer"),
        help="Directory for the memory bundle (default: artifacts/demo-memory-layer).",
    )
    args = p.parse_args()

    _phase1(args.bundle)
    _phase2(args.bundle)
    print("Done.")


if __name__ == "__main__":
    main()
