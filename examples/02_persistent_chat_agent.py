"""Chat agent with SOMA as persistent memory across process restarts.

The core pattern: retrieve relevant past conversation, pass it to the
LLM as context, store the new exchange. Close out → save. Next launch
→ load. The "brain" survives.

This example ships with a stub LLM so it runs with no external
dependencies beyond sbert. Swap in your real LLM of choice (see
``_llm_reply`` for the hook) — the memory contract doesn't change.

Requires::

    pip install -e ".[sbert]"

Run (first time)::

    python examples/02_persistent_chat_agent.py --user alex

Send a few messages, then exit (Ctrl-D / empty line). Re-run the same
command — the agent remembers your earlier exchanges.

Resets the brain::

    python examples/02_persistent_chat_agent.py --user alex --reset
"""

from __future__ import annotations

import argparse
import atexit
import sys
from pathlib import Path

import torch

from soma.memory import MemoryLayer


def _make_sbert_embed_fn(model_name: str = "all-MiniLM-L6-v2") -> tuple[callable, int]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    dim = int(model.get_sentence_embedding_dimension())

    def embed(text: str) -> torch.Tensor:
        return torch.tensor(model.encode(text, convert_to_numpy=True))

    return embed, dim


def _llm_reply(prompt: str, context: list[str]) -> str:
    """Stub LLM — replace with your real backend.

    Production wiring options:
      - Ollama:  ``ollama.chat(model="qwen3.5:4b", messages=[...])``
      - OpenAI:  ``openai.chat.completions.create(...)``
      - Claude:  ``benchmarks.industry.llm_backends.claude_runner_client``
      - LM Studio: HTTP to ``http://localhost:1234/v1/chat/completions``

    The stub is intentionally dumb so the example is self-contained.
    It echoes back what the agent "remembered" about the user.
    """
    if not context:
        return f"(no prior context) Noted: {prompt!r}"
    recalled = "; ".join(f'"{c}"' for c in context[:3])
    return f"Based on what I recall about you ({recalled}), I'd say: {prompt!r} is noted."


def _load_or_create_memory(
    bundle_path: Path,
    embed_fn: callable,
    embed_dim: int,
    *,
    reset: bool = False,
) -> MemoryLayer:
    """Load the brain if it exists, else start fresh."""
    if reset and bundle_path.exists():
        import shutil
        shutil.rmtree(bundle_path)
        print(f"(reset) removed {bundle_path}")

    if bundle_path.exists() and (bundle_path / "memory_index.json").exists():
        mem = MemoryLayer.load(bundle_path, embed_fn=embed_fn)
        print(f"(loaded {len(mem)} memories from {bundle_path})")
        return mem

    print(f"(new brain — will save to {bundle_path} on exit)")
    return MemoryLayer(embed_fn=embed_fn, embed_dim=embed_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA chat agent demo")
    parser.add_argument("--user", default="alex",
                        help="User identifier (namespaces memory by metadata)")
    parser.add_argument("--bundle", default="./data/chat-agent",
                        help="Directory for the persistent brain")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the brain before starting")
    parser.add_argument("--context-k", type=int, default=3,
                        help="How many past memories to retrieve per turn")
    args = parser.parse_args()

    bundle_path = Path(args.bundle).expanduser().resolve()

    embed_fn, embed_dim = _make_sbert_embed_fn()
    mem = _load_or_create_memory(bundle_path, embed_fn, embed_dim, reset=args.reset)

    # Save on exit so the brain persists. In a real service you'd
    # checkpoint on a schedule AND on shutdown — this atexit hook
    # handles the graceful-shutdown case.
    def _save_on_exit() -> None:
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        mem.save(bundle_path)
        print(f"\n(saved {len(mem)} memories to {bundle_path})")
    atexit.register(_save_on_exit)

    # --- Chat loop ---
    print("\nChat started. Type messages; empty line or Ctrl-D to exit.")
    print(f"User: {args.user}  |  Brain: {bundle_path}  |  Context size: {args.context_k}")
    print("-" * 60)

    try:
        while True:
            try:
                prompt = input(f"{args.user}> ").strip()
            except EOFError:
                break
            if not prompt:
                break

            # 1. Retrieve relevant past memories scoped to this user.
            past_hits = mem.retrieve(
                prompt, k=args.context_k,
                where={"user": args.user, "role": "user"},
            )
            context = [hit.text for hit in past_hits]

            # 2. Ask the LLM with memory as context.
            reply = _llm_reply(prompt, context)
            print(f"agent> {reply}")

            # 3. Store both sides of the exchange for next turn / next
            #    session. Metadata lets us filter by user and by role
            #    at retrieve time.
            mem.store(prompt, metadata={"user": args.user, "role": "user"})
            mem.store(reply, metadata={"user": args.user, "role": "assistant"})
    except KeyboardInterrupt:
        print("\n(interrupted)")

    # atexit handler saves the brain; no explicit cleanup needed.


if __name__ == "__main__":
    main()
