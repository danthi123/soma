"""Related-entry browser: walk a SOMA bundle graph-style via ``mem.related()``.

Vector DBs let you ask "give me top-k for this query." SOMA additionally
exposes the entry-to-entry edges of its store via ``mem.related(node_id)``,
so you can pick *one* memory and traverse what's near it — useful for
discovering tangential context an LLM might miss when it only sees the
query's top-k.

This demo gives you an interactive REPL:

  1. Seed the cursor by query (top-1 cosine match) or by id.
  2. Show the cursor entry + its k nearest neighbors.
  3. Pick a neighbor by number to move the cursor; pick 0 to stop.

Usage::

    python scripts/demo_related_browser.py --bundle my-brain/ \\
      --seed "where does the user live?"

    # Or by id (e.g., from `demo_memory_inspect.py dump | jq .id`):
    python scripts/demo_related_browser.py --bundle my-brain/ \\
      --seed-id 3a4b5c6d7e...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from soma.memory import MemoryLayer


def _truncate(s: str, n: int = 100) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_entry(
    label: str, node_id: str, text: str, meta: dict[str, object], score: float | None
) -> None:
    score_str = f"  score={score:.4f}" if score is not None else ""
    print(f"{label}{score_str}  id={node_id[:8]}…")
    print(f"  {_truncate(text, 200)}")
    if meta:
        meta_str = " ".join(f"{k}={v!r}" for k, v in meta.items())
        print(f"  meta: {meta_str}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--seed", help="Seed cursor by query (top-1 cosine)")
    p.add_argument("--seed-id", help="Seed cursor by node id (full or prefix)")
    p.add_argument("--k", type=int, default=5, help="Neighbors per step")
    args = p.parse_args()

    if not args.bundle.exists():
        p.error(f"bundle not found: {args.bundle}")
    if not (args.seed or args.seed_id):
        p.error("pass --seed <text> or --seed-id <id>")

    mem = MemoryLayer.load(args.bundle)
    print(f"Loaded {len(mem)} entries from {args.bundle}\n")

    # Resolve initial cursor.
    cursor_id: str
    if args.seed_id:
        match = [
            full for full in mem._ids
            if full == args.seed_id or full.startswith(args.seed_id)
        ]
        if not match:
            p.error(f"no id starts with {args.seed_id!r}")
        if len(match) > 1:
            p.error(f"prefix {args.seed_id!r} matches {len(match)} ids; be more specific")
        cursor_id = match[0]
    else:
        hits = mem.retrieve(args.seed, k=1)
        if not hits:
            print("no hits for seed query — empty bundle?")
            return
        cursor_id = hits[0].node_id

    while True:
        cursor = mem.get(cursor_id)
        if cursor is None:  # pragma: no cover — defensive
            print(f"cursor id {cursor_id} no longer exists; bailing.")
            return
        print("=" * 60)
        _print_entry(
            "CURSOR", cursor.node_id, cursor.text, cursor.metadata, score=None
        )
        try:
            neighbors = mem.related(cursor_id, k=args.k)
        except KeyError:
            print("(cursor has no neighbors — store has only one entry?)")
            return
        if not neighbors:
            print("\n(no neighbors)")
            return
        print(f"\nNearest {len(neighbors)} neighbors:")
        for i, h in enumerate(neighbors, 1):
            print()
            _print_entry(
                f"  [{i}]", h.node_id, h.text, h.metadata, score=h.score
            )

        print(f"\nPick a neighbor [1-{len(neighbors)}], or 0 to quit:")
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not raw:
            continue
        try:
            choice = int(raw)
        except ValueError:
            print("(enter a number)")
            continue
        if choice == 0:
            return
        if not 1 <= choice <= len(neighbors):
            print(f"(out of range; pick 1..{len(neighbors)})")
            continue
        cursor_id = neighbors[choice - 1].node_id


if __name__ == "__main__":
    main()
