"""Memory inspector: browse, search, and stat a SOMA MemoryLayer bundle.

A read-only CLI for any saved MemoryLayer bundle — useful when you've
ingested a wiki / chat history / corpus and want to sanity-check what
SOMA actually stored, search it without spinning up an LLM, or pull
metadata for downstream tooling.

Subcommands::

    # Quick stats: entry count, embed dim, on-disk size, metadata
    # field histogram.
    python scripts/demo_memory_inspect.py stats --bundle my-brain/

    # List N most recent entries (default 20).
    python scripts/demo_memory_inspect.py recent --bundle my-brain/ --n 50

    # Filter by metadata key=value (multiple --where flags = AND).
    python scripts/demo_memory_inspect.py filter --bundle my-brain/ \\
      --where source=chat-2026-04-15 --where role=assistant

    # Vector search without an LLM — top-k by cosine, with metadata.
    python scripts/demo_memory_inspect.py search --bundle my-brain/ \\
      --query "where does the user live?" --k 5

    # Dump the whole store to JSONL on stdout (no embeddings — text +
    # metadata only). Useful for piping into jq / external indexers.
    python scripts/demo_memory_inspect.py dump --bundle my-brain/ > store.jsonl

The inspector never modifies the bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from soma.memory import MemoryLayer


def _bundle_disk_kb(bundle: Path) -> float:
    if not bundle.exists():
        return 0.0
    total = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
    return total / 1024.0


def _truncate(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def cmd_stats(mem: MemoryLayer, bundle: Path) -> None:
    print(f"Bundle: {bundle}")
    print(f"  entries: {len(mem)}")
    print(f"  embed_dim: {mem._embed_dim}")
    print(f"  disk: {_bundle_disk_kb(bundle):.1f} KB")
    if not len(mem):
        return
    keys: Counter[str] = Counter()
    for meta in mem._metadatas:
        for k in meta:
            keys[k] += 1
    if keys:
        print("  metadata keys (count):")
        for k, n in keys.most_common():
            print(f"    {k}: {n}")
    else:
        print("  metadata: (none)")


def cmd_recent(mem: MemoryLayer, n: int) -> None:
    if not len(mem):
        print("(empty bundle)")
        return
    # Most recent = highest indices.
    start = max(0, len(mem) - n)
    for i in range(len(mem) - 1, start - 1, -1):
        meta = mem._metadatas[i]
        meta_str = " ".join(f"{k}={v!r}" for k, v in meta.items()) if meta else "—"
        print(f"#{i:6d}  {_truncate(mem._texts[i])}")
        print(f"        meta: {meta_str}")


def _meta_matches(meta: dict[str, object], where: list[tuple[str, str]]) -> bool:
    for k, v in where:
        actual = meta.get(k)
        if actual is None or str(actual) != v:
            return False
    return True


def cmd_filter(mem: MemoryLayer, where: list[tuple[str, str]], limit: int) -> None:
    if not where:
        print("error: filter needs at least one --where key=value")
        sys.exit(2)
    matches = 0
    for i, meta in enumerate(mem._metadatas):
        if not _meta_matches(meta, where):
            continue
        meta_str = " ".join(f"{k}={v!r}" for k, v in meta.items()) if meta else "—"
        print(f"#{i:6d}  {_truncate(mem._texts[i])}")
        print(f"        meta: {meta_str}")
        matches += 1
        if matches >= limit:
            break
    if matches == 0:
        print("(no matches)")
    elif matches >= limit:
        print(f"(stopped at limit={limit}; pass --limit higher for more)")


def cmd_search(mem: MemoryLayer, query: str, k: int) -> None:
    hits = mem.retrieve(query, k=k)
    if not hits:
        print("(no hits)")
        return
    for i, h in enumerate(hits, 1):
        meta_str = (
            " ".join(f"{k}={v!r}" for k, v in h.metadata.items())
            if h.metadata
            else "—"
        )
        print(f"[{i}] score={h.score:.4f}  {_truncate(h.text, 100)}")
        print(f"      meta: {meta_str}")


def cmd_dump(mem: MemoryLayer) -> None:
    """Stream JSONL to stdout. One line per entry. No embeddings."""
    for i in range(len(mem)):
        rec = {
            "id": mem._ids[i],
            "text": mem._texts[i],
            "metadata": mem._metadatas[i],
            "timestamp_step": mem._timestamps[i],
        }
        sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_where(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--where expects key=value, got: {item!r}"
            )
        k, v = item.split("=", 1)
        out.append((k, v))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--bundle", type=Path, required=True, help="Path to MemoryLayer bundle"
    )

    sub.add_parser("stats", parents=[common], help="Bundle stats")

    p_recent = sub.add_parser("recent", parents=[common], help="Most recent entries")
    p_recent.add_argument("--n", type=int, default=20)

    p_filter = sub.add_parser(
        "filter", parents=[common], help="Filter by metadata key=value"
    )
    p_filter.add_argument(
        "--where",
        action="append",
        default=[],
        help="key=value (repeat for AND)",
    )
    p_filter.add_argument("--limit", type=int, default=50)

    p_search = sub.add_parser(
        "search", parents=[common], help="Vector search without LLM"
    )
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--k", type=int, default=5)

    sub.add_parser("dump", parents=[common], help="Stream JSONL to stdout")

    args = p.parse_args()

    if not args.bundle.exists():
        p.error(f"bundle not found: {args.bundle}")

    mem = MemoryLayer.load(args.bundle)

    if args.cmd == "stats":
        cmd_stats(mem, args.bundle)
    elif args.cmd == "recent":
        cmd_recent(mem, args.n)
    elif args.cmd == "filter":
        cmd_filter(mem, _parse_where(args.where), args.limit)
    elif args.cmd == "search":
        cmd_search(mem, args.query, args.k)
    elif args.cmd == "dump":
        cmd_dump(mem)
    else:  # pragma: no cover
        p.error(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
