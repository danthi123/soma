"""Import a Chroma collection into a SOMA MemoryLayer bundle.

    python scripts/migrate_chroma.py \\
        --chroma-path ./chroma-data \\
        --collection my_memories \\
        --out ./my-brain

Reads all documents + metadata from a Chroma persistent collection and
stores them in a MemoryLayer bundle. Uses sentence-transformers for
embedding (same as MemoryLayer.with_sbert).

Requires: chromadb, sentence-transformers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chroma-path", type=Path, required=True, help="Chroma persistent dir")
    p.add_argument("--collection", type=str, required=True, help="Collection name to import")
    p.add_argument("--out", type=Path, required=True, help="Output MemoryLayer bundle path")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="sbert model name")
    p.add_argument("--batch-size", type=int, default=100, help="Chroma query batch size")
    args = p.parse_args()

    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed. pip install chromadb")
        sys.exit(1)

    from soma.memory import MemoryLayer

    print(f"Opening Chroma at {args.chroma_path}, collection={args.collection!r}...")
    client = chromadb.PersistentClient(path=str(args.chroma_path))
    try:
        col = client.get_collection(args.collection)
    except Exception as exc:
        print(f"Error opening collection: {exc}")
        available = [c.name for c in client.list_collections()]
        print(f"Available collections: {available}")
        sys.exit(1)

    total = col.count()
    print(f"  {total} documents in collection.")

    if total == 0:
        print("Nothing to migrate.")
        sys.exit(0)

    print(f"Creating MemoryLayer with sbert model={args.model!r}...")
    mem = MemoryLayer.with_sbert(args.model)

    migrated = 0
    offset = 0
    while offset < total:
        batch = col.get(
            limit=args.batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        docs = batch.get("documents", []) or []
        metas = batch.get("metadatas", []) or []
        ids = batch.get("ids", []) or []

        if not docs:
            break

        for doc, meta, chroma_id in zip(docs, metas, ids, strict=False):
            if doc is None:
                continue
            entry_meta = dict(meta) if meta else {}
            entry_meta["chroma_id"] = chroma_id
            entry_meta["source"] = "chroma-migration"
            mem.store(doc, metadata=entry_meta)
            migrated += 1

        offset += len(docs)
        print(f"  migrated {migrated}/{total}...")

    mem.save(args.out)
    print(f"\nDone. {migrated} entries migrated to {args.out}")
    print(f"Load with: MemoryLayer.load('{args.out}', embed_fn=...)")


if __name__ == "__main__":
    main()
