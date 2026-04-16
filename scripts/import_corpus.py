"""Generic corpus importer — load JSONL / JSON / CSV into a SOMA bundle.

One script for the "I have data in format X, get it into SOMA" use
case. Handles arbitrary JSONL / JSON-array / CSV inputs with
configurable field mapping, and ships presets for the common exports
from other memory systems (Mem0, Letta, Zep, raw chat history).

Usage examples::

    # Raw JSONL with custom field mapping:
    python scripts/import_corpus.py jsonl chat.jsonl \\
      --bundle my-brain/ --text-field message --metadata-field user_id

    # Mem0 export (preset):
    python scripts/import_corpus.py mem0 mem0-export.json --bundle my-brain/

    # Letta archival memory dump (preset):
    python scripts/import_corpus.py letta letta-archival.jsonl --bundle my-brain/

    # Zep message export (preset):
    python scripts/import_corpus.py zep zep-session.json --bundle my-brain/

    # CSV with header row:
    python scripts/import_corpus.py csv notes.csv --bundle my-brain/ \\
      --text-field content --metadata-field tag --metadata-field created

The importer never modifies the source file. If the bundle already
exists, new entries are *appended* (SOMA preserves insertion order).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Record:
    text: str
    metadata: dict[str, Any]


# ------------------------------------------------------------------
# Generic readers
# ------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        # Common wrapper shapes: {"items": [...]}, {"data": [...]}, {"messages": [...]}
        for key in ("items", "data", "messages", "memories", "records"):
            if key in data and isinstance(data[key], list):
                yield from data[key]
                return
        # Fallback: treat the dict itself as one record.
        yield data
    else:
        raise ValueError(f"unsupported JSON root type: {type(data).__name__}")


def _iter_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


# ------------------------------------------------------------------
# Field mapping
# ------------------------------------------------------------------


def _map_record(
    row: dict[str, Any],
    *,
    text_field: str,
    metadata_fields: list[str],
) -> Record | None:
    text = row.get(text_field)
    if not isinstance(text, str) or not text.strip():
        return None
    if metadata_fields:
        meta = {k: row[k] for k in metadata_fields if k in row}
    else:
        meta = {k: v for k, v in row.items() if k != text_field}
    return Record(text=text.strip(), metadata=meta)


# ------------------------------------------------------------------
# Presets for well-known exports
# ------------------------------------------------------------------


def _mem0_to_records(rows: Iterator[dict[str, Any]]) -> Iterator[Record]:
    """Mem0 export: dicts with 'memory' / 'text' plus optional
    'user_id', 'categories', 'created_at'."""
    for r in rows:
        text = r.get("memory") or r.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        meta = {
            k: r[k]
            for k in ("user_id", "categories", "created_at", "id", "score")
            if k in r
        }
        meta.setdefault("source", "mem0")
        yield Record(text=text.strip(), metadata=meta)


def _letta_to_records(rows: Iterator[dict[str, Any]]) -> Iterator[Record]:
    """Letta archival-memory dump: dicts with 'text' + 'metadata' +
    'created_at'. Older dumps wrap under 'archival_memory' — caller
    should unwrap upstream."""
    for r in rows:
        text = r.get("text") or r.get("content") or r.get("value")
        if not isinstance(text, str) or not text.strip():
            continue
        meta_in = r.get("metadata") or {}
        meta = dict(meta_in) if isinstance(meta_in, dict) else {}
        for k in ("created_at", "tags", "source", "id"):
            if k in r and k not in meta:
                meta[k] = r[k]
        meta.setdefault("source", "letta")
        yield Record(text=text.strip(), metadata=meta)


def _zep_to_records(rows: Iterator[dict[str, Any]]) -> Iterator[Record]:
    """Zep session export: 'role' + 'content' + optional
    'created_at' / 'metadata' per message."""
    for r in rows:
        content = r.get("content") or r.get("text")
        if not isinstance(content, str) or not content.strip():
            continue
        meta: dict[str, Any] = {}
        if "role" in r:
            meta["role"] = r["role"]
        if "created_at" in r:
            meta["created_at"] = r["created_at"]
        extra_meta = r.get("metadata")
        if isinstance(extra_meta, dict):
            meta.update(extra_meta)
        meta.setdefault("source", "zep")
        yield Record(text=content.strip(), metadata=meta)


# ------------------------------------------------------------------
# Core ingest loop
# ------------------------------------------------------------------


def ingest(records: Iterator[Record], bundle: Path) -> int:
    """Stream records into a MemoryLayer at ``bundle``. Appends if the
    bundle exists. Returns the count stored."""
    from soma.memory import MemoryLayer

    if bundle.exists() and (bundle / "memory_index.json").exists():
        mem = MemoryLayer.load(bundle)
        print(f"  appending to existing bundle ({len(mem)} entries)")
    else:
        mem = MemoryLayer.with_sbert()
    n = 0
    for rec in records:
        mem.store(rec.text, metadata=rec.metadata)
        n += 1
        if n % 500 == 0:
            print(f"  ... {n} stored")
    mem.save(bundle)
    print(f"  saved bundle to {bundle} ({len(mem)} total entries, +{n} new)")
    return n


def _pick_reader(fmt: str, path: Path) -> Iterator[dict[str, Any]]:
    if fmt == "jsonl":
        return _iter_jsonl(path)
    if fmt == "json":
        return _iter_json(path)
    if fmt == "csv":
        return _iter_csv(path)
    raise ValueError(f"unknown format {fmt!r}")


def _pick_records(
    args: argparse.Namespace,
) -> Iterator[Record]:
    cmd = args.cmd
    src: Path = args.source
    if cmd == "mem0":
        rows = _iter_jsonl(src) if src.suffix == ".jsonl" else _iter_json(src)
        return _mem0_to_records(rows)
    if cmd == "letta":
        rows = _iter_jsonl(src) if src.suffix == ".jsonl" else _iter_json(src)
        return _letta_to_records(rows)
    if cmd == "zep":
        rows = _iter_jsonl(src) if src.suffix == ".jsonl" else _iter_json(src)
        return _zep_to_records(rows)
    if cmd in {"jsonl", "json", "csv"}:
        rows = _pick_reader(cmd, src)

        def _gen() -> Iterator[Record]:
            for row in rows:
                rec = _map_record(
                    row,
                    text_field=args.text_field,
                    metadata_fields=args.metadata_field or [],
                )
                if rec is not None:
                    yield rec

        return _gen()
    raise ValueError(f"unknown cmd {cmd!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("source", type=Path, help="Source file")
    common.add_argument("--bundle", type=Path, required=True, help="Target bundle dir")

    mapped = argparse.ArgumentParser(add_help=False)
    mapped.add_argument(
        "--text-field", default="text",
        help="Field name containing the text body (default: text)",
    )
    mapped.add_argument(
        "--metadata-field", action="append", default=[],
        help="Field name(s) to carry as metadata (repeat for multiple). "
        "If omitted, all non-text fields become metadata.",
    )

    sub.add_parser("jsonl", parents=[common, mapped], help="Generic JSONL")
    sub.add_parser("json", parents=[common, mapped], help="Generic JSON (array or wrapper)")
    sub.add_parser("csv", parents=[common, mapped], help="CSV with header row")

    sub.add_parser("mem0", parents=[common], help="Mem0 export preset")
    sub.add_parser("letta", parents=[common], help="Letta archival-memory preset")
    sub.add_parser("zep", parents=[common], help="Zep session export preset")

    args = p.parse_args()

    if not args.source.exists():
        p.error(f"source file not found: {args.source}")
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    print(f"=== Importing {args.source} ({args.cmd}) -> {args.bundle} ===\n")
    ingest(_pick_records(args), args.bundle)


if __name__ == "__main__":
    main()
