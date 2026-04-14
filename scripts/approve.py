"""Operator approval for arch-carveout change specs.

``approve.py`` inspects the approval queue (``.soma-loop/state/approval_queue.jsonl``)
and transitions a single entry from ``pending`` -> ``approved`` (or ``stale`` if
the base commit has drifted). Already-terminal entries (approved/applied/
stale/rejected) are refused.

Usage::

    python scripts/approve.py --list
    python scripts/approve.py <queue-id>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_QUEUE = Path(".soma-loop/state/approval_queue.jsonl")
TERMINAL_STATES = {"approved", "rejected", "stale", "applied"}


# ---- Pure state transitions ------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _operator_name() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def mark_approved(
    entry: dict[str, Any], *, username: str | None = None, ts: str | None = None
) -> dict[str, Any]:
    updated = dict(entry)
    updated["queue_status"] = "approved"
    updated["approved_at"] = ts or _now_iso()
    updated["approved_by"] = username or _operator_name()
    return updated


def mark_stale(entry: dict[str, Any], *, reason: str, ts: str | None = None) -> dict[str, Any]:
    updated = dict(entry)
    updated["queue_status"] = "stale"
    updated["rejected_reason"] = reason
    updated["rejected_at"] = ts or _now_iso()
    return updated


def mark_rejected(entry: dict[str, Any], *, reason: str, ts: str | None = None) -> dict[str, Any]:
    updated = dict(entry)
    updated["queue_status"] = "rejected"
    updated["rejected_reason"] = reason
    updated["rejected_at"] = ts or _now_iso()
    return updated


# ---- I/O ------------------------------------------------------------------


def load_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def write_queue(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    os.replace(tmp, path)


# ---- Listing --------------------------------------------------------------


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def format_list_table(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "approval queue empty"

    header = (
        f"{'ID':<8} {'CLASS':<14} {'QUEUE_STATUS':<12} {'AGE':<20} {'BASE_SHA':<10} DESCRIPTION"
    )
    rows = [header, "-" * min(120, len(header))]
    for entry in entries:
        queue_id = str(entry.get("queue_id", ""))[:6]
        klass = str(entry.get("class") or "-")
        status = str(entry.get("queue_status", "-"))
        age = str(entry.get("queued_at", "-"))
        sha = str(entry.get("base_commit_sha", "-"))[:8]
        desc = _truncate(str(entry.get("description", "")).strip(), 60)
        rows.append(f"{queue_id:<8} {klass:<14} {status:<12} {age:<20} {sha:<10} {desc}")
    return "\n".join(rows)


# ---- Git HEAD -------------------------------------------------------------


def _git_head_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


# ---- CLI -----------------------------------------------------------------


def _find_entry(entries: list[dict[str, Any]], queue_id: str) -> tuple[int, dict[str, Any]] | None:
    for idx, entry in enumerate(entries):
        eid = str(entry.get("queue_id", ""))
        if eid == queue_id or eid.startswith(queue_id):
            return idx, entry
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Approve a queued SOMA change spec.")
    parser.add_argument("queue_id", nargs="?", help="Queue entry id (or prefix).")
    parser.add_argument("--list", action="store_true", help="List queue entries and exit.")
    parser.add_argument(
        "--queue", type=Path, default=DEFAULT_QUEUE, help="Path to approval_queue.jsonl."
    )
    args = parser.parse_args(argv)

    entries = load_queue(args.queue)

    if args.list:
        print(format_list_table(entries))
        return 0

    if not args.queue_id:
        parser.error("queue_id is required unless --list is given")

    found = _find_entry(entries, args.queue_id)
    if found is None:
        print(f"approve: no entry matches '{args.queue_id}'", file=sys.stderr)
        return 1

    idx, entry = found
    status = str(entry.get("queue_status", "pending"))
    if status in TERMINAL_STATES:
        print(
            f"approve: entry {entry.get('queue_id', '?')} already in terminal state "
            f"'{status}' — refusing to re-process",
            file=sys.stderr,
        )
        return 1

    head = _git_head_sha()
    base = str(entry.get("base_commit_sha", ""))
    if head is not None and base and base != head:
        entries[idx] = mark_stale(entry, reason=f"base_commit_sha drifted (HEAD={head[:8]})")
        write_queue(args.queue, entries)
        print(f"approve: {entry.get('queue_id')} -> stale (HEAD moved to {head[:8]})")
        return 0

    entries[idx] = mark_approved(entry)
    write_queue(args.queue, entries)
    print(f"approve: {entry.get('queue_id')} -> approved by {entries[idx]['approved_by']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
