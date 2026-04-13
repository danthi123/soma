"""Operator rejection for queued arch-carveout change specs.

Marks a pending queue entry as ``rejected`` with an optional reason.
Already-terminal entries (approved/applied/stale/rejected) are refused.

Usage::

    python scripts/reject.py <queue-id> [--reason "..."]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.approve import (
    DEFAULT_QUEUE,
    TERMINAL_STATES,
    _find_entry,
    load_queue,
    mark_rejected,
    write_queue,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject a queued SOMA change spec.")
    parser.add_argument("queue_id", help="Queue entry id (or prefix).")
    parser.add_argument(
        "--reason", default="rejected by operator", help="Rejection reason."
    )
    parser.add_argument(
        "--queue", type=Path, default=DEFAULT_QUEUE, help="Path to approval_queue.jsonl."
    )
    args = parser.parse_args(argv)

    entries = load_queue(args.queue)
    found = _find_entry(entries, args.queue_id)
    if found is None:
        print(f"reject: no entry matches '{args.queue_id}'", file=sys.stderr)
        return 1

    idx, entry = found
    status = str(entry.get("queue_status", "pending"))
    if status in TERMINAL_STATES:
        print(
            f"reject: entry {entry.get('queue_id', '?')} already in terminal state "
            f"'{status}' — refusing to re-process",
            file=sys.stderr,
        )
        return 1

    entries[idx] = mark_rejected(entry, reason=args.reason)
    write_queue(args.queue, entries)
    print(f"reject: {entry.get('queue_id')} -> rejected ({args.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
