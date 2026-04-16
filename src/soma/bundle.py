"""Bundle introspection helpers for the ``soma bundle`` CLI group.

The functions here are deliberately stdlib-only. ``soma bundle list``
needs to scan a root directory with potentially hundreds of bundles,
and a torch/numpy import per bundle would make the command unusably
slow on cold caches.

A **SOMA bundle** is a directory containing one or more of:

- ``memory_index.json`` — the canonical MemoryLayer snapshot manifest.
  Always present after a successful :meth:`MemoryLayer.save`. Holds
  ``embed_dim``, ``embed_type``, and an ``entries[]`` list.
- ``backend.json`` — optional sidecar written by non-inproc backends
  (``lancedb``, ``qdrant``). When present, its ``"backend"`` field is
  the authoritative backend name; otherwise the bundle is assumed to
  be ``inproc-flat``.
- ``memory_ops.wal.jsonl`` + ``memory_embeddings.wal.bin`` — WAL
  sidecars written when durability is enabled. Their combined size
  in bytes is exposed as :attr:`BundleInfo.wal_bytes`.

Any other directory (empty, or containing unrelated files) is NOT a
bundle. The ``delete`` subcommand refuses non-bundle paths as a
safety net against ``soma bundle delete ~``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Canonical manifest file — always present after MemoryLayer.save().
_INDEX_FILENAME = "memory_index.json"
# Sidecar written by lancedb/qdrant backends.
_BACKEND_FILENAME = "backend.json"
# WAL sidecars (both files exist together when durability is on).
_WAL_OPS_FILENAME = "memory_ops.wal.jsonl"
_WAL_EMB_FILENAME = "memory_embeddings.wal.bin"

# Hard cap on the free-form corrupt reason string so a gigantic JSON
# error never blows out the list-table column widths.
_CORRUPT_REASON_MAX_CHARS = 200

# Guardrail for list_bundles(root) — descend this many levels below
# ``root``. Keeps accidental ``soma bundle list /`` from walking the
# whole filesystem.
_LIST_MAX_DEPTH = 3


@dataclass(frozen=True)
class BundleInfo:
    """Summary of a single bundle directory for CLI rendering.

    The fields are populated from on-disk metadata only — no backend
    instantiation, no tokenizer reload. When the required files are
    missing or malformed, :attr:`corrupt` is True and
    :attr:`corrupt_reason` carries a short explanation (capped at 200
    chars); the remaining fields hold best-effort defaults (0 / empty
    string / epoch time) so callers can still render the row.
    """

    path: Path
    entries: int
    embed_dim: int
    backend: str
    last_modified: datetime
    wal_bytes: int
    corrupt: bool = False
    corrupt_reason: str = ""
    # Optional snapshot timestamp parsed from ``backend.json.snapshot_ts``
    # when present. Not every backend emits it; kept here so ``info``
    # can surface it without a second disk hit.
    snapshot_ts: str | None = field(default=None)


def _is_bundle_marker_present(p: Path) -> bool:
    """True iff ``p`` contains at least one file that identifies a bundle.

    Accepts any of: ``memory_index.json`` (snapshot), ``backend.json``
    (remote backend sidecar), or WAL ops file (fresh bundle that never
    snapshotted). An empty dir does not qualify.
    """
    return (
        (p / _INDEX_FILENAME).is_file()
        or (p / _BACKEND_FILENAME).is_file()
        or (p / _WAL_OPS_FILENAME).is_file()
    )


def is_bundle_dir(p: Path) -> bool:
    """Return ``True`` if ``p`` is a directory that looks like a SOMA bundle.

    Safe on non-existent paths, symlinks, and regular files — returns
    ``False`` rather than raising. The check is marker-file based
    (never descends) so it's O(1) disk reads per call.
    """
    try:
        if not p.is_dir():
            return False
    except OSError:
        return False
    return _is_bundle_marker_present(p)


def _truncate_reason(msg: str) -> str:
    """Clamp the corrupt reason to 200 chars, single-line, safe for tables."""
    collapsed = " ".join(msg.split())
    if len(collapsed) <= _CORRUPT_REASON_MAX_CHARS:
        return collapsed
    return collapsed[: _CORRUPT_REASON_MAX_CHARS - 3] + "..."


def _wal_bytes(p: Path) -> int:
    """Sum ``memory_ops.wal.jsonl`` + ``memory_embeddings.wal.bin`` sizes."""
    total = 0
    for name in (_WAL_OPS_FILENAME, _WAL_EMB_FILENAME):
        fp = p / name
        try:
            if fp.is_file():
                total += fp.stat().st_size
        except OSError:
            pass
    return total


def _last_modified(p: Path) -> datetime:
    """Return the mtime of the newest bundle-relevant file in ``p``.

    Falls back to the directory mtime if no relevant file is present
    or readable. Used for table sorting — not authoritative.
    """
    candidates = (
        _INDEX_FILENAME,
        _BACKEND_FILENAME,
        _WAL_OPS_FILENAME,
        _WAL_EMB_FILENAME,
        "memory_embeddings.pt",
    )
    best: float | None = None
    for name in candidates:
        fp = p / name
        try:
            if fp.is_file():
                mt = fp.stat().st_mtime
                if best is None or mt > best:
                    best = mt
        except OSError:
            continue
    if best is None:
        try:
            best = p.stat().st_mtime
        except OSError:
            best = 0.0
    return datetime.fromtimestamp(best)


def _resolve_backend_name(p: Path) -> tuple[str, str | None]:
    """Return ``(backend_name, snapshot_ts)``.

    Reads ``backend.json`` if it exists — its ``"backend"`` field wins.
    Otherwise returns the default ``"inproc-flat"`` for MemoryLayer
    snapshots. Parsing errors here do NOT flag the bundle corrupt;
    they just fall back to the default.
    """
    bp = p / _BACKEND_FILENAME
    if not bp.is_file():
        return ("inproc-flat", None)
    try:
        data = json.loads(bp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ("inproc-flat", None)
    name = str(data.get("backend") or "inproc-flat")
    snap = data.get("snapshot_ts")
    snap_str = str(snap) if snap is not None else None
    return (name, snap_str)


def load_info(p: Path) -> BundleInfo:
    """Build a :class:`BundleInfo` for ``p``.

    On any error (missing files, malformed JSON, schema mismatch) the
    returned object has ``corrupt=True`` with a short ``corrupt_reason``
    and best-effort zeroes/empties elsewhere. This shape lets
    ``soma bundle list`` render a mixed table without try/except on
    every row.
    """
    path = Path(p)
    last_mod = _last_modified(path)
    wal = _wal_bytes(path)
    backend_name, snapshot_ts = _resolve_backend_name(path)

    index_path = path / _INDEX_FILENAME
    if not index_path.is_file():
        # No snapshot. If we have a backend.json OR a WAL, report what we
        # can and don't flag corrupt unless even the dir is missing.
        if (path / _BACKEND_FILENAME).is_file() or (path / _WAL_OPS_FILENAME).is_file():
            return BundleInfo(
                path=path,
                entries=0,
                embed_dim=0,
                backend=backend_name,
                last_modified=last_mod,
                wal_bytes=wal,
                corrupt=False,
                corrupt_reason="",
                snapshot_ts=snapshot_ts,
            )
        return BundleInfo(
            path=path,
            entries=0,
            embed_dim=0,
            backend=backend_name,
            last_modified=last_mod,
            wal_bytes=wal,
            corrupt=True,
            corrupt_reason=_truncate_reason(
                f"not a bundle: missing {_INDEX_FILENAME}"
            ),
            snapshot_ts=snapshot_ts,
        )

    try:
        raw = index_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{_INDEX_FILENAME} is not a JSON object")
        if "embed_dim" not in data:
            raise ValueError(f"{_INDEX_FILENAME} missing required field 'embed_dim'")
        embed_dim = int(data["embed_dim"])
        entries = len(data.get("entries") or [])
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return BundleInfo(
            path=path,
            entries=0,
            embed_dim=0,
            backend=backend_name,
            last_modified=last_mod,
            wal_bytes=wal,
            corrupt=True,
            corrupt_reason=_truncate_reason(f"{type(exc).__name__}: {exc}"),
            snapshot_ts=snapshot_ts,
        )

    return BundleInfo(
        path=path,
        entries=entries,
        embed_dim=embed_dim,
        backend=backend_name,
        last_modified=last_mod,
        wal_bytes=wal,
        corrupt=False,
        corrupt_reason="",
        snapshot_ts=snapshot_ts,
    )


def list_bundles(root: Path) -> list[BundleInfo]:
    """Walk ``root`` up to depth 3 and return one :class:`BundleInfo` per bundle.

    Depth-limited to prevent accidental traversal of ``/`` or a user's
    full home dir. Hidden directories (``.git``, ``.cache``, etc.) are
    skipped entirely. Once a directory is identified as a bundle we
    stop descending — inner ``lancedb/`` or other state subdirs are
    NOT surfaced as separate bundles.

    Results are sorted by :attr:`BundleInfo.last_modified` descending
    (most-recent first), which is what an operator eyeballing a table
    generally wants.
    """
    root = Path(root)
    results: list[BundleInfo] = []
    if not root.is_dir():
        return results

    # Walk iteratively so we can prune: don't descend into bundles or
    # hidden dirs, cap depth at _LIST_MAX_DEPTH.
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        # Consider ``current`` itself only when it's below root; we
        # always check children below. First: is THIS a bundle?
        if current != root and is_bundle_dir(current):
            results.append(load_info(current))
            continue  # do not descend

        if depth >= _LIST_MAX_DEPTH:
            continue
        try:
            children = list(os.scandir(current))
        except OSError:
            continue
        for entry in children:
            if not entry.is_dir(follow_symlinks=False):
                continue
            name = entry.name
            if name.startswith("."):
                continue
            stack.append((Path(entry.path), depth + 1))

    results.sort(key=lambda info: info.last_modified, reverse=True)
    return results
