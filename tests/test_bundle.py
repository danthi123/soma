"""Tests for :mod:`soma.bundle` — bundle introspection helpers.

These helpers are stdlib-only and must NOT depend on MemoryLayer —
``soma bundle list`` should be cheap to run across a directory with
hundreds of bundles, so we can't afford a torch import per dir.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from soma.bundle import BundleInfo, is_bundle_dir, list_bundles, load_info


def _write_index(
    bundle: Path,
    *,
    entries: int,
    embed_dim: int,
    schema_version: int = 2,
) -> None:
    """Write a minimal ``memory_index.json`` that load_info can parse."""
    bundle.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": schema_version,
        "embed_dim": embed_dim,
        "embed_type": "text_encoder",
        "step": entries,
        "entries": [
            {
                "node_id": f"n{i:06d}",
                "text": f"entry {i}",
                "metadata": {},
                "timestamp_step": i,
            }
            for i in range(entries)
        ],
    }
    (bundle / "memory_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    # memory_embeddings.pt is a binary blob; content doesn't matter for
    # introspection, only its presence + mtime.
    (bundle / "memory_embeddings.pt").write_bytes(b"\x00" * 32)


def _write_backend_json(bundle: Path, backend_name: str) -> None:
    """Drop a ``backend.json`` next to the snapshot (lancedb/qdrant path)."""
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "backend.json").write_text(
        json.dumps({"backend": backend_name, "dim": 8}),
        encoding="utf-8",
    )


def _write_wal(bundle: Path, ops_bytes: int = 512, emb_bytes: int = 1024) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "memory_ops.wal.jsonl").write_bytes(b"{}\n" * (ops_bytes // 3))
    (bundle / "memory_embeddings.wal.bin").write_bytes(b"\x00" * emb_bytes)


# ---------------------------------------------------------------------------
# is_bundle_dir
# ---------------------------------------------------------------------------
def test_is_bundle_dir_true_for_snapshot_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "alex"
    _write_index(bundle, entries=3, embed_dim=8)
    assert is_bundle_dir(bundle) is True


def test_is_bundle_dir_true_when_only_backend_json(tmp_path: Path) -> None:
    """A qdrant-style bundle with only ``backend.json`` still counts —
    the snapshot lives in the remote, not on disk."""
    bundle = tmp_path / "bobbi"
    bundle.mkdir()
    _write_backend_json(bundle, "qdrant")
    assert is_bundle_dir(bundle) is True


def test_is_bundle_dir_true_when_only_wal(tmp_path: Path) -> None:
    """New bundle that has never been snapshotted but has WAL records."""
    bundle = tmp_path / "cam"
    _write_wal(bundle)
    assert is_bundle_dir(bundle) is True


def test_is_bundle_dir_false_for_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_bundle_dir(empty) is False


def test_is_bundle_dir_false_for_non_dir(tmp_path: Path) -> None:
    f = tmp_path / "not-a-dir.txt"
    f.write_text("hi", encoding="utf-8")
    assert is_bundle_dir(f) is False


def test_is_bundle_dir_false_for_missing_path(tmp_path: Path) -> None:
    assert is_bundle_dir(tmp_path / "does-not-exist") is False


# ---------------------------------------------------------------------------
# load_info
# ---------------------------------------------------------------------------
def test_load_info_populates_every_field_for_healthy_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "alex"
    _write_index(bundle, entries=5, embed_dim=384)
    _write_wal(bundle, ops_bytes=300, emb_bytes=200)

    info = load_info(bundle)

    assert isinstance(info, BundleInfo)
    assert info.path == bundle
    assert info.entries == 5
    assert info.embed_dim == 384
    # Default backend for MemoryLayer snapshots is inproc-flat.
    assert info.backend == "inproc-flat"
    assert isinstance(info.last_modified, datetime)
    assert info.wal_bytes > 0
    assert info.corrupt is False
    assert info.corrupt_reason == ""


def test_load_info_prefers_backend_json_when_present(tmp_path: Path) -> None:
    bundle = tmp_path / "qd"
    _write_index(bundle, entries=2, embed_dim=8)
    _write_backend_json(bundle, "qdrant")
    info = load_info(bundle)
    assert info.backend == "qdrant"


def test_load_info_corrupt_when_malformed_index(tmp_path: Path) -> None:
    bundle = tmp_path / "bad"
    bundle.mkdir()
    (bundle / "memory_index.json").write_text("{not-json", encoding="utf-8")
    info = load_info(bundle)
    assert info.corrupt is True
    assert info.corrupt_reason  # non-empty
    # Reason is truncated at 200 chars.
    assert len(info.corrupt_reason) <= 200


def test_load_info_corrupt_reason_truncated_at_200_chars(tmp_path: Path) -> None:
    bundle = tmp_path / "ugly"
    bundle.mkdir()
    # Corrupt JSON with a long context that would otherwise produce a
    # verbose error message.
    (bundle / "memory_index.json").write_text(
        "{" + "x" * 1000, encoding="utf-8"
    )
    info = load_info(bundle)
    assert info.corrupt is True
    assert len(info.corrupt_reason) <= 200


def test_load_info_corrupt_when_missing_embed_dim(tmp_path: Path) -> None:
    bundle = tmp_path / "no-dim"
    bundle.mkdir()
    (bundle / "memory_index.json").write_text(
        json.dumps({"schema_version": 2, "entries": []}),
        encoding="utf-8",
    )
    info = load_info(bundle)
    assert info.corrupt is True
    assert "embed_dim" in info.corrupt_reason


def test_load_info_wal_bytes_zero_when_no_wal(tmp_path: Path) -> None:
    bundle = tmp_path / "clean"
    _write_index(bundle, entries=1, embed_dim=8)
    info = load_info(bundle)
    assert info.wal_bytes == 0


def test_load_info_missing_bundle_is_corrupt(tmp_path: Path) -> None:
    """load_info on a non-bundle dir returns a corrupt BundleInfo —
    callers (e.g. ``soma bundle list``) render it as CORRUPT rather
    than crash."""
    empty = tmp_path / "empty"
    empty.mkdir()
    info = load_info(empty)
    assert info.corrupt is True
    assert info.corrupt_reason


# ---------------------------------------------------------------------------
# list_bundles
# ---------------------------------------------------------------------------
def test_list_bundles_finds_direct_children(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_index(a, entries=1, embed_dim=8)
    _write_index(b, entries=2, embed_dim=8)

    found = list_bundles(tmp_path)
    paths = {info.path for info in found}
    assert a in paths
    assert b in paths


def test_list_bundles_returns_sorted_desc_by_last_modified(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_index(old, entries=1, embed_dim=8)
    # Touch the old bundle's index file to an older timestamp.
    past = time.time() - 3600
    os.utime(old / "memory_index.json", (past, past))
    _write_index(new, entries=2, embed_dim=8)

    found = list_bundles(tmp_path)
    assert len(found) >= 2
    # new should come first (most recent).
    first_path = found[0].path
    second_path = found[1].path
    assert first_path == new
    assert second_path == old


def test_list_bundles_includes_corrupt_entries(tmp_path: Path) -> None:
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _write_index(good, entries=1, embed_dim=8)
    bad.mkdir()
    (bad / "memory_index.json").write_text("{not-json", encoding="utf-8")
    found = list_bundles(tmp_path)
    by_path = {info.path: info for info in found}
    assert good in by_path
    assert bad in by_path
    assert by_path[bad].corrupt is True
    assert by_path[good].corrupt is False


def test_list_bundles_descends_up_to_depth_3(tmp_path: Path) -> None:
    """Nested bundles at depth <= 3 are found; depth 4+ are ignored."""
    # root / l1 / l2 / l3 / bundle-at-depth-4  (should be skipped)
    deep = tmp_path / "l1" / "l2" / "l3" / "l4-bundle"
    _write_index(deep, entries=1, embed_dim=8)
    # root / l1 / l2 / bundle-at-depth-3  (should be found)
    ok = tmp_path / "l1" / "l2" / "ok-bundle"
    _write_index(ok, entries=1, embed_dim=8)

    found = list_bundles(tmp_path)
    paths = {info.path for info in found}
    assert ok in paths
    assert deep not in paths


def test_list_bundles_does_not_recurse_into_bundle_children(tmp_path: Path) -> None:
    """Once a dir is identified as a bundle, we stop descending — its
    internal ``wal/`` or ``lancedb/`` subdirs are not themselves bundles."""
    b = tmp_path / "outer"
    _write_index(b, entries=1, embed_dim=8)
    # Fake inner "bundle" that should NOT appear as a separate entry.
    inner = b / "lancedb"
    inner.mkdir()
    _write_backend_json(inner, "lancedb")

    found = list_bundles(tmp_path)
    paths = {info.path for info in found}
    assert b in paths
    assert inner not in paths


def test_list_bundles_empty_for_empty_root(tmp_path: Path) -> None:
    assert list_bundles(tmp_path) == []


def test_list_bundles_ignores_hidden_dirs(tmp_path: Path) -> None:
    """Skip dot-dirs (``.git``, ``.cache``) — they're never bundles and
    descending into them wastes time on big repos."""
    hidden = tmp_path / ".hidden" / "maybe-bundle"
    _write_index(hidden, entries=1, embed_dim=8)
    assert list_bundles(tmp_path) == []
