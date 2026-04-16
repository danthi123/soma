"""Tests for soma.memory.wal — append-only log with CRC-framed binary records.

The WAL pairs a JSONL metadata file with a binary embeddings file; each
binary frame is length-prefixed and CRC32-checksummed so we can detect
torn tails from crashes and truncate back to the last good record.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from soma.memory.wal import WAL, WalRecord


def _rec(op: str, nid: str, *, text: str | None = None, dim: int = 8, seed: int = 0) -> WalRecord:
    torch.manual_seed(seed)
    emb = torch.randn(dim) if op == "store" else None
    return WalRecord(
        op=op,  # type: ignore[arg-type]
        node_id=nid,
        text=text,
        metadata={"seed": seed} if op == "store" else {},
        timestamp_step=seed,
        embedding=emb,
        emb_offset=None,
    )


def test_header_written_on_open(tmp_path: Path) -> None:
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        ops = tmp_path / "memory_ops.wal.jsonl"
        assert ops.exists(), "WAL.open must create the ops jsonl"
        first_line = ops.read_text(encoding="utf-8").splitlines()[0]
        header = json.loads(first_line)
        assert header["schema"] == 1
        assert header["embed_dim"] == 8
        assert "created_at" in header
        assert "bundle_version" in header
    finally:
        wal.close()


def test_empty_wal_with_header_has_zero_records(tmp_path: Path) -> None:
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    wal.close()
    # Re-open and replay — header only, no records.
    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        records = list(wal2.replay())
        assert records == []
        assert wal2.record_count == 0
    finally:
        wal2.close()


def test_append_then_replay_roundtrip(tmp_path: Path) -> None:
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        originals = []
        for i in range(100):
            rec = _rec("store", f"node-{i}", text=f"text {i}", dim=8, seed=i)
            wal.append(rec)
            originals.append(rec)
        assert wal.record_count == 100
    finally:
        wal.close()

    # Fresh WAL on the same dir — replay must match.
    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        replayed = list(wal2.replay())
        assert len(replayed) == 100
        for orig, rep in zip(originals, replayed, strict=True):
            assert rep.op == orig.op
            assert rep.node_id == orig.node_id
            assert rep.text == orig.text
            assert rep.metadata == orig.metadata
            assert rep.timestamp_step == orig.timestamp_step
            assert orig.embedding is not None and rep.embedding is not None
            assert torch.allclose(rep.embedding, orig.embedding, atol=1e-6)
    finally:
        wal2.close()


def test_forget_record_has_no_embedding(tmp_path: Path) -> None:
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        wal.append(_rec("store", "a", text="alpha", dim=8, seed=1))
        wal.append(
            WalRecord(
                op="forget",
                node_id="a",
                text=None,
                metadata={},
                timestamp_step=2,
                embedding=None,
                emb_offset=None,
            )
        )
    finally:
        wal.close()

    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        records = list(wal2.replay())
        assert len(records) == 2
        assert records[0].op == "store"
        assert records[1].op == "forget"
        assert records[1].embedding is None
        assert records[1].text is None
    finally:
        wal2.close()


def test_torn_tail_line_is_truncated(tmp_path: Path) -> None:
    """Simulate a crash that left a half-line at the end of memory_ops.wal.jsonl.

    Replay must return the 10 complete records and the file must end at
    the offset of the last good newline so future appends don't produce
    a corrupt jsonl.
    """
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        for i in range(10):
            wal.append(_rec("store", f"node-{i}", text=f"t{i}", dim=8, seed=i))
    finally:
        wal.close()

    ops_path = tmp_path / "memory_ops.wal.jsonl"
    # Append a half line (no newline) simulating a torn write.
    with open(ops_path, "ab") as fh:
        fh.write(b'{"op":"store","node_id":"partial')
    torn_size = ops_path.stat().st_size

    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        records = list(wal2.replay())
        assert len(records) == 10, f"expected 10 good records, got {len(records)}"
        # The file must have been truncated back to the end of the last good line.
        clean_size = ops_path.stat().st_size
        assert clean_size < torn_size, "torn tail should have been truncated"
    finally:
        wal2.close()


def test_torn_binary_record_truncated(tmp_path: Path) -> None:
    """Half-written binary frame at the tail → replay stops at the last good
    record AND truncates the binary file back to the last good frame."""
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        for i in range(10):
            wal.append(_rec("store", f"n{i}", text=f"t{i}", dim=8, seed=i))
    finally:
        wal.close()

    emb_path = tmp_path / "memory_embeddings.wal.bin"
    # Append partial bytes — not enough for a full frame.
    with open(emb_path, "ab") as fh:
        fh.write(b"\x00" * 5)

    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        records = list(wal2.replay())
        assert len(records) == 10
    finally:
        wal2.close()


def test_crc_mismatch_triggers_truncate(tmp_path: Path) -> None:
    """Flip one byte in a middle record's payload: replay stops at the
    first bad CRC (does NOT skip to the next valid frame)."""
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        for i in range(10):
            wal.append(_rec("store", f"n{i}", text=f"t{i}", dim=8, seed=i))
    finally:
        wal.close()

    emb_path = tmp_path / "memory_embeddings.wal.bin"
    # Find the 5th record's offset: header is 8 bytes, payload is
    # 1 (dtype tag) + embed_dim*4 bytes. Frame size = 8 + 1 + 32 = 41.
    frame_size = 8 + 1 + 8 * 4
    data = emb_path.read_bytes()
    assert len(data) == 10 * frame_size
    # Corrupt the payload of record index 4 (5th record) — flip a byte.
    corrupt_offset = 4 * frame_size + 8 + 1  # into the embedding bytes
    buf = bytearray(data)
    buf[corrupt_offset] ^= 0xFF
    emb_path.write_bytes(bytes(buf))

    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        records = list(wal2.replay())
        # Must stop at the first bad CRC — records 0..3 only (4 good).
        assert len(records) == 4, f"expected 4 good records before CRC trip, got {len(records)}"
    finally:
        wal2.close()


def test_size_bytes_grows_with_appends(tmp_path: Path) -> None:
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        s0 = wal.size_bytes
        wal.append(_rec("store", "a", text="alpha", dim=8, seed=0))
        s1 = wal.size_bytes
        assert s1 > s0
        wal.append(_rec("store", "b", text="beta", dim=8, seed=1))
        s2 = wal.size_bytes
        assert s2 > s1
    finally:
        wal.close()


def test_truncate_resets_to_header_only(tmp_path: Path) -> None:
    """WAL.truncate() zeroes both files and rewrites the header."""
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        for i in range(5):
            wal.append(_rec("store", f"n{i}", text=f"t{i}", dim=8, seed=i))
        assert wal.record_count == 5
        wal.truncate()
        assert wal.record_count == 0
        # Header still present.
        ops = tmp_path / "memory_ops.wal.jsonl"
        lines = ops.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        header = json.loads(lines[0])
        assert header["schema"] == 1
        # Embeddings file is zero bytes.
        emb = tmp_path / "memory_embeddings.wal.bin"
        assert emb.stat().st_size == 0
        # And replay yields nothing.
        wal.close()
    finally:
        if wal._ops_fh is not None:  # type: ignore[attr-defined]
            wal.close()

    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        assert list(wal2.replay()) == []
    finally:
        wal2.close()


def test_binary_frame_format_matches_spec(tmp_path: Path) -> None:
    """Pin the binary frame layout so we don't silently break wire format.

    Frame = struct.pack("<II", length, crc32(payload)) + payload
    Payload = dtype_tag_byte (1) + embed_dim*4 bytes (float32).
    """
    wal = WAL(tmp_path, embed_dim=4)
    wal.open()
    try:
        wal.append(_rec("store", "x", text="t", dim=4, seed=0))
    finally:
        wal.close()

    emb = (tmp_path / "memory_embeddings.wal.bin").read_bytes()
    # Length + CRC header is 8 bytes.
    length, _crc = struct.unpack("<II", emb[:8])
    # Payload is dtype_tag + 4*4 bytes = 17.
    assert length == 1 + 4 * 4
    assert len(emb) == 8 + length, "binary frame must be 8-byte header + payload"


# ------------------------------------------------------------------
# update_metadata op type — Phase 2 (ConversationalMemory supersede)
# ------------------------------------------------------------------
def test_update_metadata_op_round_trips(tmp_path: Path) -> None:
    """An update_metadata record appends + replays with its patch intact."""
    wal = WAL(tmp_path, embed_dim=8)
    wal.open()
    try:
        wal.append(_rec("store", "n1", text="t1", dim=8, seed=1))
        wal.append(
            WalRecord(
                op="update_metadata",  # type: ignore[arg-type]
                node_id="n1",
                text=None,
                metadata={"patch": {"superseded_by": "n2", "reason": "moved"}},
                timestamp_step=5,
                embedding=None,
                emb_offset=None,
            )
        )
    finally:
        wal.close()

    wal2 = WAL(tmp_path, embed_dim=8)
    wal2.open()
    try:
        records = list(wal2.replay())
        assert len(records) == 2
        assert records[0].op == "store"
        assert records[1].op == "update_metadata"
        assert records[1].node_id == "n1"
        assert records[1].metadata == {
            "patch": {"superseded_by": "n2", "reason": "moved"}
        }
        assert records[1].embedding is None
    finally:
        wal2.close()


def test_update_metadata_op_consumes_no_binary_frame(tmp_path: Path) -> None:
    """Like forget, update_metadata doesn't append to the .bin file."""
    wal = WAL(tmp_path, embed_dim=4)
    wal.open()
    try:
        wal.append(_rec("store", "n1", text="t1", dim=4, seed=1))
        emb_size_after_store = (tmp_path / "memory_embeddings.wal.bin").stat().st_size
        wal.append(
            WalRecord(
                op="update_metadata",  # type: ignore[arg-type]
                node_id="n1",
                text=None,
                metadata={"patch": {"k": "v"}},
                timestamp_step=2,
                embedding=None,
                emb_offset=None,
            )
        )
        emb_size_after_update = (tmp_path / "memory_embeddings.wal.bin").stat().st_size
        assert emb_size_after_update == emb_size_after_store, (
            "update_metadata must not write to the binary embeddings file"
        )
    finally:
        wal.close()
