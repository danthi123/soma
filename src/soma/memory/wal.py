"""Write-ahead log for :class:`soma.memory.api.MemoryLayer`.

Paired files in a bundle dir:

- ``memory_ops.wal.jsonl`` — one record per line. First line is a header
  ``{"schema": 1, "embed_dim": N, "created_at": ts, "bundle_version": V}``.
  Subsequent lines are op records:

      {"op":"store","node_id":"x","text":"...","metadata":{},
       "emb_offset":N,"timestamp_step":N}
      {"op":"forget","node_id":"x","timestamp_step":N}
      {"op":"update_metadata","node_id":"x","patch":{...},"timestamp_step":N}

- ``memory_embeddings.wal.bin`` — contiguous binary frames, one per
  ``store``. Each frame is::

      struct.pack("<II", length, crc32(payload)) + payload
      payload = dtype_tag_byte + embed_dim * 4 bytes   # float32 today

Crash recovery (called on every :meth:`WAL.open`):

- Walk the jsonl looking for the last complete line. A torn tail (no
  trailing newline) is truncated away before any append touches the file.
- Walk the binary file frame by frame. A too-short tail or a CRC
  mismatch triggers truncation back to the last known-good frame. Replay
  STOPS at the first bad frame — we do not try to skip ahead and keep
  "later" records, because later records' ``emb_offset`` values point
  at byte positions that would now be wrong.

Durability modes (wired by :class:`MemoryLayer`; this module exposes
``flush()`` and treats append as a plain write):

- ``"sync"`` — caller fsyncs after every append.
- ``"batch"`` — caller fsyncs every N appends.
- ``"async"`` — caller only fsyncs on close or explicit ``flush()``.

Limitations (documented, enforced by tests):

- Only float32 embeddings are produced today; ``dtype_tag_byte`` is a
  forward-compat stub tagged as ``0x01``.
- Local filesystems only — NFS/SMB semantics are out of scope and
  callers get a documented caveat elsewhere.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO

import torch

OPS_FILENAME = "memory_ops.wal.jsonl"
EMB_FILENAME = "memory_embeddings.wal.bin"

# Format: length (u32, little-endian) + crc32 (u32, little-endian) + payload.
_FRAME_HEADER = struct.Struct("<II")
_DTYPE_FLOAT32 = 0x01

# Schema/bundle versions emitted in the WAL header. ``schema`` is the WAL
# record format itself; ``bundle_version`` records the MemoryLayer schema
# so the parent bundle can migrate without wiping the log.
WAL_SCHEMA = 1
WAL_BUNDLE_VERSION = 2


@dataclass(frozen=True)
class WalRecord:
    """One append-only WAL record.

    ``text``/``embedding`` are ``None`` for ``forget`` and
    ``update_metadata``.
    ``emb_offset`` is the byte offset of the record's frame in the
    embeddings .bin file. It is None on freshly-constructed records
    (replay fills it in; append writes it).

    For ``update_metadata``, the ``metadata`` field carries the patch
    dict under a ``"patch"`` key (e.g. ``{"patch": {"superseded_by": "x"}}``).
    Replay merges the patch into the target entry's metadata.
    """

    op: Literal["store", "forget", "update_metadata"]
    node_id: str
    text: str | None
    metadata: dict[str, Any]
    timestamp_step: int
    embedding: torch.Tensor | None
    emb_offset: int | None = None


def _pack_embedding(embedding: torch.Tensor) -> bytes:
    """Serialize a 1-D float tensor to ``dtype_tag + float32 bytes``."""
    if embedding.dim() != 1:
        raise ValueError(
            f"WAL embeddings must be 1-D, got shape {tuple(embedding.shape)}"
        )
    flat = embedding.detach().cpu().to(torch.float32).contiguous()
    return bytes([_DTYPE_FLOAT32]) + flat.numpy().tobytes()


def _unpack_embedding(payload: bytes, embed_dim: int) -> torch.Tensor:
    if not payload:
        raise ValueError("empty embedding payload")
    tag = payload[0]
    if tag != _DTYPE_FLOAT32:
        raise ValueError(f"unsupported embedding dtype tag 0x{tag:02x}")
    expected = 1 + embed_dim * 4
    if len(payload) != expected:
        raise ValueError(
            f"embedding payload length {len(payload)} != expected {expected}"
        )
    buf = payload[1:]
    arr = torch.frombuffer(bytearray(buf), dtype=torch.float32).clone()
    return arr.view(embed_dim)


class WAL:
    """Paired JSONL-metadata + CRC-framed-binary-embeddings WAL.

    See module docstring for the on-disk layout and recovery rules.
    """

    def __init__(
        self,
        bundle_dir: Path,
        embed_dim: int,
        durability: Literal["sync", "batch", "async"] = "sync",
        batch_size: int = 32,
    ) -> None:
        self._dir = Path(bundle_dir)
        self._embed_dim = int(embed_dim)
        self._durability = durability
        self._batch_size = int(batch_size)
        self._ops_path = self._dir / OPS_FILENAME
        self._emb_path = self._dir / EMB_FILENAME
        self._ops_fh: TextIO | None = None
        self._emb_fh: BinaryIO | None = None
        self._record_count: int = 0
        self._emb_size: int = 0  # mirror of _emb_fh.tell()
        self._batch_since_flush: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Open both files for append; do torn-tail recovery if needed.

        Creates a fresh header if this is a brand-new WAL. Walks the
        jsonl + binary files and truncates any torn tail so subsequent
        appends produce a parseable log.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        existed = self._ops_path.exists()
        if not existed:
            # Write the header before opening for append so we know the
            # file has a valid first line.
            self._write_header()
            # Embeddings file starts empty.
            self._emb_path.touch()

        # Recovery pass — truncate torn tails on both files.
        self._recover_and_count()

        # Open both files for append (text for jsonl, binary for .bin).
        # These are long-lived instance state, not scoped context managers.
        self._ops_fh = open(  # noqa: SIM115 — long-lived append handle
            self._ops_path, "a", encoding="utf-8", newline="\n"
        )
        self._emb_fh = open(self._emb_path, "ab")  # noqa: SIM115 — long-lived
        # For the binary file, tell() after opening in 'ab' may be 0 on
        # some platforms; record size from the filesystem.
        self._emb_size = self._emb_path.stat().st_size

    def close(self) -> None:
        if self._ops_fh is not None:
            try:
                self._ops_fh.flush()
            finally:
                self._ops_fh.close()
                self._ops_fh = None
        if self._emb_fh is not None:
            try:
                self._emb_fh.flush()
            finally:
                self._emb_fh.close()
                self._emb_fh = None

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def append(self, record: WalRecord) -> WalRecord:
        """Append one record. Returns the record with ``emb_offset`` set.

        Ordering is: write embedding frame → flush → record the offset →
        write jsonl line → flush. If the embedding write survives and the
        jsonl line does not, replay will truncate the orphan binary frame
        on next open. We never end up with a jsonl pointing at a missing
        binary frame.
        """
        if self._ops_fh is None or self._emb_fh is None:
            raise RuntimeError("WAL.append called before open()")

        emb_offset: int | None = None
        if record.op == "store":
            if record.embedding is None:
                raise ValueError("store record missing embedding")
            payload = _pack_embedding(record.embedding)
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            header = _FRAME_HEADER.pack(len(payload), crc)
            # Re-stat the file under the bundle lock so concurrent writers
            # on the same bundle agree on the next offset. The cached
            # ``_emb_size`` can lag if another process appended while we
            # held the python-side handle.
            try:
                emb_offset = self._emb_path.stat().st_size
            except FileNotFoundError:
                emb_offset = self._emb_size
            self._emb_fh.write(header)
            self._emb_fh.write(payload)
            self._emb_fh.flush()
            self._emb_size = emb_offset + len(header) + len(payload)
        elif record.op in ("forget", "update_metadata"):
            # Neither consumes the binary file — metadata-only mutations.
            pass
        else:
            raise ValueError(f"unknown WAL op {record.op!r}")

        line = self._encode_line(record, emb_offset)
        self._ops_fh.write(line + "\n")
        self._ops_fh.flush()
        self._record_count += 1

        # Durability policy: sync fsyncs immediately, batch every N,
        # async never (caller flushes on demand).
        if self._durability == "sync":
            self._fsync_both()
        elif self._durability == "batch":
            self._batch_since_flush += 1
            if self._batch_since_flush >= self._batch_size:
                self._fsync_both()
                self._batch_since_flush = 0

        return WalRecord(
            op=record.op,
            node_id=record.node_id,
            text=record.text,
            metadata=record.metadata,
            timestamp_step=record.timestamp_step,
            embedding=record.embedding,
            emb_offset=emb_offset,
        )

    def flush(self) -> None:
        """Force-sync both files to stable storage."""
        if self._ops_fh is not None:
            self._ops_fh.flush()
        if self._emb_fh is not None:
            self._emb_fh.flush()
        self._fsync_both()
        self._batch_since_flush = 0

    def truncate(self) -> None:
        """Compaction: zero both files, rewrite the header, reset counters.

        Callers hold the bundle lock across a truncate so concurrent
        readers never see a transient half-state.
        """
        was_open = self._ops_fh is not None
        if was_open:
            self.close()
        # Rewrite header on a freshly-truncated jsonl.
        self._write_header()
        # Zero the binary file.
        with open(self._emb_path, "wb"):
            pass
        self._record_count = 0
        self._emb_size = 0
        self._batch_since_flush = 0
        if was_open:
            # Re-open for continued writing.
            self._ops_fh = open(  # noqa: SIM115 — long-lived append handle
                self._ops_path, "a", encoding="utf-8", newline="\n"
            )
            self._emb_fh = open(self._emb_path, "ab")  # noqa: SIM115 — long-lived

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def replay(self) -> Iterator[WalRecord]:
        """Yield records from disk; stop at the first torn/bad record."""
        yield from self._replay_bytes(self._read_raw(), start=0)

    def replay_tail(self, start_offset: int) -> Iterator[WalRecord]:
        """Yield records whose jsonl lines start at or after ``start_offset``.

        Used by :meth:`soma.memory.api.MemoryLayer.reload_if_stale` to
        apply only the WAL tail written since the last read. When
        ``start_offset`` is beyond the file end (e.g. because a compaction
        truncated the WAL), yields nothing — callers interpret that as
        "nothing new to apply".
        """
        raw = self._read_raw()
        if start_offset < 0:
            start_offset = 0
        if start_offset >= len(raw):
            return
        yield from self._replay_bytes(raw, start=start_offset)

    def ops_size_on_disk(self) -> int:
        """Live size of the ops jsonl file. Used by reload_if_stale to
        detect whether a peer writer appended since our last read."""
        if not self._ops_path.exists():
            return 0
        return self._ops_path.stat().st_size

    def _read_raw(self) -> bytes:
        if not self._ops_path.exists():
            return b""
        with open(self._ops_path, "rb") as fh:
            return fh.read()

    def _replay_bytes(self, raw: bytes, *, start: int) -> Iterator[WalRecord]:
        """Shared replay body. ``start`` skips past already-applied bytes.

        When ``start == 0`` we also skip the header line (first line of
        the file). Otherwise we resume mid-file from a record boundary
        the caller tracked via :meth:`ops_size_on_disk` earlier.
        """
        if not raw:
            return

        emb_data: bytes = b""
        if self._emb_path.exists():
            emb_data = self._emb_path.read_bytes()

        pos = start
        if pos == 0:
            # Skip the header line.
            nl = raw.find(b"\n", pos)
            if nl == -1:
                return
            pos = nl + 1

        while pos < len(raw):
            nl = raw.find(b"\n", pos)
            if nl == -1:
                return
            line = raw[pos:nl]
            pos = nl + 1
            if not line:
                # Blank line — shouldn't happen after recovery but stop
                # rather than guess.
                return
            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            op = obj.get("op")
            if op == "store":
                emb_offset = obj.get("emb_offset")
                if not isinstance(emb_offset, int):
                    return
                emb = self._read_frame_at(emb_data, emb_offset)
                if emb is None:
                    return
                yield WalRecord(
                    op="store",
                    node_id=str(obj["node_id"]),
                    text=obj.get("text"),
                    metadata=dict(obj.get("metadata", {})),
                    timestamp_step=int(obj.get("timestamp_step", 0)),
                    embedding=emb,
                    emb_offset=emb_offset,
                )
            elif op == "forget":
                yield WalRecord(
                    op="forget",
                    node_id=str(obj["node_id"]),
                    text=None,
                    metadata={},
                    timestamp_step=int(obj.get("timestamp_step", 0)),
                    embedding=None,
                    emb_offset=None,
                )
            elif op == "update_metadata":
                patch = obj.get("patch", {})
                if not isinstance(patch, dict):
                    return
                yield WalRecord(
                    op="update_metadata",
                    node_id=str(obj["node_id"]),
                    text=None,
                    metadata={"patch": dict(patch)},
                    timestamp_step=int(obj.get("timestamp_step", 0)),
                    embedding=None,
                    emb_offset=None,
                )
            else:
                return

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def size_bytes(self) -> int:
        """Total WAL footprint: ops jsonl + embeddings binary."""
        ops_size = self._ops_path.stat().st_size if self._ops_path.exists() else 0
        emb_size = self._emb_path.stat().st_size if self._emb_path.exists() else 0
        return ops_size + emb_size

    @property
    def record_count(self) -> int:
        """Number of records currently in the WAL (excluding the header)."""
        return self._record_count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _write_header(self) -> None:
        header = {
            "schema": WAL_SCHEMA,
            "embed_dim": self._embed_dim,
            "created_at": time.time(),
            "bundle_version": WAL_BUNDLE_VERSION,
        }
        tmp = self._ops_path.with_suffix(self._ops_path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(header) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(str(tmp), str(self._ops_path))
        except BaseException:
            if tmp.exists():
                with contextlib.suppress(OSError):
                    tmp.unlink()
            raise

    def _encode_line(self, record: WalRecord, emb_offset: int | None) -> str:
        if record.op == "store":
            obj: dict[str, Any] = {
                "op": "store",
                "node_id": record.node_id,
                "text": record.text,
                "metadata": record.metadata,
                "emb_offset": emb_offset,
                "timestamp_step": record.timestamp_step,
            }
        elif record.op == "update_metadata":
            # Patch lives under metadata["patch"] so we don't collide
            # with the "metadata" key used by store records.
            patch = record.metadata.get("patch", {})
            obj = {
                "op": "update_metadata",
                "node_id": record.node_id,
                "patch": dict(patch),
                "timestamp_step": record.timestamp_step,
            }
        else:
            obj = {
                "op": "forget",
                "node_id": record.node_id,
                "timestamp_step": record.timestamp_step,
            }
        # json.dumps with default separators gives us the same round-trip
        # readers rely on.
        return json.dumps(obj)

    def _fsync_both(self) -> None:
        if self._ops_fh is not None:
            with contextlib.suppress(OSError):
                os.fsync(self._ops_fh.fileno())
        if self._emb_fh is not None:
            with contextlib.suppress(OSError):
                os.fsync(self._emb_fh.fileno())

    def _read_frame_at(self, emb_data: bytes, offset: int) -> torch.Tensor | None:
        if offset < 0 or offset + 8 > len(emb_data):
            return None
        length, crc = _FRAME_HEADER.unpack(emb_data[offset : offset + 8])
        end = offset + 8 + length
        if end > len(emb_data):
            return None
        payload = emb_data[offset + 8 : end]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            return None
        try:
            return _unpack_embedding(payload, self._embed_dim)
        except ValueError:
            return None

    def _recover_and_count(self) -> None:
        """Find the last good (line, frame) pair and truncate everything
        after. Populates ``_record_count`` and ``_emb_size``.
        """
        if not self._ops_path.exists():
            self._record_count = 0
            self._emb_size = 0
            return

        # Walk jsonl: find the last byte offset that is the end of a
        # complete line. Anything beyond that is a torn tail.
        with open(self._ops_path, "rb") as fh:
            data = fh.read()
        # Collect complete line boundaries (inclusive of the newline).
        good_boundary = 0  # offset immediately past the last complete line
        i = 0
        while i < len(data):
            j = data.find(b"\n", i)
            if j == -1:
                break
            good_boundary = j + 1
            i = j + 1
        # Truncate the torn tail (if any) so subsequent appends start on
        # a fresh line boundary.
        if good_boundary < len(data):
            with open(self._ops_path, "r+b") as fh:
                fh.truncate(good_boundary)
                fh.flush()
                with contextlib.suppress(OSError):
                    os.fsync(fh.fileno())
            data = data[:good_boundary]

        # Parse each line except the header; pair up with the binary
        # file to find the last known-good binary offset.
        emb_data = b""
        if self._emb_path.exists():
            emb_data = self._emb_path.read_bytes()

        good_emb_end = 0  # truncate embeddings to this offset at the end
        good_record_count = 0
        line_start = 0
        line_idx = 0
        while line_start < len(data):
            line_end = data.find(b"\n", line_start)
            if line_end == -1:
                break
            line = data[line_start:line_end]
            if line_idx == 0:
                # Header line — don't count it as a record, just sanity-
                # check that it parses. If corrupt, we still have to stop
                # there because we can't recover a missing header.
                try:
                    json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Header unreadable → wipe and start over.
                    self._write_header()
                    self._record_count = 0
                    self._emb_size = 0
                    with open(self._emb_path, "wb"):
                        pass
                    return
                line_idx += 1
                line_start = line_end + 1
                continue

            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            op = obj.get("op")
            if op == "store":
                emb_offset = obj.get("emb_offset")
                if not isinstance(emb_offset, int):
                    break
                if emb_offset + 8 > len(emb_data):
                    break
                length, crc = _FRAME_HEADER.unpack(
                    emb_data[emb_offset : emb_offset + 8]
                )
                frame_end = emb_offset + 8 + length
                if frame_end > len(emb_data):
                    break
                payload = emb_data[emb_offset + 8 : frame_end]
                if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
                    break
                good_emb_end = frame_end
                good_record_count += 1
                line_start = line_end + 1
            elif op in ("forget", "update_metadata"):
                good_record_count += 1
                line_start = line_end + 1
            else:
                # Unknown op — stop; we'd rather truncate than replay junk.
                break
            line_idx += 1

        # If we stopped short, truncate ops back to the last good line.
        good_ops_end = line_start
        if good_ops_end < len(data):
            with open(self._ops_path, "r+b") as fh:
                fh.truncate(good_ops_end)
                fh.flush()
                with contextlib.suppress(OSError):
                    os.fsync(fh.fileno())

        # Truncate embeddings to the last known-good frame end.
        if good_emb_end < len(emb_data):
            with open(self._emb_path, "r+b") as fh:
                fh.truncate(good_emb_end)
                fh.flush()
                with contextlib.suppress(OSError):
                    os.fsync(fh.fileno())

        self._record_count = good_record_count
        self._emb_size = good_emb_end
