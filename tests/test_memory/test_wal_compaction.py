"""Compaction tests — auto-rewrite the snapshot when the WAL gets big.

Compaction triggers fire from store/store_batch/forget. A daemon thread
copies in-memory state refs under the bundle lock, writes a fresh
snapshot outside the lock (atomic tmp + rename), then re-acquires the
lock to swap the bundle + truncate the WAL. Only one compaction runs
at a time; concurrent triggers are no-ops until the first finishes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from soma.memory.api import MemoryLayer


def _hash_embed(text: str) -> torch.Tensor:
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


def _wait_for_compaction(mem: MemoryLayer, timeout: float = 10.0) -> None:
    """Block until the background compaction thread (if any) is finished."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        thread = getattr(mem, "_compaction_thread", None)
        if thread is None or not thread.is_alive():
            return
        time.sleep(0.02)
    raise AssertionError("compaction thread did not finish in time")


def test_compaction_triggered_by_record_count(tmp_path: Path) -> None:
    """>threshold records should trigger compaction and shrink the WAL
    record count below the snapshot's entry count.
    """
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    # Force the threshold down so we don't have to write 10k records in
    # a test — the real default still fires at 10_000.
    mem._compaction_record_threshold = 50
    for i in range(60):
        mem.store(f"entry {i}")
    _wait_for_compaction(mem)
    # After compaction, snapshot exists and WAL record count is strictly
    # less than the total store count (most records were compacted into
    # the snapshot). A small tail of records appended during compaction
    # remains in the WAL — that's expected.
    assert (bundle / "memory_index.json").exists()
    assert mem._wal is not None
    assert mem._wal.record_count < 60
    # All 60 entries still readable.
    assert len(mem) == 60
    mem.close()

    # Fresh reload must recover the full state from snapshot + WAL tail.
    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem2) == 60
    mem2.close()


def test_compaction_triggered_by_size(tmp_path: Path) -> None:
    """When WAL bytes exceeds max(size_floor, 1.0 × snapshot), fire
    compaction. We tune the floor down so the check fires on a handful.
    """
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    mem._compaction_size_floor = 128  # bytes, tiny for testing
    mem._compaction_record_threshold = 10**9  # disable the record trigger
    for i in range(20):
        mem.store(f"size trigger entry {i}")
    _wait_for_compaction(mem)
    assert (bundle / "memory_index.json").exists()
    assert mem._wal is not None
    # Compaction shrank the WAL tail; at least some records rolled into
    # the snapshot.
    assert mem._wal.record_count < 20
    assert len(mem) == 20
    mem.close()

    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem2) == 20
    mem2.close()


def test_compaction_preserves_all_entries(tmp_path: Path) -> None:
    """Record count before vs after compaction is identical; every id
    still retrievable after compaction + a fresh reload.
    """
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    mem._compaction_record_threshold = 30
    ids = [mem.store(f"preserved {i}", metadata={"idx": i}) for i in range(40)]
    _wait_for_compaction(mem)
    assert len(mem) == 40
    for i, nid in enumerate(ids):
        hit = mem.get(nid)
        assert hit is not None
        assert hit.metadata == {"idx": i}
    mem.close()

    # Fresh reload must see the same state.
    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem2) == 40
    for i, nid in enumerate(ids):
        hit = mem2.get(nid)
        assert hit is not None
        assert hit.text == f"preserved {i}"
    mem2.close()


def test_compaction_under_concurrent_store(tmp_path: Path) -> None:
    """Two threads storing while compaction runs — no data loss, no
    deadlock. threading.Event gates the second thread so it hits the
    WAL mid-compaction.
    """
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    mem._compaction_record_threshold = 20

    start_second = threading.Event()
    done_second = threading.Event()
    ids_second: list[str] = []

    def worker() -> None:
        start_second.wait(timeout=5.0)
        for i in range(10):
            ids_second.append(mem.store(f"thread B {i}"))
        done_second.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    ids_first: list[str] = []
    for i in range(25):
        ids_first.append(mem.store(f"thread A {i}"))
        if i == 15:
            start_second.set()
    done_second.wait(timeout=10.0)
    t.join(timeout=5.0)
    assert not t.is_alive()
    _wait_for_compaction(mem)

    total = 25 + 10
    assert len(mem) == total
    for nid in ids_first + ids_second:
        assert mem.get(nid) is not None
    mem.close()


def test_compaction_is_serialized(tmp_path: Path) -> None:
    """Two back-to-back triggers must not spawn two compaction threads.
    The second trigger sees the first still running and is a no-op.
    """
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    mem._compaction_record_threshold = 5
    for i in range(6):
        mem.store(f"x {i}")
    first_thread = mem._compaction_thread
    # Immediately trigger again before the first finishes. We cannot
    # guarantee the first thread is still running by the time this test
    # body runs, so we assert the safety contract: _maybe_compact MUST
    # reuse an in-flight thread rather than spawning a second.
    for i in range(6, 12):
        mem.store(f"y {i}")
    # Either the first thread is still the active one OR it finished
    # and a new cycle started cleanly — what MUST NOT happen is two
    # concurrent threads at the same time.
    if first_thread is not None and first_thread.is_alive():
        assert mem._compaction_thread is first_thread
    _wait_for_compaction(mem)
    mem.close()
