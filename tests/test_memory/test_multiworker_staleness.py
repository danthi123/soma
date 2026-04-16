"""Multi-worker reload_if_stale — second reader sees fresh WAL tail.

When two uvicorn workers share one bundle dir, the second worker's
MemoryLayer instance must pick up stores appended by the first worker
before answering retrieve. ``reload_if_stale()`` scans the WAL for
records beyond ``_last_wal_offset`` and applies them in-memory under
the bundle lock.
"""

from __future__ import annotations

from pathlib import Path

import torch

from soma.memory.api import MemoryLayer


def _hash_embed(text: str) -> torch.Tensor:
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


def test_second_process_sees_new_stores_on_retrieve(tmp_path: Path) -> None:
    """Writer on one MemoryLayer, reader on a separate MemoryLayer over
    the same bundle. Reader must see the writer's stores after
    reload_if_stale() — mimics two uvicorn workers on one bundle.
    """
    bundle = tmp_path / "shared"
    writer = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    reader = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)

    # Reader is empty — writer has not stored anything yet.
    assert len(reader) == 0
    assert reader.retrieve("nothing", k=5) == []

    # Writer stores three entries. Reader cannot see them yet (separate
    # in-memory list).
    ids = [writer.store(f"fresh {i}") for i in range(3)]
    assert len(writer) == 3
    assert len(reader) == 0  # still stale

    # reload_if_stale picks up the three records from the WAL tail.
    reader.reload_if_stale()
    assert len(reader) == 3
    for nid in ids:
        hit = reader.get(nid)
        assert hit is not None

    writer.close()
    reader.close()


def test_reload_if_stale_is_idempotent(tmp_path: Path) -> None:
    """Calling reload_if_stale twice with no new WAL records is a cheap
    no-op — nothing changes, no records double-applied.
    """
    bundle = tmp_path / "idem"
    writer = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    reader = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    writer.store("once")
    reader.reload_if_stale()
    assert len(reader) == 1
    reader.reload_if_stale()
    assert len(reader) == 1
    writer.close()
    reader.close()


def test_reload_if_stale_noop_without_bundle(tmp_path: Path) -> None:
    """In-memory-only MemoryLayer (no bundle_path) — reload_if_stale
    must be a safe no-op, not raise."""
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem.store("x")
    mem.reload_if_stale()  # must not raise
    assert len(mem) == 1


def test_reload_if_stale_picks_up_forget_tombstone(tmp_path: Path) -> None:
    """Writer stores then forgets; reader's reload_if_stale must apply
    the tombstone so the forgotten id no longer retrieves.
    """
    bundle = tmp_path / "forget-sync"
    writer = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    reader = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    a = writer.store("keeper")
    b = writer.store("doomed")
    reader.reload_if_stale()
    assert len(reader) == 2

    writer.forget(b)
    reader.reload_if_stale()
    assert len(reader) == 1
    assert reader.get(a) is not None
    assert reader.get(b) is None
    writer.close()
    reader.close()


def test_serve_calls_reload_if_stale_on_retrieve(tmp_path: Path) -> None:
    """_get_mem in serve.py must call reload_if_stale on the cached
    MemoryLayer so a second-worker retrieve sees a first-worker store.

    We simulate both workers inline: preload a fake cache entry that is
    already open on the bundle, then have an external writer append.
    The next _get_mem() should return a fresh-reloaded instance.
    """
    from soma import serve

    bundle = tmp_path / "serve-sync"
    # Inject a stub embedder so no sbert download is needed.
    serve._embed_fn_cache = _hash_embed
    serve.BUNDLE_PATH = bundle
    serve._mem_cache.clear()

    # Pre-seed the cache with a reader MemoryLayer over this bundle.
    reader = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    serve._mem_cache["__default__"] = reader

    # External writer appends via a separate MemoryLayer.
    writer = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    nid = writer.store("fresh entry from writer")
    writer.close()

    # Getting the cached reader and issuing a retrieve should pick up
    # the new entry through the _get_mem -> reload_if_stale path. We
    # emulate the serve.py wiring by calling _get_mem + retrieve.
    mem = serve._get_mem()
    hit = mem.get(nid)
    assert hit is not None, "reader did not pick up writer's entry"
    assert hit.text == "fresh entry from writer"

    reader.close()
    serve._mem_cache.clear()
    serve._embed_fn_cache = None
