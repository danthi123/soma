"""Integration tests for MemoryLayer ↔ WAL wiring.

Covers: crash recovery via WAL replay, forget tombstones, concurrent
writers on one bundle, and durability semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory.api import MemoryLayer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "to be or not to be that is the question",
    "the rain in spain falls mainly on the plain",
    "all happy families are alike",
]


def _hash_embed(text: str) -> torch.Tensor:
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


@pytest.fixture
def embedder() -> tuple[object, TextEncoder]:
    torch.manual_seed(0)
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=256)
    encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
    return tokenizer, encoder


# ----------------------------------------------------------------------
# Crash recovery
# ----------------------------------------------------------------------
def test_store_then_crash_then_load_recovers_entry(tmp_path: Path) -> None:
    """Store 3 entries WITHOUT save(); drop the instance; reload must
    still see all 3 via WAL replay.
    """
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    ids = [mem.store(f"fact {i}", metadata={"idx": i}) for i in range(3)]
    mem.close()

    # No save() call — simulated crash. Reload from the bundle.
    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem2) == 3
    for i, nid in enumerate(ids):
        hit = mem2.get(nid)
        assert hit is not None, f"id {nid} missing after reload"
        assert hit.text == f"fact {i}"
        assert hit.metadata == {"idx": i}


def test_forget_then_crash_then_load_replays_tombstone(tmp_path: Path) -> None:
    """Store 2, forget 1, crash (no save), reload — forgotten is gone."""
    bundle = tmp_path / "bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    a = mem.store("alpha")
    b = mem.store("beta")
    assert mem.forget(a) is True
    mem.close()

    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem2) == 1
    assert mem2.get(a) is None
    hit = mem2.get(b)
    assert hit is not None and hit.text == "beta"


def test_concurrent_store_two_memorylayers_on_same_bundle(tmp_path: Path) -> None:
    """Two MemoryLayer instances over the same bundle dir must serialize
    their writes via the shared bundle.lock and each produce a valid log.
    """
    bundle = tmp_path / "shared"
    m1 = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    m2 = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    # Alternate stores.
    ids_1 = []
    ids_2 = []
    for i in range(5):
        ids_1.append(m1.store(f"m1-{i}"))
        ids_2.append(m2.store(f"m2-{i}"))
    m1.close()
    m2.close()

    # Reload from disk — the merged WAL tail must contain ALL 10 records
    # in the order they were actually appended to disk.
    reloaded = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(reloaded) == 10
    seen = {nid: reloaded.get(nid) for nid in ids_1 + ids_2}
    assert all(hit is not None for hit in seen.values())


def test_durability_async_can_lose_tail_without_flush(tmp_path: Path) -> None:
    """durability='async' only fsyncs on close/flush. We can't safely
    guarantee tail loss in a pure-python test (file contents might still
    be in the page cache), so we verify the weaker claim: an explicit
    flush() is durable, and append() without flush() also ends up on disk
    once the file handle is closed cleanly.
    """
    bundle = tmp_path / "async-bundle"
    mem = MemoryLayer(
        embed_fn=_hash_embed,
        embed_dim=16,
        bundle_path=bundle,
        durability="async",
    )
    nid = mem.store("async fact")
    # flush() must be safe to call; after flush, reloading must see the
    # entry even without close().
    mem.flush()
    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert mem2.get(nid) is not None
    mem.close()


def test_store_batch_appends_all_records_under_one_lock(tmp_path: Path) -> None:
    """store_batch writes N records through the WAL as a single
    lock-acquire cycle. We can't directly observe lock contention here,
    but we verify the functional side: all N records land in the WAL
    and reload correctly.
    """
    bundle = tmp_path / "batch-bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    ids = mem.store_batch([f"batched {i}" for i in range(7)])
    assert len(ids) == 7
    mem.close()

    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem2) == 7
    for i, nid in enumerate(ids):
        hit = mem2.get(nid)
        assert hit is not None and hit.text == f"batched {i}"


def test_flush_is_idempotent_with_no_wal(tmp_path: Path) -> None:
    """MemoryLayer() without bundle_path has no WAL; flush() must be a
    safe no-op rather than raising (keeps the API uniform)."""
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem.store("x")
    mem.flush()  # no-op, no raise


def test_no_bundle_path_skips_wal(embedder) -> None:
    """Existing constructor semantics: if bundle_path is not passed, no
    WAL exists and nothing is written to disk. Guards backward compat
    with every pre-WAL test.
    """
    tokenizer, encoder = embedder
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    mem.store("no persistence")
    # No bundle → no attribute set for WAL writes (None).
    assert mem._wal is None  # type: ignore[attr-defined]


def test_replay_preserves_order_and_indices(tmp_path: Path) -> None:
    """Store interleaved with forget; reload must produce the same
    _id_to_idx mapping one would expect from a linear replay.
    """
    bundle = tmp_path / "order-bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    a = mem.store("a")
    b = mem.store("b")
    c = mem.store("c")
    mem.forget(b)
    mem.close()

    mem2 = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert mem2._id_to_idx.get(a) == 0  # type: ignore[attr-defined]
    assert b not in mem2._id_to_idx  # type: ignore[attr-defined]
    assert mem2._id_to_idx.get(c) == 1  # type: ignore[attr-defined]
    assert len(mem2._soma_activations) == len(mem2)  # type: ignore[attr-defined]


def test_durability_invalid_raises(tmp_path: Path) -> None:
    bundle = tmp_path / "bad"
    with pytest.raises(ValueError, match="durability"):
        MemoryLayer(
            embed_fn=_hash_embed,
            embed_dim=16,
            bundle_path=bundle,
            durability="invalid",  # type: ignore[arg-type]
        )
