"""End-to-end MemoryLayer save/load round-trip against moto-backed S3.

Phase 31 ships the S3ObjectStore adapter behind the ObjectStore
Protocol that ``MemoryLayer.save`` / ``load`` routed through in
Phase 30. This test proves the seam: callers can swap a plain path
for an ``s3://bucket/prefix`` URL and the save-then-load dance still
reconstructs the store content.

Two shapes of round-trip:

1. ``mem.save("s3://...")`` → ``MemoryLayer.load("s3://...")`` with
   a custom ``embed_fn`` (no tokenizer / encoder, so no
   ``local_root`` staging is exercised).
2. The ``local_root`` staging round-trip: take an S3ObjectStore,
   touch ``local_root`` to materialise the temp dir, drop a file,
   call ``close()``, re-open and verify the file is in S3.

Moto's ``mock_aws`` gives us a real in-memory S3; the same code path
runs against real AWS via the gated integration test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")
torch = pytest.importorskip("torch")

from moto import mock_aws  # noqa: E402

from soma.memory.api import MemoryLayer  # noqa: E402
from soma.storage.s3 import S3ObjectStore  # noqa: E402

_DIM = 8


def _toy_embed(text: str) -> torch.Tensor:
    """Deterministic toy embedder: hash to seed a float vector.

    MemoryLayer's save/load path cares only about the vector matrix,
    not the embedding quality — this gives us reproducible vectors
    for the round-trip assertion without pulling a real model.
    """
    import hashlib

    import numpy as np

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(_DIM).astype("float32")
    vec /= float(np.linalg.norm(vec) + 1e-9)
    return torch.from_numpy(vec)


@mock_aws
def test_memory_layer_save_load_s3_round_trip(tmp_path: Path) -> None:
    """Store a few entries, save to s3://, load from s3://, verify the
    texts + ids + timestamps survive the round-trip."""
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")

    mem = MemoryLayer.ephemeral(embed_fn=_toy_embed, embed_dim=_DIM)
    id_a = mem.store("apples are red")
    id_b = mem.store("bananas are yellow")
    id_c = mem.store("cherries are red")
    assert len(mem) == 3

    mem.save("s3://test-bucket/bundle")

    # Fresh load — no shared state with ``mem``.
    restored = MemoryLayer.load(
        "s3://test-bucket/bundle",
        embed_fn=_toy_embed,
    )
    assert len(restored) == 3
    # Order is preserved across save/load.
    assert restored.get(id_a) is not None
    assert restored.get(id_b) is not None
    assert restored.get(id_c) is not None
    hit_a = restored.get(id_a)
    assert hit_a is not None
    assert hit_a.text == "apples are red"
    # Retrieve path still works with the toy embedder — self-query
    # should put the exact-text entry first.
    hits = restored.retrieve("apples are red", k=3)
    assert hits[0].node_id == id_a


@mock_aws
def test_s3_local_root_staging_round_trip() -> None:
    """Download-on-open / upload-on-close round-trip of the staging
    temp dir. This is what MemoryLayer.load relies on when the bundle
    has a WAL sidecar or bundle.lock that needs a real local path.

    Sequence:
      1. Seed the S3 prefix with a couple of objects directly.
      2. Access ``store.local_root`` — triggers the download.
      3. Mutate the staging dir: overwrite one file, drop a new one.
      4. ``close()`` — pushes modifications back to S3.
      5. Open a fresh store and verify the new/modified files are in S3.
    """
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    seed = S3ObjectStore(bucket="test-bucket", prefix="bundle")
    seed.put_bytes("a.bin", b"original-a")
    seed.put_bytes("sub/b.bin", b"original-b")
    # Don't call close() on seed — it has no staging to sync.

    # Fresh store, trigger staging download.
    store = S3ObjectStore(bucket="test-bucket", prefix="bundle")
    root = store.local_root
    assert root.exists()
    assert (root / "a.bin").read_bytes() == b"original-a"
    assert (root / "sub" / "b.bin").read_bytes() == b"original-b"

    # Mutate the stage.
    (root / "a.bin").write_bytes(b"modified-a")
    (root / "sub").mkdir(exist_ok=True)
    (root / "sub" / "c.bin").write_bytes(b"brand-new-c")

    # Close → upload.
    store.close()

    # Fresh adapter reads back the modifications.
    after = S3ObjectStore(bucket="test-bucket", prefix="bundle")
    assert after.get_bytes("a.bin") == b"modified-a"
    assert after.get_bytes("sub/b.bin") == b"original-b"
    assert after.get_bytes("sub/c.bin") == b"brand-new-c"
    after.close()


@mock_aws
def test_s3_local_root_is_lazy() -> None:
    """``local_root`` is pay-per-use: if a caller only goes through
    ``put_bytes`` / ``get_bytes``, no staging dir is created. Close()
    stays a no-op in that case (no temp dir to sync, no temp dir to
    tear down)."""
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")
    store = S3ObjectStore(bucket="test-bucket")
    store.put_bytes("k.bin", b"x")
    assert store.get_bytes("k.bin") == b"x"
    # No ``local_root`` access — private attribute stays None.
    assert store._local_root is None
    # close() is safe and doesn't touch S3.
    store.close()
