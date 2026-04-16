"""End-to-end MemoryLayer save/load round-trip against the GCS emulator.

Phase 32 ships the GCSObjectStore adapter behind the ObjectStore
Protocol that ``MemoryLayer.save`` / ``load`` routed through in
Phase 30. This test proves the seam: callers can swap a plain path
for a ``gs://bucket/prefix`` URL and the save-then-load dance still
reconstructs the store content.

Mirrors the Phase-31 S3 test shape byte-for-byte — only the URL
scheme and the mock backend differ.

Mock strategy: ``gcp-storage-emulator`` (community in-process HTTP
server speaking the real GCS JSON API). Respected by
``google-cloud-storage`` via the ``STORAGE_EMULATOR_HOST`` env var.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("google.cloud.storage")
pytest.importorskip("gcp_storage_emulator")
torch = pytest.importorskip("torch")

from gcp_storage_emulator.server import create_server  # noqa: E402
from google.cloud import storage as _gcs  # noqa: E402

from soma.memory.api import MemoryLayer  # noqa: E402
from soma.storage.gcs import GCSObjectStore  # noqa: E402

_DIM = 8


def _toy_embed(text: str) -> torch.Tensor:
    """Deterministic toy embedder: hash to seed a float vector.

    Matches the Phase-31 S3 round-trip test so the two adapters
    exercise the same MemoryLayer save/load path with interchangeable
    inputs.
    """
    import hashlib

    import numpy as np

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(_DIM).astype("float32")
    vec /= float(np.linalg.norm(vec) + 1e-9)
    return torch.from_numpy(vec)


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


@pytest.fixture
def emulator(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Per-test emulator so MemoryLayer save/load sees a clean bucket."""
    port = _free_port()
    server = create_server("localhost", port, in_memory=True, default_bucket="")
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(1.0)
    base = f"http://localhost:{port}"
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", base)
    try:
        yield base
    finally:
        with contextlib.suppress(Exception):
            server.stop()


def test_memory_layer_save_load_gcs_round_trip(
    emulator: str, tmp_path: Path
) -> None:
    """Store a few entries, save to gs://, load from gs://, verify the
    texts + ids survive the round-trip."""
    bucket = f"mem-{uuid.uuid4().hex[:8]}"
    client = _gcs.Client.create_anonymous_client()
    client.project = "test-project"
    with contextlib.suppress(Exception):
        client.create_bucket(bucket)

    mem = MemoryLayer.ephemeral(embed_fn=_toy_embed, embed_dim=_DIM)
    id_a = mem.store("apples are red")
    id_b = mem.store("bananas are yellow")
    id_c = mem.store("cherries are red")
    assert len(mem) == 3

    mem.save(f"gs://{bucket}/bundle")

    # Fresh load — no shared state with ``mem``.
    restored = MemoryLayer.load(
        f"gs://{bucket}/bundle",
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


def test_gcs_local_root_staging_round_trip(emulator: str) -> None:
    """Download-on-open / upload-on-close round-trip of the staging
    temp dir. This is what MemoryLayer.load relies on when the bundle
    has a WAL sidecar or ``bundle.lock`` that needs a real local path.

    Mirrors the Phase-31 S3 test (``test_s3_local_root_staging_round_trip``)
    byte-for-byte.
    """
    bucket = f"stage-{uuid.uuid4().hex[:8]}"
    client = _gcs.Client.create_anonymous_client()
    client.project = "test-project"
    with contextlib.suppress(Exception):
        client.create_bucket(bucket)

    seed = GCSObjectStore(bucket=bucket, prefix="bundle", client=client)
    seed.put_bytes("a.bin", b"original-a")
    seed.put_bytes("sub/b.bin", b"original-b")

    store = GCSObjectStore(bucket=bucket, prefix="bundle", client=client)
    root = store.local_root
    assert root.exists()
    assert (root / "a.bin").read_bytes() == b"original-a"
    assert (root / "sub" / "b.bin").read_bytes() == b"original-b"

    # Mutate the stage.
    (root / "a.bin").write_bytes(b"modified-a")
    (root / "sub").mkdir(exist_ok=True)
    (root / "sub" / "c.bin").write_bytes(b"brand-new-c")

    store.close()

    after = GCSObjectStore(bucket=bucket, prefix="bundle", client=client)
    assert after.get_bytes("a.bin") == b"modified-a"
    assert after.get_bytes("sub/b.bin") == b"original-b"
    assert after.get_bytes("sub/c.bin") == b"brand-new-c"
    after.close()
