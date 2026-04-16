"""End-to-end demo: round-trip a MemoryLayer bundle through s3://.

This is a copy-paste starting point for operators standing up SOMA
against S3 / S3-compatible object storage. It's **not** a maintained
tool — read it, adapt it, then throw it away.

The script runs against a moto-backed in-memory S3, so it needs no
real AWS credentials and no network. Point it at real S3 by deleting
the ``mock_aws`` block and exporting ``AWS_ACCESS_KEY_ID`` +
``AWS_SECRET_ACCESS_KEY`` + ``AWS_DEFAULT_REGION`` instead.

Requires: ``pip install "soma[s3,s3-test]"`` (moto is in the test
extra; in production you only need the ``s3`` extra).

Run:

    python examples/cloud_s3_demo.py

See docs/cloud.md for the deployment-flavoured walkthrough.
"""

from __future__ import annotations

import hashlib
import time

import boto3
import numpy as np
import torch
from moto import mock_aws

from soma.memory.api import MemoryLayer

_BUCKET = "soma-demo-bucket"
_PREFIX = "bundles/demo"
_BUNDLE_URL = f"s3://{_BUCKET}/{_PREFIX}"
_EMBED_DIM = 16


def _toy_embed(text: str) -> torch.Tensor:
    """Deterministic hash-based embedder — keeps the demo self-contained.

    Swap for ``MemoryLayer.with_sbert()`` or your own embed closure in
    real deployments.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(_EMBED_DIM).astype("float32")
    vec /= float(np.linalg.norm(vec) + 1e-9)
    return torch.from_numpy(vec)


@mock_aws
def main() -> None:
    # In real deployments the bucket already exists. Moto gives us a
    # clean in-memory S3 per decorator, so we create it inline.
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=_BUCKET)

    # 1. Build a tiny MemoryLayer and seed a few entries.
    mem = MemoryLayer.ephemeral(embed_fn=_toy_embed, embed_dim=_EMBED_DIM)
    for text in [
        "SOMA bundles can live on S3 or GCS.",
        "MemoryLayer.save writes four objects under the prefix.",
        "MemoryLayer.load reads the whole bundle on cold start.",
        "Cloudflare R2 is S3-compatible via an endpoint override.",
    ]:
        mem.store(text)
    print(f"Seeded {len(mem)} entries.")

    # 2. Save to s3://. Boto3 picks up the moto-mocked credentials
    #    from the decorator automatically.
    t0 = time.perf_counter()
    mem.save(_BUNDLE_URL)
    save_ms = (time.perf_counter() - t0) * 1000
    print(f"save({_BUNDLE_URL}) -> {save_ms:.1f} ms")

    # 3. Load back into a fresh MemoryLayer with no shared state.
    t0 = time.perf_counter()
    restored = MemoryLayer.load(_BUNDLE_URL, embed_fn=_toy_embed)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"load({_BUNDLE_URL}) -> {load_ms:.1f} ms, {len(restored)} entries")

    # 4. Retrieve — the top hit should match the query text exactly.
    query = "SOMA bundles can live on S3 or GCS."
    t0 = time.perf_counter()
    hits = restored.retrieve(query, k=3)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    print(f"retrieve({query!r}) -> {retrieve_ms:.1f} ms")
    for rank, hit in enumerate(hits, start=1):
        print(f"  {rank}. score={hit.score:+.3f}  {hit.text}")

    # 5. Sanity check — the self-query hit is the same text we stored.
    assert hits and hits[0].text == query, "top-1 retrieval should match query"
    print("OK")


if __name__ == "__main__":
    main()
