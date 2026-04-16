"""Schema v2 migration tests.

v1 bundles predate the WAL work. They must still load cleanly (legacy
mode, no WAL). On the next save() the bundle is rewritten as v2 —
schema_version=2 in memory_index.json and WAL sidecars present.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from soma.memory.api import MemoryLayer


def _hash_embed(text: str) -> torch.Tensor:
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


def _write_legacy_v1_bundle(path: Path) -> list[str]:
    """Hand-craft a v1 bundle (no WAL), matching what the pre-WAL
    MemoryLayer.save() would produce. Returns the node_ids in order.
    """
    path.mkdir(parents=True, exist_ok=True)
    ids = ["id-a", "id-b", "id-c"]
    texts = ["alpha fact", "beta fact", "gamma fact"]
    torch.manual_seed(0)
    embeds = torch.randn(3, 16)
    torch.save(embeds, path / "memory_embeddings.pt")
    index = {
        "schema_version": 1,
        "embed_dim": 16,
        "embed_type": "custom",
        "step": 3,
        "entries": [
            {"node_id": ids[i], "text": texts[i], "metadata": {"i": i}, "timestamp_step": i}
            for i in range(3)
        ],
    }
    (path / "memory_index.json").write_text(json.dumps(index), encoding="utf-8")
    return ids


def test_loads_v1_bundle_without_wal(tmp_path: Path) -> None:
    """A hand-written v1 bundle (no WAL files) loads cleanly and keeps
    exactly the stored entries. No WAL should be required.
    """
    bundle = tmp_path / "v1-bundle"
    ids = _write_legacy_v1_bundle(bundle)
    mem = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem) == 3
    for nid in ids:
        hit = mem.get(nid)
        assert hit is not None
    mem.close()


def test_save_writes_v2_and_starts_wal(tmp_path: Path) -> None:
    """After the current save() path runs, the bundle is v2:
    memory_index.json has schema_version=2 and the WAL sidecars exist.
    """
    bundle = tmp_path / "v2-bundle"
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16, bundle_path=bundle)
    for i in range(3):
        mem.store(f"fact {i}")
    mem.save(bundle)
    index = json.loads((bundle / "memory_index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    assert (bundle / "memory_ops.wal.jsonl").exists()
    assert (bundle / "memory_embeddings.wal.bin").exists()
    mem.close()


def test_v1_bundle_upgraded_on_next_save(tmp_path: Path) -> None:
    """Load a v1 bundle, add one entry, save: the resulting bundle is v2."""
    bundle = tmp_path / "upgrade-bundle"
    legacy_ids = _write_legacy_v1_bundle(bundle)
    mem = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(mem) == 3
    new_id = mem.store("fourth fact after upgrade")
    mem.save(bundle)
    index = json.loads((bundle / "memory_index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    # Entries include all 4 (3 legacy + 1 new).
    node_ids = [e["node_id"] for e in index["entries"]]
    assert set(node_ids) == set(legacy_ids) | {new_id}
    mem.close()


def test_v1_load_then_roundtrip_preserves_data(tmp_path: Path) -> None:
    """After a v1 → v2 upgrade + reload, all original entries still
    have the right text and metadata."""
    bundle = tmp_path / "roundtrip"
    _write_legacy_v1_bundle(bundle)
    mem = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    mem.store("new after upgrade")
    mem.save(bundle)
    mem.close()

    reloaded = MemoryLayer.load(bundle, embed_fn=_hash_embed)
    assert len(reloaded) == 4
    # Original three ids still present with their original text.
    expected = {"id-a": "alpha fact", "id-b": "beta fact", "id-c": "gamma fact"}
    for nid, txt in expected.items():
        hit = reloaded.get(nid)
        assert hit is not None
        assert hit.text == txt
    reloaded.close()


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """If a bundle reports schema_version=99, load must refuse rather
    than silently mis-interpret it."""
    bundle = tmp_path / "future-bundle"
    _write_legacy_v1_bundle(bundle)
    # Tamper with the schema version.
    idx_path = bundle / "memory_index.json"
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    data["schema_version"] = 99
    idx_path.write_text(json.dumps(data), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="schema"):
        MemoryLayer.load(bundle, embed_fn=_hash_embed)
