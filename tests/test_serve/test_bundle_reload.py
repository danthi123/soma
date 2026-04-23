"""Server rehydrates sbert-backed bundles under their persisted model
name, not the current ``SOMA_EMBED_MODEL`` env.

Covers a correctness bug where the server always called
``MemoryLayer.load(path, embed_fn=_embed_fn())`` regardless of how the
bundle was saved — so a bundle saved under model A would silently be
reopened under model B if the server's env pointed elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma.memory import MemoryLayer


def _tiny_embed(dim: int):
    import torch

    def _fn(text: str) -> torch.Tensor:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        # sha256 is 32 bytes; tile to reach ``dim`` floats.
        buf = (h * ((dim // len(h)) + 1))[:dim]
        vec = [b / 255.0 for b in buf]
        return torch.tensor(vec, dtype=torch.float32)

    return _fn


@pytest.fixture()
def sbert_bundle(tmp_path: Path) -> Path:
    """Bundle whose memory_index.json records a specific sbert name."""
    bundle = tmp_path / "brain"
    mem = MemoryLayer(embed_fn=_tiny_embed(384), embed_dim=384)
    mem.store("alex lives in portland")
    mem._sbert_model_name = "sentence-transformers/all-mpnet-base-v2"
    mem.save(bundle)
    idx = json.loads((bundle / "memory_index.json").read_text())
    assert idx.get("sbert_model_name") == "sentence-transformers/all-mpnet-base-v2"
    return bundle


def test_get_mem_uses_persisted_sbert_name(sbert_bundle, monkeypatch):
    from soma import serve

    monkeypatch.setattr(serve, "BUNDLE_PATH", sbert_bundle)
    monkeypatch.setattr(serve, "EMBED_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setattr(serve, "_mem_cache", {})

    captured: dict[str, object] = {}

    def fake_load_with_sbert(cls, src, *, model_name=None, **kwargs):
        captured["called"] = True
        captured["src"] = src
        captured["model_name"] = model_name
        return MemoryLayer(embed_fn=_tiny_embed(384), embed_dim=384)

    monkeypatch.setattr(MemoryLayer, "load_with_sbert", classmethod(fake_load_with_sbert))
    original_load = MemoryLayer.load

    def fake_load(cls, *args, **kwargs):
        captured["load_called"] = True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(MemoryLayer, "load", classmethod(fake_load))

    _ = serve._get_mem()

    assert captured.get("called") is True
    assert "load_called" not in captured


def test_get_mem_falls_back_to_plain_load_when_no_sbert_name(tmp_path, monkeypatch):
    from soma import serve

    bundle = tmp_path / "brain"
    mem = MemoryLayer(embed_fn=_tiny_embed(64), embed_dim=64)
    mem.store("hello")
    mem.save(bundle)
    idx_path = bundle / "memory_index.json"
    idx = json.loads(idx_path.read_text())
    idx.pop("sbert_model_name", None)
    idx_path.write_text(json.dumps(idx))

    monkeypatch.setattr(serve, "BUNDLE_PATH", bundle)
    monkeypatch.setattr(serve, "_mem_cache", {})
    monkeypatch.setattr(serve, "_embed_fn", lambda: _tiny_embed(64))

    captured: dict[str, object] = {}

    def fake_load(cls, src, *, embed_fn=None, **kwargs):
        captured["load_called"] = True
        return MemoryLayer(embed_fn=_tiny_embed(64), embed_dim=64)

    def fake_load_with_sbert(cls, *args, **kwargs):
        captured["load_with_sbert_called"] = True
        raise AssertionError("should not be called")

    monkeypatch.setattr(MemoryLayer, "load", classmethod(fake_load))
    monkeypatch.setattr(MemoryLayer, "load_with_sbert", classmethod(fake_load_with_sbert))

    _ = serve._get_mem()

    assert captured.get("load_called") is True
    assert "load_with_sbert_called" not in captured
