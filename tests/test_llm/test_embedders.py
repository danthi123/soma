"""Tests for LLM teacher embedders (Direction 4a)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestOllamaEmbedder:
    def test_embed_returns_tensor_of_expected_dim(self) -> None:
        from soma.llm.embedders import OllamaEmbedder

        fake_payload = {"embedding": [0.1] * 1024}
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps(fake_payload).encode("utf-8")
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)

        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        with patch(
            "soma.llm.embedders.urllib.request.urlopen",
            return_value=fake_response,
        ):
            result = embedder.embed("hello world")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1024,)
        assert result.dtype == torch.float32

    def test_embed_raises_on_server_unreachable(self) -> None:
        import urllib.error

        from soma.llm.embedders import OllamaEmbedder

        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        with patch(
            "soma.llm.embedders.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Ollama"):
                embedder.embed("hello")

    def test_embed_batch_returns_stacked_tensor(self) -> None:
        from soma.llm.embedders import OllamaEmbedder

        fake_payload = {"embedding": [0.1] * 1024}

        def fake_urlopen(*args, **kwargs):  # type: ignore[no-untyped-def]
            resp = MagicMock()
            resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        with patch(
            "soma.llm.embedders.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = embedder.embed_batch(["a", "b", "c"])
        assert result.shape == (3, 1024)

    def test_embedder_name_reflects_model(self) -> None:
        from soma.llm.embedders import OllamaEmbedder

        embedder = OllamaEmbedder(model="nomic-embed-text")
        assert "nomic-embed-text" in embedder.name


class TestCachedEmbedder:
    def test_cache_hit_skips_underlying_call(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        underlying.embed.return_value = torch.tensor([1.0, 2.0, 3.0])

        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        r1 = wrapped.embed("cat")
        r2 = wrapped.embed("cat")
        assert torch.equal(r1, r2)
        # Second call should be served from cache
        assert underlying.embed.call_count == 1

    def test_cache_miss_calls_underlying(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        underlying.embed.side_effect = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        ]

        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        wrapped.embed("a")
        wrapped.embed("b")
        assert underlying.embed.call_count == 2

    def test_cache_persists_across_instances(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        under1 = MagicMock()
        under1.name = "ollama-embed:mxbai-embed-large"
        under1.embed.return_value = torch.tensor([0.5, 0.5])

        wrapper1 = CachedEmbedder(teacher=under1, cache_dir=str(tmp_path))
        wrapper1.embed("foo")
        assert under1.embed.call_count == 1

        # Second wrapper using same cache dir should hit disk
        under2 = MagicMock()
        under2.name = "ollama-embed:mxbai-embed-large"
        under2.embed.return_value = torch.tensor([0.5, 0.5])
        wrapper2 = CachedEmbedder(teacher=under2, cache_dir=str(tmp_path))
        result = wrapper2.embed("foo")
        assert torch.equal(result, torch.tensor([0.5, 0.5]))
        assert under2.embed.call_count == 0  # Cache hit from disk

    def test_embed_batch_partially_caches(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        underlying.embed.side_effect = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        ]

        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        wrapped.embed_batch(["hot", "cold"])
        assert underlying.embed.call_count == 2

        # Re-batch: should all come from cache
        underlying.embed.side_effect = None
        underlying.embed.reset_mock()
        result = wrapped.embed_batch(["hot", "cold"])
        assert result.shape == (2, 2)
        assert underlying.embed.call_count == 0

    def test_different_models_have_separate_caches(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        under_a = MagicMock()
        under_a.name = "ollama-embed:model-a"
        under_a.embed.return_value = torch.tensor([1.0])
        under_b = MagicMock()
        under_b.name = "ollama-embed:model-b"
        under_b.embed.return_value = torch.tensor([2.0])

        wa = CachedEmbedder(teacher=under_a, cache_dir=str(tmp_path))
        wb = CachedEmbedder(teacher=under_b, cache_dir=str(tmp_path))
        wa.embed("same text")
        wb.embed("same text")
        assert under_a.embed.call_count == 1
        assert under_b.embed.call_count == 1  # not a cache hit from model-a

    def test_cached_embedder_name_preserves_model(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        assert "mxbai-embed-large" in wrapped.name
