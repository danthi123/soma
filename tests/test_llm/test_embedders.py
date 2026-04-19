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
