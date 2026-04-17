"""Tests for the interaction loop."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop


def _make_config() -> SOMAConfig:
    return SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )


class TestInteractionLoop:
    def test_init(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        assert loop.predictive_soma is not None
        assert loop.tracker is not None

    def test_encode_text(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        loop.train_tokenizer(["hello world", "test input"])
        vec = loop.encode_text("hello world")
        assert isinstance(vec, torch.Tensor)
        assert vec.shape[-1] == 64

    def test_encode_text_requires_tokenizer(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
        )
        try:
            loop.encode_text("hello")
            raise AssertionError("Expected RuntimeError")  # noqa: TRY301
        except RuntimeError:
            pass

    def test_process_without_llm(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        loop.train_tokenizer(["hello world", "how are you"])
        result = loop.process_input("hello world", call_llm=False)
        assert "soma_state" in result
        assert "prediction_error" in result
        assert result["response"] is None

    @patch("soma.developmental.interaction.requests.post")
    def test_process_with_llm(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "I'm learning about this."}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        loop.train_tokenizer(["hello world", "test"])
        result = loop.process_input("hello world", call_llm=True)
        assert result["response"] == "I'm learning about this."
        assert mock_post.called

    def test_tracker_records_steps(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        loop.train_tokenizer(["one", "two", "three"])
        loop.process_input("one", call_llm=False)
        loop.process_input("two", call_llm=False)
        assert loop.tracker.step_count == 2
