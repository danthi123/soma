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

    def test_encoder_training_changes_embeddings(self) -> None:
        """Encoder training should modify embedding weights over time."""
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            train_encoder=True,
            encoder_lr=0.01,
        )
        corpus = ["hello world", "the weather is nice", "cooking pasta"]
        loop.train_tokenizer(corpus)

        # Snapshot initial embedding weights
        initial_weights = loop._encoder.embedding.weight.data.clone()

        # Process several inputs
        for text in corpus * 3:
            loop.process_input(text, call_llm=False)

        # Weights should have changed
        final_weights = loop._encoder.embedding.weight.data
        weight_diff = (final_weights - initial_weights).abs().sum().item()
        assert weight_diff > 0, "Encoder weights should change with training"

    def test_encoder_training_disabled(self) -> None:
        """With train_encoder=False, embeddings should not change."""
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            train_encoder=False,
        )
        corpus = ["hello world", "the weather is nice"]
        loop.train_tokenizer(corpus)

        initial_weights = loop._encoder.embedding.weight.data.clone()

        for text in corpus * 3:
            loop.process_input(text, call_llm=False)

        final_weights = loop._encoder.embedding.weight.data
        weight_diff = (final_weights - initial_weights).abs().sum().item()
        assert weight_diff == 0, "Encoder weights should NOT change without training"

    def test_save_load_preserves_encoder(self, tmp_path) -> None:
        """Save/load should preserve encoder weights."""
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            train_encoder=True,
            encoder_lr=0.01,
        )
        corpus = ["hello world", "cooking pasta", "playing music"]
        loop.train_tokenizer(corpus)

        # Train for a few steps
        for text in corpus * 3:
            loop.process_input(text, call_llm=False)

        # Save
        save_dir = str(tmp_path / "state")
        loop.save(save_dir)

        # Create a new loop, train tokenizer, load
        loop2 = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            train_encoder=True,
        )
        loop2.train_tokenizer(corpus)
        loop2.load(save_dir)

        # Encoder weights should match
        w1 = loop._encoder.embedding.weight.data
        w2 = loop2._encoder.embedding.weight.data
        assert torch.allclose(w1, w2, atol=1e-6)

    def test_encode_text_keep_grad(self) -> None:
        """keep_grad=True should return a tensor with grad_fn."""
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            train_encoder=True,
        )
        loop.train_tokenizer(["hello world", "test"])

        vec_grad = loop.encode_text("hello world", keep_grad=True)
        assert vec_grad.grad_fn is not None, "keep_grad=True should preserve grad"

        vec_no_grad = loop.encode_text("hello world", keep_grad=False)
        assert vec_no_grad.grad_fn is None, "keep_grad=False should detach"
