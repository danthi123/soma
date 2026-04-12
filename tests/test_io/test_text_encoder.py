"""Tests for ``soma.io.text_encoder``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.io.text_encoder import TextEncoder, load_tokenizer, train_bpe_tokenizer


@pytest.fixture(scope="module")
def tiny_corpus() -> list[str]:
    """Small training corpus — enough to produce a non-trivial vocab."""
    return [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "hello there, world!",
        "how are you today?",
        "python and pytorch are friends",
        "soma learns to speak by listening",
        "tokens become embeddings which become activations",
    ] * 4  # repeat so the trainer has enough samples


@pytest.fixture(scope="module")
def tiny_tokenizer(tiny_corpus: list[str]) -> object:
    return train_bpe_tokenizer(tiny_corpus, vocab_size=256)


class TestTrainBpeTokenizer:
    def test_basic_training(self, tiny_tokenizer: object) -> None:
        # The trained tokenizer should be able to round-trip text via
        # its encode/decode API.
        tok = tiny_tokenizer
        encoding = tok.encode("hello world")  # type: ignore[attr-defined]
        assert len(encoding.ids) > 0
        # decode should produce a string back (may not be byte-for-byte identical).
        decoded = tok.decode(encoding.ids)  # type: ignore[attr-defined]
        assert isinstance(decoded, str)

    def test_rejects_bad_vocab_size(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            train_bpe_tokenizer(["hi"], vocab_size=0)


class TestTextEncoder:
    def test_encode_returns_list_of_embeddings(self, tiny_tokenizer: object) -> None:
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16)
        embeddings = encoder.encode("hello world")
        assert len(embeddings) > 0
        for emb in embeddings:
            assert emb.shape == (16,)

    def test_encode_batch_stacks(self, tiny_tokenizer: object) -> None:
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16)
        stacked = encoder.encode_batch("hello world")
        assert stacked.ndim == 2
        assert stacked.shape[-1] == 16

    def test_empty_text_returns_empty(self, tiny_tokenizer: object) -> None:
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16)
        assert encoder.encode("") == []
        assert encoder.encode_batch("").shape == (0, 16)

    def test_position_changes_embedding(self, tiny_tokenizer: object) -> None:
        """The same token at different positions should have different embeddings."""
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16)
        # Force position encoding to be non-trivial.
        with torch.no_grad():
            encoder.position_encoding.weight.copy_(
                torch.arange(encoder.max_seq_len * 16, dtype=torch.float).reshape(
                    encoder.max_seq_len, 16
                )
                * 0.01
            )
        # Use a text where the same token repeats.
        embeddings = encoder.encode("a a a a")
        if len(embeddings) >= 2:
            assert not torch.allclose(embeddings[0], embeddings[1])

    def test_truncation_respects_max_seq_len(self, tiny_tokenizer: object) -> None:
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16, max_seq_len=4)
        ids = encoder.tokenize("the quick brown fox jumps over the lazy dog")
        assert len(ids) <= 4

    def test_rejects_bad_dims(self, tiny_tokenizer: object) -> None:
        with pytest.raises(ValueError, match="embed_dim"):
            TextEncoder(tiny_tokenizer, embed_dim=0)
        with pytest.raises(ValueError, match="max_seq_len"):
            TextEncoder(tiny_tokenizer, embed_dim=16, max_seq_len=0)

    def test_learnable_params(self, tiny_tokenizer: object) -> None:
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16)
        names = {n for n, _ in encoder.named_parameters()}
        assert "embedding.weight" in names
        assert "position_encoding.weight" in names

    def test_vocab_size_matches_tokenizer(self, tiny_tokenizer: object) -> None:
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16)
        assert encoder.vocab_size == tiny_tokenizer.get_vocab_size()  # type: ignore[attr-defined]
        assert encoder.embedding.num_embeddings == encoder.vocab_size


class TestTokenizerPersistence:
    def test_save_and_load(self, tiny_tokenizer: object, tmp_path: Path) -> None:
        encoder = TextEncoder(tiny_tokenizer, embed_dim=16)
        path = tmp_path / "tokenizer.json"
        encoder.save_tokenizer(path)
        assert path.exists()
        loaded = load_tokenizer(path)
        # Encoding the same text should yield the same ids.
        original_ids = encoder.tokenize("hello world")
        loaded_ids = loaded.encode("hello world").ids  # type: ignore[attr-defined]
        assert original_ids == list(loaded_ids)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.json"
        with pytest.raises(FileNotFoundError):
            load_tokenizer(missing)
