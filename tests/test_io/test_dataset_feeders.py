"""Tests for ``soma.io.dataset_feeders.TextDatasetFeeder``."""

from __future__ import annotations

import itertools

import pytest

from soma.io.dataset_feeders import Sample, TextDatasetFeeder
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer


@pytest.fixture(scope="module")
def encoder() -> TextEncoder:
    corpus = [
        "hello world " * 20,
        "the quick brown fox jumps over the lazy dog " * 20,
        "soma learns to speak by listening " * 20,
        "tokens become embeddings which become activations " * 20,
    ]
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=128)
    return TextEncoder(tokenizer, embed_dim=16, max_seq_len=256)


class TestNextChunk:
    def test_produces_sample(self, encoder: TextEncoder) -> None:
        texts = ["the quick brown fox jumps over the lazy dog " * 10]
        feeder = TextDatasetFeeder(encoder, texts, chunk_size=4, target_size=4)
        sample = next(iter(feeder))
        assert isinstance(sample, Sample)
        assert "text" in sample.inputs
        assert sample.inputs["text"].shape == (4, encoder.embed_dim)
        assert sample.target.shape == (4, encoder.embed_dim)

    def test_skips_short_texts(self, encoder: TextEncoder) -> None:
        """Texts shorter than chunk + target should be skipped."""
        texts = ["hi"]
        feeder = TextDatasetFeeder(
            encoder,
            texts,
            chunk_size=32,
            target_size=32,
            cycle=False,
        )
        samples = list(feeder)
        assert samples == []

    def test_generate_experience(self, encoder: TextEncoder) -> None:
        texts = ["the quick brown fox jumps over the lazy dog " * 10]
        feeder = TextDatasetFeeder(encoder, texts, chunk_size=4, target_size=4)
        exp = feeder.generate_experience()
        assert exp is not None
        assert "text" in exp.inputs


class TestSlidingWindow:
    def test_yields_multiple_samples(self, encoder: TextEncoder) -> None:
        texts = ["the quick brown fox jumps over the lazy dog " * 10]
        feeder = TextDatasetFeeder(
            encoder,
            texts,
            chunk_size=4,
            target_size=4,
            strategy="sliding_window",
            stride=4,
            cycle=False,
        )
        samples = list(feeder)
        assert len(samples) > 1  # sliding should produce more than one sample

    def test_stride_controls_count(self, encoder: TextEncoder) -> None:
        texts = ["the quick brown fox jumps over the lazy dog " * 10]
        few = list(
            TextDatasetFeeder(
                encoder,
                texts,
                chunk_size=4,
                target_size=4,
                strategy="sliding_window",
                stride=8,
                cycle=False,
            )
        )
        many = list(
            TextDatasetFeeder(
                encoder,
                texts,
                chunk_size=4,
                target_size=4,
                strategy="sliding_window",
                stride=2,
                cycle=False,
            )
        )
        assert len(many) > len(few)


class TestCycle:
    def test_cycles_forever(self, encoder: TextEncoder) -> None:
        texts = ["the quick brown fox jumps over the lazy dog " * 10]
        feeder = TextDatasetFeeder(encoder, texts, chunk_size=4, target_size=4, cycle=True)
        # Take 10 samples — cycling means we always get results.
        taken = list(itertools.islice(feeder, 10))
        assert len(taken) == 10

    def test_no_cycle_stops(self, encoder: TextEncoder) -> None:
        texts = ["the quick brown fox jumps over the lazy dog " * 10]
        feeder = TextDatasetFeeder(
            encoder,
            texts,
            chunk_size=4,
            target_size=4,
            strategy="next_chunk",
            cycle=False,
        )
        taken = list(feeder)
        # next_chunk yields at most one sample per text.
        assert len(taken) <= len(texts)


class TestValidation:
    def test_rejects_empty_texts(self, encoder: TextEncoder) -> None:
        with pytest.raises(ValueError, match="empty"):
            TextDatasetFeeder(encoder, [])

    def test_rejects_bad_chunk_size(self, encoder: TextEncoder) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            TextDatasetFeeder(encoder, ["x"], chunk_size=0)

    def test_rejects_unknown_strategy(self, encoder: TextEncoder) -> None:
        with pytest.raises(ValueError, match="strategy"):
            TextDatasetFeeder(
                encoder,
                ["x"],
                chunk_size=1,
                target_size=1,
                strategy="nonsense",  # type: ignore[arg-type]
            )

    def test_rejects_bad_stride(self, encoder: TextEncoder) -> None:
        with pytest.raises(ValueError, match="stride"):
            TextDatasetFeeder(
                encoder,
                ["x"],
                chunk_size=1,
                target_size=1,
                strategy="sliding_window",
                stride=0,
            )

    def test_rejects_bad_target_size(self, encoder: TextEncoder) -> None:
        with pytest.raises(ValueError, match="target_size"):
            TextDatasetFeeder(encoder, ["x"], chunk_size=1, target_size=0)
