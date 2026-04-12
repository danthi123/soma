"""Tests for ``soma.io.text_decoder.TextDecoder``."""

from __future__ import annotations

import pytest
import torch

from soma.io.text_decoder import TextDecoder
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer


@pytest.fixture(scope="module")
def tiny_tokenizer() -> object:
    corpus = [
        "hello world " * 3,
        "the quick brown fox jumps over the lazy dog " * 3,
        "soma learns to speak by listening " * 3,
    ]
    return train_bpe_tokenizer(corpus, vocab_size=128)


class TestBasic:
    def test_decode_token_returns_int(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        token = dec.decode_token(torch.randn(16))
        assert isinstance(token, int)
        assert 0 <= token < dec.vocab_size

    def test_decode_returns_string(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        out = dec.decode(torch.randn(16))
        assert isinstance(out, str)

    def test_logits_shape(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        logits = dec.logits(torch.randn(16))
        assert logits.shape == (dec.vocab_size,)

    def test_log_probs_sum_to_one(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        lp = dec.log_probs(torch.randn(16))
        assert torch.isclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-5)

    def test_decode_sequence(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        seq = torch.randn(5, 16)
        text = dec.decode_sequence(seq)
        assert isinstance(text, str)

    def test_decode_sequence_rejects_bad_shape(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        with pytest.raises(ValueError, match="shape"):
            dec.decode_sequence(torch.randn(16))  # 1D instead of 2D

    def test_rejects_shape_mismatch(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        with pytest.raises(ValueError, match="shape"):
            dec.decode_token(torch.randn(7))

    def test_rejects_bad_embed_dim(self, tiny_tokenizer: object) -> None:
        with pytest.raises(ValueError, match="embed_dim"):
            TextDecoder(tiny_tokenizer, embed_dim=0)


class TestSampling:
    def test_temperature_zero_is_greedy(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        act = torch.randn(16)
        greedy = dec.decode_token(act)
        sampled = dec.sample_token(act, temperature=0.0)
        assert sampled == greedy

    def test_top_k_constrains_output(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        act = torch.randn(16)
        top_k = 5
        logits = dec.logits(act)
        top_ids = set(torch.topk(logits, k=top_k).indices.tolist())
        rng = torch.Generator().manual_seed(0)
        # Sample many times; every result must be in the top_k set.
        for _ in range(50):
            token = dec.sample_token(act, temperature=1.0, top_k=top_k, rng=rng)
            assert token in top_ids

    def test_rejects_bad_top_k(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        with pytest.raises(ValueError, match="top_k"):
            dec.sample_token(torch.randn(16), top_k=0)


class TestTieWeights:
    def test_tie_weights_shares_tensor(self, tiny_tokenizer: object) -> None:
        enc = TextEncoder(tiny_tokenizer, embed_dim=16)
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        dec.tie_weights(enc)
        assert dec.output_proj.weight is enc.embedding.weight

    def test_tie_weights_gradient_flows_both_ways(self, tiny_tokenizer: object) -> None:
        """Updating decoder weight should be visible on encoder embedding."""
        enc = TextEncoder(tiny_tokenizer, embed_dim=16)
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        dec.tie_weights(enc)
        with torch.no_grad():
            dec.output_proj.weight[0].fill_(7.0)
        assert torch.all(enc.embedding.weight[0] == 7.0)

    def test_tie_weights_reproduces_token_from_embedding(self, tiny_tokenizer: object) -> None:
        """A token whose embedding we point at, then decode, should come back."""
        enc = TextEncoder(tiny_tokenizer, embed_dim=16)
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        dec.tie_weights(enc)
        token_id = 7
        # Pull out that token's embedding as the activation.
        activation = enc.embedding.weight[token_id].detach().clone()
        # With a tied decoder + zero bias, the token whose embedding matches
        # should be the argmax candidate.
        with torch.no_grad():
            dec.output_proj.bias.zero_()
        assert dec.decode_token(activation) == token_id

    def test_tie_weights_rejects_missing_embedding(self, tiny_tokenizer: object) -> None:
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)

        class NoEmbedding:
            pass

        with pytest.raises(ValueError, match="embedding"):
            dec.tie_weights(NoEmbedding())  # type: ignore[arg-type]

    def test_tie_weights_rejects_shape_mismatch(self, tiny_tokenizer: object) -> None:
        enc = TextEncoder(tiny_tokenizer, embed_dim=32)
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        with pytest.raises(ValueError, match="shape mismatch"):
            dec.tie_weights(enc)


class TestCoRoundTrip:
    def test_encoder_decoder_share_vocab(self, tiny_tokenizer: object) -> None:
        enc = TextEncoder(tiny_tokenizer, embed_dim=16)
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        assert enc.vocab_size == dec.vocab_size

    def test_with_trained_weights_can_reproduce_token(self, tiny_tokenizer: object) -> None:
        """If we fake-train the decoder, specific activations should map to
        specific tokens. We set up a one-to-one mapping: embedding -> logits."""
        dec = TextDecoder(tiny_tokenizer, embed_dim=16)
        token_id = 5  # arbitrary
        with torch.no_grad():
            # Make the logit for ``token_id`` huge when the activation is
            # all-ones; other logits will be near-zero.
            dec.output_proj.weight.zero_()
            dec.output_proj.bias.zero_()
            dec.output_proj.weight[token_id].fill_(1.0)
        activation = torch.ones(16)
        assert dec.decode_token(activation) == token_id
