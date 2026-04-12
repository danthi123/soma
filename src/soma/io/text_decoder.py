"""Text output decoder: activation vector -> token id -> string.

Whitepaper Section 8.2.

Design notes:
- Input activation lives in ``embed_dim``-space (same as the encoder) so
  the decoder can share a tokenizer with the encoder.
- ``output_proj`` is a single learnable linear layer that scores every
  token in the vocabulary. The whitepaper uses argmax; we also expose
  ``logits`` and top-k/sample modes so future units (beam search,
  temperature sampling) have a clean attachment point.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F  # noqa: N812


class TextDecoder(nn.Module):
    """Project an activation vector onto the vocabulary and decode to text.

    Parameters
    ----------
    tokenizer:
        Shared tokenizer (typically the same instance the encoder uses).
    embed_dim:
        Size of the incoming activation vector.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        embed_dim: int = 64,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        self.tokenizer = tokenizer
        self.vocab_size: int = int(tokenizer.get_vocab_size())
        self.embed_dim = embed_dim
        self.output_proj = nn.Linear(embed_dim, self.vocab_size)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def logits(self, activation: torch.Tensor) -> torch.Tensor:
        """Return ``(vocab_size,)`` logits for a single activation vector."""
        self._check_shape(activation)
        result: torch.Tensor = self.output_proj(activation)
        return result

    def log_probs(self, activation: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.logits(activation), dim=-1)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode_token(self, activation: torch.Tensor) -> int:
        """Greedy argmax over the vocabulary."""
        token_id = int(self.logits(activation).argmax().item())
        return token_id

    def decode(self, activation: torch.Tensor) -> str:
        """Decode a single activation to a single-token string via argmax."""
        token_id = self.decode_token(activation)
        return str(self.tokenizer.decode([token_id]))

    def decode_sequence(self, activations: torch.Tensor) -> str:
        """Decode a ``(T, embed_dim)`` stack of activations to one string."""
        if activations.ndim != 2 or activations.shape[-1] != self.embed_dim:
            raise ValueError(
                f"decode_sequence expects shape (T, {self.embed_dim}), "
                f"got {tuple(activations.shape)}"
            )
        with torch.no_grad():
            all_logits = self.output_proj(activations)
            ids = [int(i) for i in all_logits.argmax(dim=-1).tolist()]
        return str(self.tokenizer.decode(ids))

    def sample_token(
        self,
        activation: torch.Tensor,
        temperature: float = 1.0,
        *,
        top_k: int | None = None,
        rng: torch.Generator | None = None,
    ) -> int:
        """Sample a token id from the temperature-scaled distribution.

        ``top_k`` restricts sampling to the ``top_k`` most-likely tokens;
        use ``None`` to sample from the full vocabulary. ``temperature``
        <= 0 falls back to greedy argmax.
        """
        if temperature <= 0.0:
            return self.decode_token(activation)
        scaled_logits = self.logits(activation) / temperature
        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be positive, got {top_k}")
            k = min(top_k, self.vocab_size)
            top_values, top_indices = torch.topk(scaled_logits, k=k)
            probs = F.softmax(top_values, dim=-1)
            choice = torch.multinomial(probs, num_samples=1, generator=rng)
            return int(top_indices[choice].item())
        probs = F.softmax(scaled_logits, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return int(choice.item())

    # ------------------------------------------------------------------
    # Save / Load helpers
    # ------------------------------------------------------------------
    def save_tokenizer(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _check_shape(self, activation: torch.Tensor) -> None:
        if activation.shape != (self.embed_dim,):
            raise ValueError(
                f"TextDecoder expects shape ({self.embed_dim},), got {tuple(activation.shape)}"
            )
