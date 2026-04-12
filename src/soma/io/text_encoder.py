"""Text input encoder: BPE tokenizer + learnable token & position embeddings.

Whitepaper Section 8.1.

Design notes:
- The tokenizer is a standalone ``tokenizers.Tokenizer`` (byte-level BPE)
  that lives outside the ``nn.Module``. The module holds the embedding
  tables (learnable parameters that participate in backprop).
- ``encode(text)`` returns a list of per-token embedding tensors, one per
  token, each of shape ``(embed_dim,)``. The list format matches the
  whitepaper pseudocode and composes cleanly with dataset feeders that
  stack into batched tensors.
- Tokenization is deterministic; randomness enters only during the
  module's parameter initialization.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch import nn

DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<UNK>", "<PAD>", "<BOS>", "<EOS>")


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 8192,
    *,
    special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer from an iterable of text strings.

    ``texts`` can be any iterable that yields UTF-8 strings (a list, a
    streaming dataset, etc.). The returned ``Tokenizer`` is ready for use
    (and can be saved via ``tokenizer.save(path)``).
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a previously-saved ``tokenizers.Tokenizer`` from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return Tokenizer.from_file(str(path))


class TextEncoder(nn.Module):
    """Turns UTF-8 text into per-token embedding vectors.

    Parameters
    ----------
    tokenizer:
        A trained ``tokenizers.Tokenizer`` instance (typically built by
        :func:`train_bpe_tokenizer`).
    embed_dim:
        Size of each embedding vector. Must match the sensor-node input
        dim it feeds into.
    max_seq_len:
        Inputs longer than this are truncated. Position embedding table
        is sized to ``max_seq_len``.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        embed_dim: int = 64,
        max_seq_len: int = 512,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.tokenizer = tokenizer
        self.vocab_size: int = int(tokenizer.get_vocab_size())
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Small-scale init so embeddings don't dominate sensor activations
        # at t=0 (whitepaper's homeostatic regulator would compensate, but
        # a sensible init helps early training).
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

        self.position_encoding = nn.Embedding(max_seq_len, embed_dim)
        nn.init.normal_(self.position_encoding.weight, mean=0.0, std=0.1)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        """Return the BPE token-id sequence for ``text``, truncated to max_len."""
        encoding = self.tokenizer.encode(text)
        ids = list(encoding.ids)
        return ids[: self.max_seq_len]

    def encode(self, text: str) -> list[torch.Tensor]:
        """Return a list of token+position embeddings for ``text``.

        Empty text yields an empty list.
        """
        ids = self.tokenize(text)
        if not ids:
            return []
        device = self.embedding.weight.device
        token_ids = torch.tensor(ids, device=device, dtype=torch.long)
        positions = torch.arange(len(ids), device=device, dtype=torch.long)
        combined = self.embedding(token_ids) + self.position_encoding(positions)
        return list(combined.unbind(dim=0))

    def encode_batch(self, text: str) -> torch.Tensor:
        """Return the stacked embedding tensor shape ``(T, embed_dim)``."""
        parts = self.encode(text)
        if not parts:
            device = self.embedding.weight.device
            return torch.empty((0, self.embed_dim), device=device)
        return torch.stack(parts, dim=0)

    # ------------------------------------------------------------------
    # Save / Load helpers
    # ------------------------------------------------------------------
    def save_tokenizer(self, path: str | Path) -> None:
        """Persist the tokenizer to ``path`` (JSON format)."""
        self.tokenizer.save(str(path))
