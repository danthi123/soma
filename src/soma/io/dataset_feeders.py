"""Dataset feeders: produce ``(inputs, target)`` training pairs on demand.

Whitepaper Section 8.3. The feeder is an iterable of
``(inputs_by_modality, target_tensor)`` pairs that the main training
loop consumes one at a time.

The ``TextDatasetFeeder`` splits each text sample into a context prefix
and a target suffix. Multiple sampling strategies are supported:

- ``"next_chunk"`` (whitepaper default): first half = context, next
  ``chunk_size`` tokens = target.
- ``"sliding_window"``: slide a fixed-size window over the token stream
  and predict the next ``target_size`` tokens at each position.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

import torch

from soma.io.text_encoder import TextEncoder

SplitStrategy = Literal["next_chunk", "sliding_window"]


@dataclass(frozen=True)
class Sample:
    """One training example produced by a dataset feeder."""

    inputs: dict[str, torch.Tensor]  # modality -> (T, embed_dim)
    target: torch.Tensor  # (T_target, embed_dim)


class TextDatasetFeeder:
    """Iterate over a text corpus emitting ``Sample`` training pairs.

    Parameters
    ----------
    encoder:
        ``TextEncoder`` used to tokenize and embed the text.
    texts:
        Any iterable of UTF-8 strings (list, generator, streaming
        dataset wrapper).
    chunk_size:
        Size of the context window (in tokens). Samples whose text is
        shorter than ``2 * chunk_size`` are skipped — we need enough
        tokens for both context and target.
    target_size:
        Number of tokens to predict per sample. Defaults to
        ``chunk_size``.
    strategy:
        ``"next_chunk"`` (one sample per text) or ``"sliding_window"``
        (many samples per text, stride = ``stride``).
    stride:
        Step between sliding windows; ignored for ``"next_chunk"``.
    cycle:
        When True (default), iteration restarts from the beginning once
        the source is exhausted. Useful with finite lists; pass False
        when the source is an infinite generator.
    """

    def __init__(
        self,
        encoder: TextEncoder,
        texts: Iterable[str],
        *,
        chunk_size: int = 32,
        target_size: int | None = None,
        strategy: SplitStrategy = "next_chunk",
        stride: int = 8,
        cycle: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        resolved_target = target_size if target_size is not None else chunk_size
        if resolved_target <= 0:
            raise ValueError(f"target_size must be positive, got {resolved_target}")
        if strategy not in ("next_chunk", "sliding_window"):
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.encoder = encoder
        self.chunk_size = chunk_size
        self.target_size = resolved_target
        self.strategy: SplitStrategy = strategy
        self.stride = stride
        self.cycle = cycle
        self._texts: list[str] = list(texts)  # materialize for cycling
        if not self._texts:
            raise ValueError("texts iterable is empty")

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Sample]:
        while True:
            for text in self._texts:
                yield from self._samples_from_text(text)
            if not self.cycle:
                return

    def generate_experience(self) -> Sample | None:
        """Produce one sample (whitepaper-compatible API).

        Returns ``None`` if the underlying iterator is exhausted and
        ``cycle=False``.
        """
        try:
            return next(iter(self))
        except StopIteration:
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _samples_from_text(self, text: str) -> Iterator[Sample]:
        ids = self.encoder.tokenize(text)
        if self.strategy == "next_chunk":
            sample = self._next_chunk_sample(ids)
            if sample is not None:
                yield sample
        else:  # sliding_window
            yield from self._sliding_window_samples(ids)

    def _next_chunk_sample(self, ids: list[int]) -> Sample | None:
        if len(ids) < self.chunk_size + self.target_size:
            return None
        context_ids = ids[: self.chunk_size]
        target_ids = ids[self.chunk_size : self.chunk_size + self.target_size]
        return self._build_sample(context_ids, target_ids)

    def _sliding_window_samples(self, ids: list[int]) -> Iterator[Sample]:
        for start in range(0, len(ids) - self.chunk_size - self.target_size + 1, self.stride):
            context_ids = ids[start : start + self.chunk_size]
            target_ids = ids[start + self.chunk_size : start + self.chunk_size + self.target_size]
            yield self._build_sample(context_ids, target_ids)

    def _build_sample(self, context_ids: list[int], target_ids: list[int]) -> Sample:
        context = self._embed_ids(context_ids)
        target = self._embed_ids(target_ids)
        return Sample(inputs={"text": context}, target=target)

    def _embed_ids(self, ids: list[int]) -> torch.Tensor:
        device = self.encoder.embedding.weight.device
        tokens = torch.tensor(ids, device=device, dtype=torch.long)
        positions = torch.arange(len(ids), device=device, dtype=torch.long)
        # Guard against positions exceeding the encoder's positional table.
        positions = positions.clamp_max(self.encoder.max_seq_len - 1)
        result: torch.Tensor = self.encoder.embedding(tokens) + self.encoder.position_encoding(
            positions
        )
        return result
