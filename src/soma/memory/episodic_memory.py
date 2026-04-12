"""Episodic memory: content-addressable fast store with replay sampling.

Whitepaper Section 4.2. Analogous to the hippocampus: one-shot encoding
of experiences, similarity-based retrieval, and prioritized replay for
consolidation.

Storage is a fixed-capacity circular buffer of ``(key, value)`` pairs
plus per-entry surprise score, timestamp, and replay count. A learnable
``key_encoder`` maps value-space experiences into key-space so retrieval
can use cosine similarity even when queries are in a different
modality.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from soma.core.config import SOMAConfig


class Retrieval(NamedTuple):
    """A single retrieved episode + its similarity score."""

    value: torch.Tensor
    similarity: float


class EpisodicMemory(nn.Module):
    """Fixed-capacity one-shot memory with cosine retrieval + priority replay.

    Parameters
    ----------
    capacity, key_dim, value_dim:
        Sizes of the storage buffers.
    age_sweet_spot, age_scale:
        Controls the Gaussian age bias in ``sample_for_replay``. Mid-age
        entries get the highest replay priority. Defaults match the
        whitepaper (1000, 2000).
    """

    def __init__(
        self,
        capacity: int = 10_000,
        key_dim: int = 128,
        value_dim: int = 256,
        *,
        age_sweet_spot: float = 1000.0,
        age_scale: float = 2000.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError(f"key_dim and value_dim must be positive, got {key_dim}, {value_dim}")
        if age_scale <= 0:
            raise ValueError(f"age_scale must be positive, got {age_scale}")

        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.age_sweet_spot = age_sweet_spot
        self.age_scale = age_scale

        # Non-learnable storage buffers.
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.register_buffer("surprise", torch.zeros(capacity))
        self.register_buffer("timestamps", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("valid", torch.zeros(capacity, dtype=torch.bool))
        self.register_buffer("replay_count", torch.zeros(capacity, dtype=torch.long))
        # write_head as a 0-d long buffer so it round-trips via state_dict.
        self.register_buffer("write_head", torch.zeros((), dtype=torch.long))

        # Learnable key encoder: value_dim -> key_dim.
        self.key_encoder = nn.Sequential(
            nn.Linear(value_dim, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

        if device is not None:
            self.to(device)

    @classmethod
    def from_config(
        cls,
        config: SOMAConfig,
        *,
        device: torch.device | str | None = None,
    ) -> EpisodicMemory:
        return cls(
            capacity=config.episodic_capacity,
            key_dim=config.key_dim,
            value_dim=config.value_dim,
            device=device,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(
        self,
        experience: torch.Tensor,
        prediction_error: float,
        current_step: int,
    ) -> int:
        """Store one experience. Returns the slot index it landed in.

        The circular write head overwrites the oldest entry once the
        buffer is full.
        """
        self._check_value_shape(experience)
        if current_step < 0:
            raise ValueError(f"current_step must be non-negative, got {current_step}")
        key = self.key_encoder(experience.detach())
        with torch.no_grad():
            idx = int(self._write_head_value())
            self._keys()[idx] = key.detach()
            self._values()[idx] = experience.detach()
            self._surprise()[idx] = float(prediction_error)
            self._timestamps()[idx] = int(current_step)
            self._valid()[idx] = True
            self._replay_count()[idx] = 0
            self._write_head_buffer().fill_((idx + 1) % self.capacity)
        return idx

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> list[Retrieval]:
        """Return up to ``top_k`` most-similar episodes by cosine similarity.

        The query must have shape ``(key_dim,)`` — callers should pass
        keys produced by ``self.key_encoder`` (or any equivalent key-space
        vector). Returns an empty list when the buffer is empty.
        """
        self._check_key_shape(query)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not bool(self._valid().any().item()):
            return []
        valid_mask = self._valid()
        valid_keys = self._keys()[valid_mask]
        valid_values = self._values()[valid_mask]
        sims = F.cosine_similarity(query.unsqueeze(0), valid_keys, dim=-1)
        k = min(top_k, sims.shape[0])
        top = torch.topk(sims, k=k)
        return [
            Retrieval(value=valid_values[i].detach().clone(), similarity=float(s.item()))
            for i, s in zip(top.indices, top.values, strict=True)
        ]

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def sample_for_replay(
        self,
        current_step: int,
        batch_size: int = 32,
        *,
        rng: torch.Generator | None = None,
    ) -> list[torch.Tensor]:
        """Sample a batch of experiences for consolidation replay.

        Prioritization: ``surprise * gaussian_age_weight``, where the age
        weight peaks around ``age_sweet_spot`` steps of age. Entries with
        zero priority are excluded.

        Updates ``replay_count`` in place for sampled entries so
        ``decay_old_entries`` can eventually invalidate exhausted memories.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if current_step < 0:
            raise ValueError(f"current_step must be non-negative, got {current_step}")

        valid = self._valid()
        if not bool(valid.any().item()):
            return []

        priority = self._compute_replay_priority(current_step)
        # Guard against an all-zero priority (e.g., all surprises are 0).
        total = float(priority.sum().item())
        if total <= 0.0:
            return []
        normalized = priority / (total + 1e-8)

        num_valid = int(valid.sum().item())
        sample_count = min(batch_size, num_valid)
        indices = torch.multinomial(normalized, sample_count, replacement=False, generator=rng)

        with torch.no_grad():
            self._replay_count()[indices] += 1

        return [self._values()[i].detach().clone() for i in indices.tolist()]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def decay_old_entries(
        self,
        current_step: int,
        *,
        max_age: int = 100_000,
        min_replays: int = 5,
    ) -> int:
        """Invalidate entries that are both old and sufficiently replayed.

        An entry is dropped when ``age > max_age`` AND ``replay_count >=
        min_replays``. Returns the number of entries invalidated.
        """
        if max_age <= 0:
            raise ValueError(f"max_age must be positive, got {max_age}")
        if min_replays < 0:
            raise ValueError(f"min_replays must be non-negative, got {min_replays}")
        with torch.no_grad():
            ages = current_step - self._timestamps()
            sufficient = (ages > max_age) & (self._replay_count() >= min_replays) & self._valid()
            count = int(sufficient.sum().item())
            self._valid()[sufficient] = False
        return count

    def clear(self) -> None:
        """Reset the store so every slot is marked invalid."""
        with torch.no_grad():
            self._valid().fill_(False)
            self._keys().zero_()
            self._values().zero_()
            self._surprise().zero_()
            self._timestamps().zero_()
            self._replay_count().zero_()
            self._write_head_buffer().zero_()

    @property
    def num_valid(self) -> int:
        return int(self._valid().sum().item())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_replay_priority(self, current_step: int) -> torch.Tensor:
        ages = (current_step - self._timestamps()).float()
        diff = (ages - self.age_sweet_spot) / self.age_scale
        age_weight = torch.exp(-(diff * diff))
        priority = self._surprise() * age_weight * self._valid().float()
        return priority

    def _check_value_shape(self, tensor: torch.Tensor) -> None:
        if tensor.shape != (self.value_dim,):
            raise ValueError(
                f"EpisodicMemory expects value shape ({self.value_dim},), got {tuple(tensor.shape)}"
            )

    def _check_key_shape(self, tensor: torch.Tensor) -> None:
        if tensor.shape != (self.key_dim,):
            raise ValueError(
                f"EpisodicMemory expects key shape ({self.key_dim},), got {tuple(tensor.shape)}"
            )

    def _keys(self) -> torch.Tensor:
        assert isinstance(self.keys, torch.Tensor)
        return self.keys

    def _values(self) -> torch.Tensor:
        assert isinstance(self.values, torch.Tensor)
        return self.values

    def _surprise(self) -> torch.Tensor:
        assert isinstance(self.surprise, torch.Tensor)
        return self.surprise

    def _timestamps(self) -> torch.Tensor:
        assert isinstance(self.timestamps, torch.Tensor)
        return self.timestamps

    def _valid(self) -> torch.Tensor:
        assert isinstance(self.valid, torch.Tensor)
        return self.valid

    def _replay_count(self) -> torch.Tensor:
        assert isinstance(self.replay_count, torch.Tensor)
        return self.replay_count

    def _write_head_value(self) -> int:
        return int(self._write_head_buffer().item())

    def _write_head_buffer(self) -> torch.Tensor:
        assert isinstance(self.write_head, torch.Tensor)
        return self.write_head
