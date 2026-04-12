"""Working memory: fixed-capacity attention-addressed buffer with decay.

Whitepaper Section 4.1. Analogous to prefrontal-cortex short-term storage.

- ``slots``: ``(num_slots, wm_dim)`` — content tensor.
- ``usage``: ``(num_slots,)`` — how "full" each slot is, in [0, 1].
- ``age``: ``(num_slots,)`` — steps since last write.
- ``decay_rate``: multiplicative decay applied to ``usage`` each step.

Reads use soft attention (dot-product + softmax). Writes are gated by a
learnable predicate and go to the least-used slot.

All three state tensors are registered buffers so they round-trip through
``state_dict``; they are not learnable. Only ``query_proj``, ``write_gate``,
and ``erase_gate`` are learnable parameters.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from soma.core.config import SOMAConfig


class WorkingMemory(nn.Module):
    """Fixed-capacity slot memory with attention read and gated write."""

    def __init__(
        self,
        num_slots: int = 32,
        wm_dim: int = 128,
        decay_rate: float = 0.95,
        *,
        fade_threshold: float = 0.1,
        fade_factor: float = 0.1,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if wm_dim <= 0:
            raise ValueError(f"wm_dim must be positive, got {wm_dim}")
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be in (0, 1], got {decay_rate}")
        if not 0.0 <= fade_threshold <= 1.0:
            raise ValueError(f"fade_threshold must be in [0, 1], got {fade_threshold}")
        if not 0.0 <= fade_factor <= 1.0:
            raise ValueError(f"fade_factor must be in [0, 1], got {fade_factor}")

        self.num_slots = num_slots
        self.wm_dim = wm_dim
        self.decay_rate = decay_rate
        self.fade_threshold = fade_threshold
        self.fade_factor = fade_factor

        # Non-learnable state — registered so it round-trips through
        # state_dict.
        self.register_buffer("slots", torch.zeros(num_slots, wm_dim))
        self.register_buffer("usage", torch.zeros(num_slots))
        self.register_buffer("age", torch.zeros(num_slots, dtype=torch.long))

        # Learnable parameters.
        self.query_proj = nn.Linear(wm_dim, wm_dim)
        self.write_gate = nn.Linear(wm_dim * 2, 1)
        self.erase_gate = nn.Linear(wm_dim, num_slots)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: SOMAConfig,
        *,
        device: torch.device | str | None = None,
    ) -> WorkingMemory:
        """Build a WorkingMemory from ``SOMAConfig``."""
        return cls(
            num_slots=config.wm_slots,
            wm_dim=config.wm_dim,
            decay_rate=config.wm_decay_rate,
            device=device,
        )

    # ------------------------------------------------------------------
    # Read / Write / Step
    # ------------------------------------------------------------------
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Attention-based read. Returns a ``(wm_dim,)`` combined vector.

        Query is projected, then a scaled dot-product attention over slots
        yields weights, which combine the slot contents linearly.
        """
        self._check_query_shape(query)
        q = self.query_proj(query)
        # scores: (num_slots,)
        scores = torch.matmul(self._slots(), q) / math.sqrt(self.wm_dim)
        attention = F.softmax(scores, dim=0)
        # Weighted combination.
        return (attention.unsqueeze(-1) * self._slots()).sum(dim=0)

    def write(self, content: torch.Tensor, context: torch.Tensor) -> bool:
        """Gated write to the least-used slot. Returns True if a write occurred.

        The gate looks at ``[content, context]`` and produces a probability;
        writes happen only when the probability exceeds 0.5 (deterministic
        threshold — the whitepaper uses the same rule).
        """
        self._check_query_shape(content)
        self._check_query_shape(context)
        gate_input = torch.cat([content, context], dim=-1)
        write_prob = torch.sigmoid(self.write_gate(gate_input)).item()
        if write_prob <= 0.5:
            return False
        with torch.no_grad():
            slot_idx = int(torch.argmin(self._usage()).item())
            self._slots()[slot_idx] = content.detach()
            self._usage()[slot_idx] = 1.0
            self._age()[slot_idx] = 0
        return True

    def step(self) -> None:
        """Apply one timestep of decay to every slot.

        - ``usage`` multiplied by ``decay_rate``.
        - ``age`` incremented.
        - Slots whose usage falls below ``fade_threshold`` have their
          content multiplied by ``(1 - fade_factor)`` to erase them
          gradually.
        """
        with torch.no_grad():
            self._usage().mul_(self.decay_rate)
            self._age().add_(1)
            fade_mask = (self._usage() < self.fade_threshold).to(self._slots().dtype)
            scale = 1.0 - fade_mask.unsqueeze(-1) * self.fade_factor
            self._slots().mul_(scale)

    def clear(self) -> None:
        """Reset all slots (useful for tests and between sessions)."""
        with torch.no_grad():
            self._slots().zero_()
            self._usage().zero_()
            self._age().zero_()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def occupancy(self) -> float:
        """Fraction of slots currently considered "in use" (usage >= threshold)."""
        used = (self._usage() >= self.fade_threshold).sum().item()
        return float(used) / self.num_slots

    # ------------------------------------------------------------------
    # Internal helpers — typed accessors keep mypy happy around buffers.
    # ------------------------------------------------------------------
    def _slots(self) -> torch.Tensor:
        slots = self.slots
        assert isinstance(slots, torch.Tensor)
        return slots

    def _usage(self) -> torch.Tensor:
        usage = self.usage
        assert isinstance(usage, torch.Tensor)
        return usage

    def _age(self) -> torch.Tensor:
        age = self.age
        assert isinstance(age, torch.Tensor)
        return age

    def _check_query_shape(self, tensor: torch.Tensor) -> None:
        if tensor.shape != (self.wm_dim,):
            raise ValueError(
                f"WorkingMemory expects shape ({self.wm_dim},), got {tuple(tensor.shape)}"
            )
