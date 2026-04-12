"""Multimodal curriculum: modality weights that shift over development.

Whitepaper Section 8.3.

A ``MultimodalCurriculum`` holds a list of windows ``(start_step,
end_step, weights)`` and returns the current weight dict at a given
step. ``sample_modality`` picks one modality by weighted random draw.
``None`` for ``end_step`` means "open-ended", matching the whitepaper's
third stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CurriculumWindow:
    """One entry in a multimodal schedule."""

    start_step: int
    end_step: int | None  # None = open-ended
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError(
                f"end_step ({self.end_step}) must be > start_step ({self.start_step}) "
                f"or None (open-ended)"
            )
        if not self.weights:
            raise ValueError("CurriculumWindow requires at least one modality weight")
        for name, w in self.weights.items():
            if not name:
                raise ValueError("Modality names must be non-empty")
            if w < 0.0:
                raise ValueError(f"Weight for {name!r} must be non-negative, got {w}")
        if sum(self.weights.values()) <= 0.0:
            raise ValueError("Total weights across modalities must be > 0")

    def contains(self, step: int) -> bool:
        if step < self.start_step:
            return False
        return self.end_step is None or step < self.end_step


class MultimodalCurriculum:
    """Looks up the appropriate modality weights for a given step."""

    def __init__(self, windows: Iterable[CurriculumWindow]) -> None:
        self.windows: tuple[CurriculumWindow, ...] = tuple(windows)
        if not self.windows:
            raise ValueError("MultimodalCurriculum requires at least one window")
        # Require the schedule to start at step 0 so every possible step
        # is covered (up to the end of the last window or forever).
        if self.windows[0].start_step != 0:
            raise ValueError("First curriculum window must start at step 0")
        for prev, nxt in zip(self.windows, self.windows[1:], strict=False):
            if prev.end_step is None:
                raise ValueError("Open-ended window may only appear as the final entry")
            if nxt.start_step != prev.end_step:
                raise ValueError(
                    f"Schedule gap: window ending at {prev.end_step} followed by "
                    f"window starting at {nxt.start_step}"
                )

    @classmethod
    def default_text_image(cls) -> MultimodalCurriculum:
        """Whitepaper Section 8.3 default: text-heavy -> balanced -> interleaved."""
        return cls(
            [
                CurriculumWindow(0, 10_000, {"text": 0.8, "image": 0.2}),
                CurriculumWindow(10_000, 50_000, {"text": 0.5, "image": 0.5}),
                CurriculumWindow(50_000, None, {"text": 0.4, "image": 0.3, "interleaved": 0.3}),
            ]
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def current_weights(self, step: int) -> dict[str, float]:
        """Return the modality weights active at ``step``."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        for window in self.windows:
            if window.contains(step):
                return dict(window.weights)
        raise ValueError(f"step {step} falls outside the curriculum schedule")

    def current_window(self, step: int) -> CurriculumWindow:
        for window in self.windows:
            if window.contains(step):
                return window
        raise ValueError(f"step {step} falls outside the curriculum schedule")

    def sample_modality(
        self,
        step: int,
        *,
        rng: torch.Generator | None = None,
    ) -> str:
        """Pick one modality name weighted by the current-window weights."""
        weights = self.current_weights(step)
        names = list(weights.keys())
        probs = torch.tensor([weights[n] for n in names], dtype=torch.float32)
        probs = probs / probs.sum()
        idx = int(torch.multinomial(probs, num_samples=1, generator=rng).item())
        return names[idx]
