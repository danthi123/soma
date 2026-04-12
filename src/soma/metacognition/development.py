"""Developmental stages: critical periods that shape plasticity over time.

Whitepaper Section 5.5.

Each ``CriticalPeriod`` is a triangular envelope over (``start_step``,
``peak_step``, ``end_step``) that boosts plasticity / synaptogenesis
for a specific ``NodeType`` set. The envelope's peak value at
``peak_step`` is ``plasticity_multiplier`` (or
``synaptogenesis_multiplier``); it ramps linearly up on the rising edge
and linearly down on the falling edge.

``DevelopmentSchedule`` aggregates multiple overlapping periods and
returns the maximum multiplier across all applicable ones at a given
step and node type.
"""

from __future__ import annotations

from dataclasses import dataclass

from soma.core.node import NodeType


@dataclass(frozen=True)
class CriticalPeriod:
    """Triangular plasticity envelope over a window of global steps."""

    name: str
    start_step: int
    peak_step: int
    end_step: int
    affected_node_types: tuple[NodeType, ...]
    plasticity_multiplier: float
    synaptogenesis_multiplier: float

    def __post_init__(self) -> None:
        if not (self.start_step <= self.peak_step <= self.end_step):
            raise ValueError(
                f"CriticalPeriod {self.name!r}: require "
                f"start ({self.start_step}) <= peak ({self.peak_step}) "
                f"<= end ({self.end_step})"
            )
        if self.plasticity_multiplier < 1.0:
            raise ValueError(
                f"plasticity_multiplier must be >= 1.0, got {self.plasticity_multiplier}"
            )
        if self.synaptogenesis_multiplier < 1.0:
            raise ValueError(
                f"synaptogenesis_multiplier must be >= 1.0, got {self.synaptogenesis_multiplier}"
            )

    def progress(self, step: int) -> float:
        """Return the triangular envelope value at ``step``, in [0, 1]."""
        if step < self.start_step or step > self.end_step:
            return 0.0
        if step == self.peak_step:
            return 1.0
        if step < self.peak_step:
            rising = self.peak_step - self.start_step
            if rising == 0:
                return 1.0
            return (step - self.start_step) / rising
        falling = self.end_step - self.peak_step
        if falling == 0:
            return 1.0
        return 1.0 - (step - self.peak_step) / falling

    def applies_to(self, node_type: NodeType, step: int) -> bool:
        return node_type in self.affected_node_types and self.start_step <= step <= self.end_step


def _default_periods() -> tuple[CriticalPeriod, ...]:
    """Whitepaper Section 5.5 defaults."""
    return (
        CriticalPeriod(
            name="sensory_discrimination",
            start_step=0,
            peak_step=5_000,
            end_step=20_000,
            affected_node_types=(NodeType.SENSOR,),
            plasticity_multiplier=3.0,
            synaptogenesis_multiplier=5.0,
        ),
        CriticalPeriod(
            name="association_formation",
            start_step=5_000,
            peak_step=25_000,
            end_step=50_000,
            affected_node_types=(NodeType.ASSOCIATOR,),
            plasticity_multiplier=2.5,
            synaptogenesis_multiplier=3.0,
        ),
        CriticalPeriod(
            name="integration",
            start_step=20_000,
            peak_step=50_000,
            end_step=100_000,
            affected_node_types=(NodeType.INTEGRATOR,),
            plasticity_multiplier=2.0,
            synaptogenesis_multiplier=2.0,
        ),
        CriticalPeriod(
            name="output_refinement",
            start_step=30_000,
            peak_step=60_000,
            end_step=120_000,
            affected_node_types=(NodeType.OUTPUT,),
            plasticity_multiplier=2.0,
            synaptogenesis_multiplier=1.5,
        ),
    )


class DevelopmentSchedule:
    """Aggregates critical periods; returns max multiplier for a step/type."""

    def __init__(self, periods: tuple[CriticalPeriod, ...] | None = None) -> None:
        self.periods: tuple[CriticalPeriod, ...] = (
            periods if periods is not None else _default_periods()
        )

    def get_plasticity_multiplier(self, step: int, node_type: NodeType) -> float:
        return self._max_multiplier(step, node_type, synaptogenesis=False)

    def get_synaptogenesis_multiplier(self, step: int, node_type: NodeType) -> float:
        return self._max_multiplier(step, node_type, synaptogenesis=True)

    def active_periods(self, step: int) -> list[CriticalPeriod]:
        """Return all periods that overlap ``step`` for any node type."""
        return [p for p in self.periods if p.start_step <= step <= p.end_step]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _max_multiplier(
        self,
        step: int,
        node_type: NodeType,
        *,
        synaptogenesis: bool,
    ) -> float:
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        multiplier = 1.0
        for period in self.periods:
            if not period.applies_to(node_type, step):
                continue
            peak = (
                period.synaptogenesis_multiplier if synaptogenesis else period.plasticity_multiplier
            )
            boost = 1.0 + (peak - 1.0) * period.progress(step)
            if boost > multiplier:
                multiplier = boost
        return multiplier
