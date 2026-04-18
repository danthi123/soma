"""Open-ended learning environments for SOMA's developmental mechanisms.

These environments test neurogenesis, consolidation, structural plasticity,
and critical periods on tasks evaluated by adaptation speed and capacity,
not static retrieval accuracy. See
docs/plans/2026-04-18-open-ended-env-scope.md for rationale.
"""

from soma.environments.sequence_env import (
    RegimeSchedule,
    SequenceEnv,
    make_capacity_schedule,
    make_default_schedule,
)

__all__ = [
    "RegimeSchedule",
    "SequenceEnv",
    "make_capacity_schedule",
    "make_default_schedule",
]
