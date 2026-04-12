"""SOMA meta-cognitive subsystem: curiosity, homeostasis, development schedule."""

from soma.metacognition.curiosity import CuriosityModule
from soma.metacognition.development import CriticalPeriod, DevelopmentSchedule
from soma.metacognition.homeostasis import HomeostaticRegulator

__all__ = [
    "CriticalPeriod",
    "CuriosityModule",
    "DevelopmentSchedule",
    "HomeostaticRegulator",
]
