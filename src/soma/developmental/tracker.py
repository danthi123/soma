"""Track developmental metrics over time."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class DevelopmentTracker:
    """Records SOMA development metrics at each step and provides summary statistics."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = defaultdict(list)
        self._steps: list[int] = []

    @property
    def step_count(self) -> int:
        return len(self._steps)

    def record(
        self,
        step: int,
        prediction_error: float,
        novelty: float,
        num_nodes: int,
        num_edges: int,
        wm_occupancy: int,
        episodic_count: int,
        **extra: Any,
    ) -> None:
        """Record metrics for a single step."""
        self._steps.append(step)
        self.history["prediction_error"].append(float(prediction_error))
        self.history["novelty"].append(float(novelty))
        self.history["num_nodes"].append(float(num_nodes))
        self.history["num_edges"].append(float(num_edges))
        self.history["wm_occupancy"].append(float(wm_occupancy))
        self.history["episodic_count"].append(float(episodic_count))
        for key, value in extra.items():
            self.history[key].append(float(value))

    def summary(self, last_n: int | None = None) -> dict[str, dict[str, float]]:
        """Compute summary statistics for each metric.

        Args:
            last_n: If provided, only use the last N values for each metric.

        Returns:
            Dict mapping metric name to dict with mean, min, max, last, count.
        """
        result: dict[str, dict[str, float]] = {}
        for key, values in self.history.items():
            subset = values[-last_n:] if last_n is not None else values
            if not subset:
                continue
            result[key] = {
                "mean": sum(subset) / len(subset),
                "min": min(subset),
                "max": max(subset),
                "last": subset[-1],
                "count": float(len(subset)),
            }
        return result

    def save(self, path: str | Path) -> None:
        """Write tracker state to a JSON file."""
        data = {
            "steps": self._steps,
            "history": dict(self.history),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DevelopmentTracker:
        """Reconstruct a tracker from a JSON file."""
        data = json.loads(Path(path).read_text())
        tracker = cls()
        tracker._steps = data["steps"]
        for key, values in data["history"].items():
            tracker.history[key] = values
        return tracker
