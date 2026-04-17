"""Predictive processing loop for developmental SOMA.

Core idea: SOMA predicts the next input's activation pattern.
Prediction error drives Hebbian learning, neurogenesis, pruning,
and curiosity — unifying all brain-inspired mechanisms under one
signal (Free Energy Principle).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn

from soma.core.config import SOMAConfig
from soma.system import SOMA


class PredictiveSOMA(nn.Module):
    """Wraps SOMA with a next-activation prediction loop.

    Each call to :meth:`process_input` feeds the input through SOMA,
    then compares the resulting activation to the prediction made on the
    *previous* step.  The MSE between predicted and actual activations
    is stored as ``prediction_error`` and appended to ``error_history``.
    """

    def __init__(
        self,
        config: SOMAConfig,
        *,
        device: torch.device | None = None,
        error_history_size: int = 1000,
    ) -> None:
        super().__init__()
        self.soma = SOMA(config, device=device)
        self.config = config
        self.device = device or torch.device("cpu")
        self.prediction_head = nn.Linear(
            config.sensor_output_dim, config.sensor_output_dim
        ).to(self.device)
        self._last_prediction: torch.Tensor | None = None
        self.prediction_error: float = 0.0
        self.error_history: deque[float] = deque(maxlen=error_history_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_activation_summary(self, step_result: dict[str, Any]) -> torch.Tensor:
        """Extract a fixed-size activation vector from a SOMA step result."""
        outputs: dict[str, torch.Tensor] | None = step_result.get("outputs")
        dim = self.config.sensor_output_dim

        if outputs:
            # Take the first modality's output tensor.
            tensor = next(iter(outputs.values()))
            flat = tensor.detach().reshape(-1)
            if flat.shape[0] >= dim:
                summary = flat[:dim]
            else:
                summary = torch.zeros(dim, device=self.device)
                summary[: flat.shape[0]] = flat
            return summary.to(self.device)

        return torch.zeros(dim, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_input(
        self,
        input_tensor: torch.Tensor,
        *,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Run one prediction-loop step.

        Parameters
        ----------
        input_tensor:
            Raw input (e.g. token IDs or embedding vector).
        targets:
            Optional supervision targets forwarded to ``SOMA.step``.

        Returns
        -------
        dict with keys: activations, prediction_error, novelty, outputs,
        global_step, num_nodes, num_edges.
        """
        input_tensor = input_tensor.to(self.device)

        modality = self.config.input_modalities[0]
        inputs = {modality: input_tensor}

        step_result = self.soma.step(inputs, targets=targets)

        current_summary = self._get_activation_summary(step_result)

        if self._last_prediction is not None:
            self.prediction_error = float(
                torch.nn.functional.mse_loss(self._last_prediction, current_summary).item()
            )
        else:
            self.prediction_error = 0.0

        self.error_history.append(self.prediction_error)

        self._last_prediction = self.prediction_head(current_summary.detach())

        return {
            "activations": current_summary,
            "prediction_error": self.prediction_error,
            "novelty": step_result.get("curiosity", 0.0),
            "outputs": step_result.get("outputs"),
            "global_step": step_result.get("global_step"),
            "num_nodes": step_result.get("num_nodes"),
            "num_edges": step_result.get("num_edges"),
        }
