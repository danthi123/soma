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
        self._last_summary: torch.Tensor | None = None
        self._pred_optimizer = torch.optim.Adam(
            self.prediction_head.parameters(), lr=0.0003,
        )
        self.prediction_error: float = 0.0
        self.error_history: deque[float] = deque(maxlen=error_history_size)
        # Text store: maps step → original text for verbalization
        self.text_store: dict[int, str] = {}
        # Activation fingerprints: maps step → output activation vector
        # Used for graph-driven retrieval (cosine sim between query
        # activation and stored activations)
        self._activation_store: dict[int, torch.Tensor] = {}

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

    def _get_node_fingerprint(
        self, inhibition_ratio: float = 0.1,
    ) -> torch.Tensor:
        """Build a sparse fingerprint via lateral inhibition.

        Concatenates all node activations, but zeros out the weakest
        nodes — only the top ``inhibition_ratio`` fraction keep their
        activations.  This forces different inputs to produce different
        sparse patterns (biological lateral inhibition).

        Without inhibition all nodes fire similarly for all inputs
        (cross-topic cosine ~0.995).  At 30% keep ratio, similarity
        drops to ~0.93, enabling meaningful graph-driven retrieval.
        """
        nodes = sorted(self.soma.graph.all_nodes(), key=lambda n: n.id)

        # Collect activations and magnitudes
        acts: list[torch.Tensor] = []
        mags: list[float] = []
        for node in nodes:
            if node.last_activation is not None:
                acts.append(node.last_activation.detach())
                mags.append(node.last_activation.norm().item())
            else:
                acts.append(torch.zeros(node.output_dim, device=self.device))
                mags.append(0.0)

        if not acts:
            return torch.zeros(1, device=self.device)

        # Lateral inhibition: keep only top-K nodes by magnitude
        k = max(1, int(len(mags) * inhibition_ratio))
        if len(mags) > k:
            threshold = sorted(mags, reverse=True)[k - 1]
        else:
            threshold = 0.0

        parts: list[torch.Tensor] = []
        for mag, act in zip(mags, acts):
            if mag >= threshold:
                parts.append(act)
            else:
                parts.append(torch.zeros_like(act))

        return torch.cat(parts)

    def _apply_lateral_inhibition(self, keep_ratio: float = 0.1) -> None:
        """Suppress weakest nodes' last_activation in-place.

        After SOMA.step(), zero out the activations of the least active
        nodes.  This affects the stored fingerprint AND future
        synaptogenesis (suppressed nodes aren't counted as co-active).
        """
        from soma.core.node import NodeType

        nodes = self.soma.graph.all_nodes()
        # Don't inhibit SENSOR/OUTPUT boundary nodes
        eligible = [
            n for n in nodes
            if n.node_type not in (NodeType.SENSOR, NodeType.OUTPUT)
            and n.last_activation is not None
        ]
        if not eligible:
            return

        mags = [(n, n.last_activation.norm().item()) for n in eligible]
        k = max(1, int(len(mags) * keep_ratio))
        threshold = sorted([m for _, m in mags], reverse=True)[min(k - 1, len(mags) - 1)]

        for node, mag in mags:
            if mag < threshold:
                node.last_activation = torch.zeros_like(node.last_activation)

    def retrieve_by_graph(
        self,
        query_tensor: torch.Tensor,
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """Retrieve stored texts by graph activation similarity.

        Processes *query_tensor* through SOMA's graph (eval mode, no
        learning), compares the resulting activation to stored activation
        fingerprints, and returns the top-k most similar texts.

        Returns list of (step, text, similarity) tuples.
        """
        if not self._activation_store:
            return []

        query_tensor = query_tensor.to(self.device)
        modality = self.config.input_modalities[0]

        # Run query through graph WITHOUT learning
        self.soma.step(
            {modality: query_tensor}, eval_mode=True,
        )
        query_act = self._get_node_fingerprint()

        # Cosine similarity against stored activations
        scored: list[tuple[float, int]] = []
        for step, stored_act in self._activation_store.items():
            sim = float(torch.nn.functional.cosine_similarity(
                query_act.unsqueeze(0),
                stored_act.unsqueeze(0),
            ).item())
            scored.append((sim, step))

        scored.sort(key=lambda t: -t[0])

        results: list[tuple[int, str, float]] = []
        for sim, step in scored[:top_k]:
            text = self.text_store.get(step, "")
            if text:
                results.append((step, text, sim))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_input(
        self,
        input_tensor: torch.Tensor,
        *,
        targets: dict[str, torch.Tensor] | None = None,
        source_text: str | None = None,
    ) -> dict[str, Any]:
        """Run one prediction-loop step.

        Parameters
        ----------
        input_tensor:
            Raw input (e.g. token IDs or embedding vector).
        source_text:
            Original text (stored for retrieval/verbalization).
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

        # Self-supervised: use the input as its own target so SOMA
        # computes a loss, runs Hebbian learning, and triggers growth.
        if targets is None:
            out_modality = self.config.output_modalities[0]
            targets = {out_modality: input_tensor.detach()}

        step_result = self.soma.step(inputs, targets=targets)

        # Lateral inhibition: suppress weakest nodes' activations.
        # This drives specialization — synaptogenesis only wires
        # co-active (non-suppressed) nodes, so different inputs
        # strengthen different subgraphs over time.
        self._apply_lateral_inhibition()

        # Store original text for retrieval/verbalization
        step_num = step_result.get("global_step", len(self.text_store))
        if source_text is not None:
            self.text_store[step_num] = source_text

        current_summary = self._get_activation_summary(step_result)

        # Store activation fingerprint for graph-driven retrieval
        if source_text is not None:
            self._activation_store[step_num] = self._get_node_fingerprint()

        # Train prediction head: re-predict from last summary,
        # compare to current summary, backprop.
        if self._last_summary is not None:
            predicted = self.prediction_head(self._last_summary)
            pred_loss = torch.nn.functional.mse_loss(
                predicted, current_summary.detach(),
            )
            self.prediction_error = pred_loss.item()

            # Only update if error is still meaningful — prevent
            # over-convergence that collapses all fingerprints.
            # Biological analogy: synaptic plasticity decreases
            # for well-learned patterns but never reaches zero.
            if self.prediction_error > 1e-5:
                self._pred_optimizer.zero_grad()
                pred_loss.backward()
                self._pred_optimizer.step()
        else:
            self.prediction_error = 0.0

        self.error_history.append(self.prediction_error)

        # Feed prediction error into SOMA's recent errors so it can
        # trigger neurogenesis when error is persistently high.
        self.soma._recent_errors.append(self.prediction_error)

        # Save current summary for next step's prediction training
        self._last_summary = current_summary.detach().clone()
        # Cache the prediction for verbalization / inspection
        with torch.no_grad():
            self._last_prediction = self.prediction_head(
                current_summary.detach(),
            )

        return {
            "activations": current_summary,
            "prediction_error": self.prediction_error,
            "novelty": step_result.get("curiosity", 0.0),
            "outputs": step_result.get("outputs"),
            "global_step": step_result.get("global_step"),
            "num_nodes": step_result.get("num_nodes"),
            "num_edges": step_result.get("num_edges"),
            "loss": step_result.get("loss"),
        }
