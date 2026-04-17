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
        # Win counts per node — used to penalize dominant nodes
        self._win_counts: dict[str, int] = {}
        # Text store: maps step → original text for verbalization
        self.text_store: dict[int, str] = {}
        # Activation fingerprints: maps step → {node_id: magnitude}
        # Used for graph-driven retrieval. Dict format is robust
        # to neurogenesis (new nodes get new keys).
        self._activation_store: dict[int, dict[str, float]] = {}

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
    ) -> dict[str, float]:
        """Build a sparse fingerprint via lateral inhibition.

        Returns a dict mapping node_id → activation magnitude for the
        top ``inhibition_ratio`` fraction of nodes.  Suppressed nodes
        are omitted (implicitly zero).

        Using a dict (not a fixed-size tensor) makes fingerprints
        robust to neurogenesis — new nodes get new keys, old
        fingerprints just don't have those keys.
        """
        nodes = self.soma.graph.all_nodes()

        # Collect magnitudes
        node_mags: list[tuple[str, float]] = []
        for node in nodes:
            if node.last_activation is not None:
                mag = node.last_activation.norm().item()
            else:
                mag = 0.0
            node_mags.append((node.id, mag))

        if not node_mags:
            return {}

        # Lateral inhibition: keep only top-K by magnitude
        k = max(1, int(len(node_mags) * inhibition_ratio))
        threshold = sorted([m for _, m in node_mags], reverse=True)[
            min(k - 1, len(node_mags) - 1)
        ]

        return {
            nid: mag for nid, mag in node_mags if mag >= threshold
        }

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

    def _competitive_learning(
        self,
        input_tensor: torch.Tensor,
        lr: float = 0.001,
        margin: float = 1.2,
    ) -> None:
        """Competitive learning: winner node adapts toward the input.

        After lateral inhibition, the most active non-boundary node
        is the "winner" — but only if it's at least ``margin`` times
        more active than the runner-up.  This prevents a single node
        from claiming all inputs.

        The winner's first-layer weights are nudged toward the input,
        making it more responsive to similar inputs in the future.
        Suppressed and losing nodes don't learn, so they remain
        available to specialize for other inputs.
        """
        from soma.core.node import NodeType

        nodes = self.soma.graph.all_nodes()
        eligible = [
            n for n in nodes
            if n.node_type not in (NodeType.SENSOR, NodeType.OUTPUT)
            and n.last_activation is not None
        ]
        if len(eligible) < 2:
            return

        # Score by how much this activation DEVIATES from the node's
        # running average — not raw magnitude. A node that fires
        # equally for everything has low surprise; a node that fires
        # unusually strongly for this input is genuinely selective.
        scored = []
        for n in eligible:
            mag = n.last_activation.norm().item()
            avg = n.activation_ema  # running average magnitude
            surprise = mag - avg if avg > 0 else mag
            scored.append((surprise, n))

        scored.sort(key=lambda t: -t[0])
        winner = scored[0][1]

        # Only update if the winner is genuinely surprised (above avg)
        if scored[0][0] <= 0:
            return

        # Adapt winner's first-layer weights toward the input
        with torch.no_grad():
            w = winner.linear1.weight  # (hidden_dim, input_dim)
            inp = input_tensor.detach().to(w.device)

            # Resize input to match weight's input_dim
            if inp.shape[0] != w.shape[1]:
                if inp.shape[0] > w.shape[1]:
                    inp = inp[: w.shape[1]]
                else:
                    padded = torch.zeros(w.shape[1], device=w.device)
                    padded[: inp.shape[0]] = inp
                    inp = padded

            # SOM update: w_new = w + lr * (input - w)
            delta = inp.unsqueeze(0) - w
            w.add_(delta, alpha=lr)

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

        # Similarity between dict fingerprints: dot product of magnitudes
        # over shared node IDs, normalized by vector norms.
        scored: list[tuple[float, int]] = []
        for step, stored_fp in self._activation_store.items():
            sim = self._fingerprint_similarity(query_act, stored_fp)
            scored.append((sim, step))

        scored.sort(key=lambda t: -t[0])

        results: list[tuple[int, str, float]] = []
        for sim, step in scored[:top_k]:
            text = self.text_store.get(step, "")
            if text:
                results.append((step, text, sim))
        return results

    @staticmethod
    def _fingerprint_similarity(
        a: dict[str, float], b: dict[str, float],
    ) -> float:
        """Cosine similarity between two sparse dict fingerprints."""
        shared = set(a.keys()) & set(b.keys())
        if not shared:
            return 0.0
        dot = sum(a[k] * b[k] for k in shared)
        norm_a = sum(v * v for v in a.values()) ** 0.5
        norm_b = sum(v * v for v in b.values()) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

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

        # NOTE: competitive learning is implemented but disabled —
        # with fully-connected initialization all nodes receive the
        # same signal, so one node always dominates. Needs sparse
        # initial connectivity to work. See _competitive_learning().

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
