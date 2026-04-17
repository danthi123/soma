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

        # Random input diversifier: for each associator node, a frozen
        # random projection that transforms the input differently.
        # This gives each node a unique "view" of the input — the
        # software equivalent of different dendritic arbors in biology.
        from soma.core.node import NodeType

        self._input_projections: dict[str, torch.Tensor] = {}
        dim = config.sensor_output_dim
        gen = torch.Generator()
        if config.seed is not None:
            gen.manual_seed(config.seed + 13)
        for node in self.soma.graph.all_nodes():
            if node.node_type == NodeType.ASSOCIATOR:
                # Random orthogonal-ish projection matrix
                proj = torch.randn(dim, dim, generator=gen).to(self.device)
                # Normalize rows so projections preserve magnitude
                proj = proj / (proj.norm(dim=1, keepdim=True) + 1e-8)
                self._input_projections[node.id] = proj
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
        # Activation fingerprints: maps step → fixed-size tensor.
        # Hash-bucketed so fingerprint size is stable across neurogenesis.
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

        Projects each node's activation into a fixed-size hash bucket,
        so the fingerprint size doesn't change when neurogenesis adds
        nodes.  Suppressed nodes (below top ``inhibition_ratio``) are
        zeroed out, creating input-dependent sparse patterns.
        """
        nodes = self.soma.graph.all_nodes()
        fingerprint_dim = 256  # fixed size regardless of node count

        # Collect activations and magnitudes
        node_data: list[tuple[str, float, torch.Tensor | None]] = []
        for node in nodes:
            if node.last_activation is not None:
                mag = node.last_activation.norm().item()
                node_data.append((node.id, mag, node.last_activation.detach()))
            else:
                node_data.append((node.id, 0.0, None))

        if not node_data:
            return torch.zeros(fingerprint_dim, device=self.device)

        # Lateral inhibition: keep only top-K by magnitude
        k = max(1, int(len(node_data) * inhibition_ratio))
        threshold = sorted([m for _, m, _ in node_data], reverse=True)[
            min(k - 1, len(node_data) - 1)
        ]

        # Hash each active node's activation into fixed-size buckets
        fp = torch.zeros(fingerprint_dim, device=self.device)
        for nid, mag, act in node_data:
            if mag < threshold or act is None:
                continue
            # Hash node ID to a starting bucket
            bucket = hash(nid) % fingerprint_dim
            # Scatter the activation vector into the fingerprint
            act_flat = act.reshape(-1)
            for i in range(min(len(act_flat), fingerprint_dim)):
                idx = (bucket + i) % fingerprint_dim
                fp[idx] += act_flat[i]

        return fp

    def _diversify_activations(self, input_tensor: torch.Tensor) -> None:
        """Modulate each associator's activation by its unique input view.

        Multiplies each associator's ``last_activation`` element-wise by
        the dot product of the input with that node's random projection.
        Nodes whose projection aligns well with the input get amplified;
        others get dampened. This creates genuinely different activation
        patterns across nodes for different inputs.
        """
        inp = input_tensor.detach().to(self.device)
        for node_id, proj in self._input_projections.items():
            if node_id not in self.soma.graph.nodes:
                continue
            node = self.soma.graph.nodes[node_id]
            if node.last_activation is None:
                continue

            # Compute a scalar gain from the projection: how well
            # does this node's random "receptive field" match the input?
            alignment = torch.dot(torch.mv(proj, inp), inp)
            # Normalize to a gain factor centered on 1.0
            gain = 0.5 + 1.5 * torch.sigmoid(alignment / (inp.norm() ** 2 + 1e-8))
            # Scale this node's activation by the gain
            node.last_activation = node.last_activation * gain.item()

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

        # Cosine similarity against stored fingerprints
        scored: list[tuple[float, int]] = []
        for step, stored_fp in self._activation_store.items():
            sim = float(torch.nn.functional.cosine_similarity(
                query_act.unsqueeze(0),
                stored_fp.unsqueeze(0),
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
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save full developmental state to disk.

        Saves SOMA graph/memory state + PredictiveSOMA's text store,
        activation fingerprints, input projections, prediction head,
        and error history.
        """
        from pathlib import Path

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save SOMA core state
        self.soma.save_state(save_dir / "soma_state.pt")

        # Save PredictiveSOMA additions
        torch.save({
            "prediction_head": self.prediction_head.state_dict(),
            "prediction_error": self.prediction_error,
            "error_history": list(self.error_history),
            "text_store": self.text_store,
            "activation_store": {
                k: v.cpu() for k, v in self._activation_store.items()
            },
            "input_projections": {
                k: v.cpu() for k, v in self._input_projections.items()
            },
            "win_counts": self._win_counts,
            "last_summary": (
                self._last_summary.cpu()
                if self._last_summary is not None
                else None
            ),
            "last_prediction": (
                self._last_prediction.cpu()
                if self._last_prediction is not None
                else None
            ),
        }, save_dir / "predictive_state.pt")

    def load(self, path: str) -> None:
        """Load full developmental state from disk."""
        from pathlib import Path

        save_dir = Path(path)

        # Load SOMA core state
        self.soma.load_state(save_dir / "soma_state.pt")

        # Load PredictiveSOMA additions
        state = torch.load(
            save_dir / "predictive_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.prediction_head.load_state_dict(state["prediction_head"])
        self.prediction_head.to(self.device)
        self.prediction_error = state["prediction_error"]
        self.error_history = deque(
            state["error_history"], maxlen=self.error_history.maxlen,
        )
        self.text_store = state["text_store"]
        self._activation_store = {
            k: v.to(self.device) for k, v in state["activation_store"].items()
        }
        self._input_projections = {
            k: v.to(self.device)
            for k, v in state["input_projections"].items()
        }
        self._win_counts = state.get("win_counts", {})
        self._last_summary = (
            state["last_summary"].to(self.device)
            if state["last_summary"] is not None
            else None
        )
        self._last_prediction = (
            state["last_prediction"].to(self.device)
            if state["last_prediction"] is not None
            else None
        )

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

        # Input diversification: each associator's activation gets
        # modulated by its unique random projection of the input.
        # This creates different "views" per node — the software
        # equivalent of different dendritic receptive fields.
        self._diversify_activations(input_tensor)

        # Lateral inhibition: suppress weakest nodes' activations.
        # This drives specialization — synaptogenesis only wires
        # co-active (non-suppressed) nodes, so different inputs
        # strengthen different subgraphs over time.
        self._apply_lateral_inhibition()

        # Competitive learning: winner node adapts toward the input.
        # Requires sparse_init_connectivity < 1.0 so nodes receive
        # different input subsets and can genuinely specialize.
        self._competitive_learning(input_tensor)

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
