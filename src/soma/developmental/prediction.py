"""Predictive processing loop for developmental SOMA.

Core idea: SOMA predicts the next input's activation pattern.
Prediction error drives Hebbian learning, neurogenesis, pruning,
and curiosity — unifying all brain-inspired mechanisms under one
signal (Free Energy Principle).
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable
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

        # Direction 2: projections can be either frozen Tensors or
        # learnable nn.Parameters. Type is chosen once at init and
        # preserved across neurogenesis-triggered _ensure_projection calls.
        self._input_projections: dict[str, torch.Tensor] = {}
        dim = config.sensor_output_dim
        gen = torch.Generator()
        if config.seed is not None:
            gen.manual_seed(config.seed + 13)
        for node in self.soma.graph.all_nodes():
            if node.node_type == NodeType.ASSOCIATOR:
                proj = self._build_projection(gen=gen)
                self._input_projections[node.id] = proj
        # Direction 4b: position_projector maps flat W_i -> position_dim
        # via a fixed random linear projection. Used only when position
        # distillation is active (llm_spatial target + learnable positions).
        # Registered as a non-Parameter tensor (frozen). Johnson-Lindenstrauss
        # preserves pairwise distances, which is what the locality filter reads.
        self._position_projector: torch.Tensor | None = None
        if (
            config.position_mode == "learnable"
            and config.projection_distillation_target == "llm_spatial"
        ):
            proj_gen = torch.Generator()
            if config.seed is not None:
                proj_gen.manual_seed(config.seed + 31)
            self._position_projector = torch.randn(
                config.sensor_output_dim ** 2,
                config.position_dim,
                generator=proj_gen,
            ).to(self.device)
        self._last_prediction: torch.Tensor | None = None
        self._last_summary: torch.Tensor | None = None
        # When projections are learnable, include them in the prediction
        # optimizer so the prediction loss trains them jointly with
        # prediction_head. Otherwise only prediction_head is optimized.
        opt_params: list[torch.nn.Parameter] = list(
            self.prediction_head.parameters()
        )
        if config.projection_mode == "learnable":
            opt_params.extend(
                p for p in self._input_projections.values()
                if isinstance(p, torch.nn.Parameter)
            )
            self._pred_optimizer = torch.optim.Adam(
                [
                    {"params": list(self.prediction_head.parameters()), "lr": 0.0003},
                    {
                        "params": [
                            p for p in self._input_projections.values()
                            if isinstance(p, torch.nn.Parameter)
                        ],
                        "lr": config.projection_lr,
                    },
                ]
            )
        else:
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
        # Token cache: maps step → set of BPE token IDs for token-overlap
        # retrieval. Populated during process_input when a tokenizer is
        # available (set by the caller via set_tokenizer).
        self._token_cache: dict[int, set[int]] = {}
        self._tokenizer_fn: Callable[[str], list[int]] | None = None
        # Node activation index: for each node, which memory steps
        # had this node among the top-K active. This enables
        # topology-based retrieval — finding memories that share
        # active nodes with a query, which captures learned structural
        # associations that fingerprint comparison misses.
        self._node_memory_index: dict[str, set[int]] = {}
        # Cache SHA256 digests for fingerprint hash positions
        self._node_hash_cache: dict[str, bytes] = {}
        # Precomputed index arrays for fingerprint: node_id -> (src_indices, tgt_indices)
        self._fp_index_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # Direction 4a: optional LLM teacher for projection distillation.
        # None means no distillation regardless of config. Attached via
        # attach_teacher() after construction so swapping teachers is
        # decoupled from SOMA construction.
        self._teacher: Any = None

    def set_tokenizer(self, tokenize_fn: Callable[[str], list[int]]) -> None:
        """Register a tokenization function for token-overlap retrieval.

        ``tokenize_fn(text) -> list[int]`` should return BPE token IDs.
        """
        self._tokenizer_fn = tokenize_fn

    def attach_teacher(self, teacher: Any) -> None:
        """Attach an LLM teacher for Direction 4a projection distillation.

        Expects an object with ``embed(text) -> torch.Tensor``. The
        teacher is only consulted when
        ``config.projection_distillation_target == "llm_embedding"``
        and a ``source_text`` is passed to :meth:`process_input`.
        No teacher attached means no distillation regardless of config.
        """
        self._teacher = teacher

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

    def _get_node_fingerprint(self) -> torch.Tensor:
        """Build a fixed-size fingerprint from node activations.

        Uses ALL active nodes (lateral inhibition has already zeroed
        suppressed nodes). Each node's activation is projected to a
        short vector via a deterministic hash-based projection, then
        scattered at the node's hash position. This preserves
        directional information while keeping per-node footprint small
        enough to avoid collision saturation.
        """
        from soma.core.node import NodeType

        nodes = self.soma.graph.all_nodes()
        fingerprint_dim = 256
        values_per_node = 16  # project activation to this many values

        fp = torch.zeros(fingerprint_dim, device=self.device)

        for node in nodes:
            # Skip boundary nodes — sensor = raw input (same for all),
            # output = graph's final computation (dominated by one path)
            if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                continue
            act = node.last_activation
            if act is None:
                continue
            mag = act.norm().item()
            if mag < 1e-8:
                continue

            act_flat = act.reshape(-1)
            n = act_flat.shape[0]

            # Precompute index arrays once per node (cached).
            cache_key = node.id
            if cache_key not in self._fp_index_cache:
                if cache_key not in self._node_hash_cache:
                    self._node_hash_cache[cache_key] = hashlib.sha256(
                        cache_key.encode()
                    ).digest()
                h = self._node_hash_cache[cache_key]
                base_src = int.from_bytes(h[:8], "little")
                base_tgt = int.from_bytes(h[8:16], "little")
                stride_src = (int.from_bytes(h[16:20], "little") | 1) % n or 1
                stride_tgt = (int.from_bytes(h[20:24], "little") | 1) % fingerprint_dim or 1
                src_idx = torch.tensor(
                    [(base_src + i * stride_src) % n for i in range(values_per_node)],
                    dtype=torch.long, device=self.device,
                )
                tgt_idx = torch.tensor(
                    [(base_tgt + i * stride_tgt) % fingerprint_dim for i in range(values_per_node)],
                    dtype=torch.long, device=self.device,
                )
                self._fp_index_cache[cache_key] = (src_idx, tgt_idx)

            src_idx, tgt_idx = self._fp_index_cache[cache_key]
            fp.scatter_add_(0, tgt_idx, act_flat[src_idx])

        return fp

    def _modulate_growth_rates(self) -> None:
        """Adjust SOMA config growth rates based on prediction error.

        High prediction error means the graph lacks structure for the
        current input pattern — increase synaptogenesis and neurogenesis
        rates.  Low error means the pattern is well-learned — reduce
        growth to prevent over-connection.

        This replaces static config tuning with self-regulation.
        """
        if len(self.error_history) < 10:
            return  # not enough data yet

        recent = list(self.error_history)[-10:]
        avg_error = sum(recent) / len(recent)

        # Scale synaptogenesis rate: baseline * (1 + 10 * error)
        # At error=0.005 (typical early): rate = 2.0 * 1.05 = 2.1
        # At error=0.0001 (well-learned): rate = 2.0 * 1.001 ≈ 2.0
        # At error=0.01 (novel input): rate = 2.0 * 1.1 = 2.2
        base_syn = 2.0
        self.config.synaptogenesis_rate = base_syn * (1.0 + 10.0 * avg_error)

        # NOTE: dynamic pruning was tested but over-corrects — input
        # diversification already prevents over-connection. Keep pruning
        # at the static default (200 steps).

    def _build_projection(
        self, gen: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Build a new input projection respecting ``config.projection_mode``.

        - ``frozen_random``: plain Tensor, normalized rows. Backward-compat
          with pre-Direction-2 behavior.
        - ``learnable``: nn.Parameter with same initial shape/distribution
          so behavior at step 0 is identical to the frozen variant.

        ``gen`` is an optional torch.Generator for deterministic init.
        """
        dim = self.config.sensor_output_dim
        if gen is None:
            raw = torch.randn(dim, dim).to(self.device)
        else:
            raw = torch.randn(dim, dim, generator=gen).to(self.device)
        normalized = raw / (raw.norm(dim=1, keepdim=True) + 1e-8)
        if self.config.projection_mode == "learnable":
            return torch.nn.Parameter(normalized)
        return normalized

    def _ensure_projection(self, node_id: str) -> None:
        """Create an input projection for a node if missing.

        Called lazily when new nodes are born via neurogenesis, ensuring
        every associator has its own "receptive field" for diversification.
        The type (plain Tensor vs nn.Parameter) follows
        ``config.projection_mode`` — consistent with the projections
        created at __init__. New learnable projections are ALSO added
        to the prediction optimizer so they participate in backprop.
        """
        if node_id in self._input_projections:
            return
        proj = self._build_projection()
        self._input_projections[node_id] = proj
        # Learnable projections need to be registered with the
        # optimizer; new nodes born via neurogenesis would otherwise
        # have parameters that torch.optim never sees.
        if (
            self.config.projection_mode == "learnable"
            and isinstance(proj, torch.nn.Parameter)
        ):
            self._pred_optimizer.add_param_group(
                {"params": [proj], "lr": self.config.projection_lr}
            )

    def _diversify_activations(
        self, input_tensor: torch.Tensor, temperature: float = 5.0,
    ) -> None:
        """Modulate each associator's activation by its unique input view.

        Each node has a frozen random projection ("receptive field").
        The dot product of the projection with the input determines a
        scalar gain. High temperature makes the gain sharply selective:
        nodes whose projection aligns well get boosted, others get
        nearly zeroed. This creates genuinely different winner sets
        for different inputs.

        ``temperature`` controls selectivity:
        - 1.0: gentle modulation (gains ≈ [0.5, 2.0])
        - 5.0: sharp selection (gains bimodal: near 0 or near 2)
        """
        from soma.core.node import NodeType

        inp = input_tensor.detach().to(self.device)
        inp_norm_sq = inp.norm() ** 2 + 1e-8

        for node in self.soma.graph.all_nodes():
            if node.node_type != NodeType.ASSOCIATOR:
                continue
            if node.last_activation is None:
                continue

            self._ensure_projection(node.id)
            proj = self._input_projections[node.id]

            alignment = torch.dot(torch.mv(proj, inp), inp)
            # Temperature-scaled sigmoid: higher T = sharper selection
            gain = 0.5 + 1.5 * torch.sigmoid(
                temperature * alignment / inp_norm_sq
            )
            node.last_activation = node.last_activation * gain.item()

    def _apply_lateral_inhibition(
        self, keep_ratio: float = 0.1, min_active: int = 3,
    ) -> list[str]:
        """Suppress weakest nodes' last_activation in-place.

        After SOMA.step(), zero out the activations of the least active
        nodes.  This affects the stored fingerprint AND future
        synaptogenesis (suppressed nodes aren't counted as co-active).

        ``min_active`` ensures at least this many nodes survive even in
        small graphs where ``keep_ratio`` would leave only 1-2 winners,
        making fingerprints too coarse to discriminate inputs.

        Returns the IDs of suppressed nodes (used by anti-Hebbian
        learning to push suppressed nodes away from the input).
        """
        from soma.core.node import NodeType

        nodes = self.soma.graph.all_nodes()
        eligible = [
            n for n in nodes
            if n.node_type not in (NodeType.SENSOR, NodeType.OUTPUT)
            and n.last_activation is not None
        ]
        if not eligible:
            return []

        mags = [(n, n.last_activation.norm().item()) for n in eligible]
        k = max(min_active, int(len(mags) * keep_ratio))
        k = min(k, len(mags))
        threshold = sorted([m for _, m in mags], reverse=True)[min(k - 1, len(mags) - 1)]

        suppressed: list[str] = []
        for node, mag in mags:
            if mag < threshold:
                node.last_activation = torch.zeros_like(node.last_activation)
                suppressed.append(node.id)
        return suppressed

    def _competitive_learning(
        self,
        input_tensor: torch.Tensor,
        suppressed_ids: list[str] | None = None,
        lr: float = 0.001,
        anti_lr: float = 0.0003,
    ) -> None:
        """Competitive learning with anti-Hebbian suppression.

        Winner node (most surprised by this input) adapts toward it.
        Suppressed nodes adapt AWAY — anti-Hebbian learning makes them
        less responsive to this pattern, freeing them to specialize
        for other inputs.

        Biology: inhibited cortical neurons undergo synaptic depression
        for the active input pattern, making them selectively responsive
        to different stimuli over time.
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

        # Score by surprise (deviation from running average)
        scored = []
        for n in eligible:
            mag = n.last_activation.norm().item()
            avg = n.activation_ema
            surprise = mag - avg if avg > 0 else mag
            scored.append((surprise, n))

        scored.sort(key=lambda t: -t[0])
        winner = scored[0][1]

        # Prepare input vector for weight updates
        inp = input_tensor.detach().to(winner.linear1.weight.device)
        w_shape = winner.linear1.weight.shape[1]
        if inp.shape[0] != w_shape:
            if inp.shape[0] > w_shape:
                inp = inp[:w_shape]
            else:
                padded = torch.zeros(w_shape, device=inp.device)
                padded[: inp.shape[0]] = inp
                inp = padded

        # Hebbian: winner adapts toward input
        if scored[0][0] > 0:
            with torch.no_grad():
                w = winner.linear1.weight
                delta = inp.unsqueeze(0) - w
                w.add_(delta, alpha=lr)

        # Anti-Hebbian: suppressed nodes adapt away from input
        if suppressed_ids:
            with torch.no_grad():
                for nid in suppressed_ids:
                    if nid not in self.soma.graph.nodes:
                        continue
                    node = self.soma.graph.nodes[nid]
                    w = node.linear1.weight
                    if w.shape[1] != inp.shape[0]:
                        continue
                    # Push weights AWAY from input
                    delta = inp.unsqueeze(0) - w
                    w.add_(delta, alpha=-anti_lr)

        # Learnable diversification projections: update the random
        # projections used by _diversify_activations() so that winners
        # become MORE responsive to this input class and suppressed
        # nodes become LESS responsive. This breaks the frozen-projection
        # bottleneck where input->winner mapping is static.
        proj_lr = lr * 0.1  # slower than weight updates
        inp_full = input_tensor.detach().to(self.device)
        inp_norm = inp_full / (inp_full.norm() + 1e-8)
        outer = torch.outer(inp_norm, inp_norm)  # rank-1 update

        with torch.no_grad():
            # Winner: increase alignment with this input direction
            if winner.id in self._input_projections:
                self._input_projections[winner.id].add_(outer, alpha=proj_lr)

            # Suppressed: decrease alignment with this input direction
            if suppressed_ids:
                for nid in suppressed_ids:
                    if nid in self._input_projections:
                        self._input_projections[nid].add_(
                            outer, alpha=-proj_lr * 0.3,
                        )

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

        # Apply the same diversification + inhibition pipeline used
        # during storage so query fingerprints are comparable.
        self._diversify_activations(query_tensor)
        self._apply_lateral_inhibition()
        query_act = self._get_node_fingerprint()

        # Batched cosine similarity against all stored fingerprints.
        steps = list(self._activation_store.keys())
        stored_matrix = torch.stack(
            [self._activation_store[s] for s in steps]
        )  # (N, fp_dim)
        sims = torch.nn.functional.cosine_similarity(
            query_act.unsqueeze(0), stored_matrix, dim=1,
        )  # (N,)

        # Top-k indices
        k = min(top_k, len(steps))
        top_sims, top_indices = torch.topk(sims, k)

        results: list[tuple[int, str, float]] = []
        for i in range(k):
            step = steps[top_indices[i].item()]
            text = self.text_store.get(step, "")
            if text:
                results.append((step, text, float(top_sims[i].item())))
        return results

    def retrieve_by_tokens(
        self,
        query_tokens: set[int],
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """Retrieve stored texts by BPE token overlap (Jaccard sim).

        A simpler retrieval method that uses character-level token
        overlap instead of graph activation fingerprints. Outperforms
        graph-based retrieval when the encoder uses random (untrained)
        embeddings, because token overlap directly captures subword
        sharing without depending on learned graph structure.

        ``query_tokens`` should be a set of BPE token IDs from the
        query text. The caller is responsible for tokenizing.
        """
        if not self.text_store:
            return []

        scored: list[tuple[float, int]] = []
        for step, _text in self.text_store.items():
            if step not in self._token_cache:
                continue
            t_ids = self._token_cache[step]
            if not query_tokens or not t_ids:
                scored.append((0.0, step))
                continue
            overlap = len(query_tokens & t_ids)
            union = len(query_tokens | t_ids)
            jaccard = overlap / union if union > 0 else 0.0
            scored.append((jaccard, step))

        scored.sort(key=lambda t: -t[0])
        k = min(top_k, len(scored))
        results: list[tuple[int, str, float]] = []
        for sim, step in scored[:k]:
            text = self.text_store.get(step, "")
            if text:
                results.append((step, text, sim))
        return results

    def retrieve_by_topology(
        self,
        query_tensor: torch.Tensor,
        top_k: int = 5,
    ) -> list[tuple[int, str, float]]:
        """Retrieve stored texts by shared active nodes in the graph.

        Instead of comparing activation fingerprint vectors, this method
        finds which graph nodes the query activates and looks up which
        stored memories activated the SAME nodes. Memories that share
        more active nodes with the query rank higher.

        This captures learned structural associations: if SOMA's graph
        learns (through Hebbian + synaptogenesis) that certain nodes
        respond to both "art" and "dogs," then a query about art will
        retrieve dog memories through their shared active nodes —
        cross-domain association that fingerprint comparison cannot do.
        """
        if not self._node_memory_index:
            return []

        query_tensor = query_tensor.to(self.device)
        modality = self.config.input_modalities[0]

        # Run query through graph
        self.soma.step({modality: query_tensor}, eval_mode=True)
        self._diversify_activations(query_tensor)
        self._apply_lateral_inhibition()

        # Find which nodes are active for this query
        from soma.core.node import NodeType

        active_nodes: set[str] = set()
        for node in self.soma.graph.all_nodes():
            if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                continue
            if (
                node.last_activation is not None
                and node.last_activation.norm().item() > 1e-8
            ):
                active_nodes.add(node.id)

        if not active_nodes:
            return []

        # Count how many active nodes each stored memory shares
        memory_scores: dict[int, float] = {}
        for nid in active_nodes:
            if nid not in self._node_memory_index:
                continue
            for step in self._node_memory_index[nid]:
                memory_scores[step] = memory_scores.get(step, 0) + 1.0

        # Normalize by total active nodes for a Jaccard-like score
        n_active = len(active_nodes)
        scored = [
            (count / n_active, step)
            for step, count in memory_scores.items()
        ]
        scored.sort(key=lambda t: -t[0])

        results: list[tuple[int, str, float]] = []
        for sim, step in scored[:top_k]:
            text = self.text_store.get(step, "")
            if text:
                results.append((step, text, sim))
        return results

    def retrieve_hybrid(
        self,
        query_tensor: torch.Tensor,
        corpus_embeddings: torch.Tensor,
        corpus_step_map: dict[int, int],
        *,
        recall_k: int = 20,
        top_k: int = 5,
        gate_threshold: float = 0.05,
        rerank_weight: float = 0.2,
        adaptive_weight: bool = False,
    ) -> list[tuple[int, str, float]]:
        """Confidence-gated hybrid retrieval.

        Uses embedding cosine similarity for candidate recall, then
        selectively reranks using SOMA's graph fingerprint when the
        graph's confidence exceeds ``gate_threshold``.

        This architecture lets SOMA add value without hurting: the
        graph only intervenes when it has a strong structural signal
        (learned co-occurrence / temporal patterns). On queries where
        the graph is unsure, pure embedding similarity is used.

        Parameters
        ----------
        query_tensor:
            Query embedding (same dim as corpus_embeddings).
        corpus_embeddings:
            Stacked embeddings for the full corpus, shape (N, dim).
        corpus_step_map:
            Maps SOMA step numbers to corpus indices, for cross-
            referencing graph fingerprints with corpus entries.
        recall_k:
            Number of candidates to retrieve via embedding similarity.
        top_k:
            Number of final results to return.
        gate_threshold:
            Minimum fingerprint confidence (top-1 minus top-2 sim)
            required to apply graph reranking. Lower = more aggressive
            (more queries reranked). 0.05 is a good default.
        rerank_weight:
            Maximum weight of graph signal in reranking formula:
            ``(1-w)*emb_sim + w*fp_sim``. 0.2 is a good default.
        adaptive_weight:
            When True, scale rerank_weight by the absolute quality
            of the best fingerprint match (fp_sorted[0]). This
            prevents confidently-wrong reranking: a high confidence
            gap between low-similarity fingerprints gets attenuated.

        Returns
        -------
        list of (step, text, score) tuples, sorted by combined score.
        """
        if not self._activation_store:
            # No graph data yet — fall back to pure embedding retrieval
            return []

        query_tensor = query_tensor.to(self.device)
        modality = self.config.input_modalities[0]

        # Step 1: Embedding recall
        sims = torch.nn.functional.cosine_similarity(
            query_tensor.unsqueeze(0), corpus_embeddings, dim=1,
        )
        k = min(recall_k, len(corpus_embeddings))
        top_k_sims, top_k_indices = torch.topk(sims, k)

        # Step 2: Graph fingerprint for query
        self.soma.step({modality: query_tensor}, eval_mode=True)
        self._diversify_activations(query_tensor)
        self._apply_lateral_inhibition()
        query_fp = self._get_node_fingerprint()

        # Step 3: Compute fingerprint similarity for each candidate
        # Build reverse map: corpus_idx -> step_num
        cidx_to_step: dict[int, int] = {}
        for step_num, cidx in corpus_step_map.items():
            cidx_to_step[cidx] = step_num

        fp_sims: list[float] = []
        for i in range(k):
            cidx = top_k_indices[i].item()
            step_num = cidx_to_step.get(cidx)
            fp_sim = 0.0
            if (
                step_num is not None
                and step_num in self._activation_store
            ):
                fp_sim = torch.nn.functional.cosine_similarity(
                    query_fp.unsqueeze(0),
                    self._activation_store[step_num].unsqueeze(0),
                ).item()
            fp_sims.append(fp_sim)

        # Step 4: Confidence gate
        if fp_sims:
            fp_sorted = sorted(fp_sims, reverse=True)
            confidence = fp_sorted[0] - (
                fp_sorted[1] if len(fp_sorted) > 1 else 0.0
            )
        else:
            confidence = 0.0

        # Step 5: Rerank if confident, else use embedding order
        if confidence >= gate_threshold:
            w = rerank_weight
            if adaptive_weight and fp_sims:
                fp_sorted_vals = sorted(fp_sims, reverse=True)
                w = rerank_weight * max(0.0, min(fp_sorted_vals[0], 1.0))
            scored = [
                (
                    (1 - w) * top_k_sims[i].item()
                    + w * fp_sims[i],
                    top_k_indices[i].item(),
                )
                for i in range(k)
            ]
            scored.sort(key=lambda t: -t[0])
            final_indices = [idx for _, idx in scored[:top_k]]
        else:
            final_indices = [
                top_k_indices[i].item()
                for i in range(min(top_k, k))
            ]

        # Step 6: Build results
        results: list[tuple[int, str, float]] = []
        for cidx in final_indices:
            step_num = cidx_to_step.get(cidx)
            if step_num is not None:
                text = self.text_store.get(step_num, "")
                if text:
                    score = sims[cidx].item()
                    results.append((step_num, text, score))
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
            "token_cache": self._token_cache,
            "node_memory_index": {
                k: list(v) for k, v in self._node_memory_index.items()
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
        self._token_cache = state.get("token_cache", {})
        raw_index = state.get("node_memory_index", {})
        self._node_memory_index = {
            k: set(v) for k, v in raw_index.items()
        }
        # Preserve projection type across load: when config says
        # learnable, reconstruct as nn.Parameter so gradients continue
        # to flow. Otherwise stay as plain Tensor.
        self._input_projections = {}
        for k, v in state["input_projections"].items():
            tensor = v.to(self.device)
            if self.config.projection_mode == "learnable":
                self._input_projections[k] = torch.nn.Parameter(tensor)
            else:
                self._input_projections[k] = tensor
        # Rebuild the optimizer so reloaded Parameter instances are
        # actually optimized (old optimizer references the old instances).
        if self.config.projection_mode == "learnable":
            self._pred_optimizer = torch.optim.Adam(
                [
                    {"params": list(self.prediction_head.parameters()), "lr": 0.0003},
                    {
                        "params": [
                            p for p in self._input_projections.values()
                            if isinstance(p, torch.nn.Parameter)
                        ],
                        "lr": self.config.projection_lr,
                    },
                ]
            )
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
    # Encoder fine-tuning support
    # ------------------------------------------------------------------

    def compute_contrastive_loss(
        self,
        query_embedding: torch.Tensor,
        recent_steps: int = 50,
        temperature: float = 0.1,
    ) -> torch.Tensor | None:
        """Compute a contrastive loss using the graph's topology.

        Uses the node memory index as supervision: memories that share
        active nodes with the current query should have similar
        embeddings, and memories that don't should be pushed apart.

        This loss has a grad_fn connected to ``query_embedding``, so
        calling ``.backward()`` on it will propagate gradients into
        whatever encoder produced the embedding.

        Returns ``None`` if there aren't enough stored memories yet.
        """
        if len(self._activation_store) < 10:
            return None

        query_embedding = query_embedding.to(self.device)
        modality = self.config.input_modalities[0]

        # Run query through graph to find active nodes
        self.soma.step({modality: query_embedding.detach()}, eval_mode=True)
        self._diversify_activations(query_embedding.detach())
        self._apply_lateral_inhibition()

        from soma.core.node import NodeType

        active_nodes: set[str] = set()
        for node in self.soma.graph.all_nodes():
            if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                continue
            if (
                node.last_activation is not None
                and node.last_activation.norm().item() > 1e-8
            ):
                active_nodes.add(node.id)

        if not active_nodes:
            return None

        # Find positive and negative memories based on node overlap
        n_active = len(active_nodes)
        all_steps = list(self._activation_store.keys())[-recent_steps:]
        if len(all_steps) < 4:
            return None

        positives: list[int] = []
        negatives: list[int] = []
        for step in all_steps:
            overlap = sum(
                1 for nid in active_nodes
                if nid in self._node_memory_index
                and step in self._node_memory_index[nid]
            )
            ratio = overlap / n_active
            if ratio >= 0.5:
                positives.append(step)
            elif ratio == 0.0:
                negatives.append(step)

        if not positives or not negatives:
            return None

        # Contrastive loss: pull query toward positive fingerprints,
        # push away from negative fingerprints.
        # Uses the stored fingerprints as anchors (detached).
        pos_fps = torch.stack(
            [self._activation_store[s].detach() for s in positives[:8]]
        )
        neg_fps = torch.stack(
            [self._activation_store[s].detach() for s in negatives[:8]]
        )

        # Project query embedding to fingerprint space for comparison
        query_fp = self._get_node_fingerprint()  # detached from graph

        # The loss: we want query_fp to be close to pos_fps and far
        # from neg_fps. But query_fp is detached from the encoder.
        # Instead, use the raw embedding similarity as a proxy:
        # the encoder should produce embeddings where same-topology
        # memories are closer together.
        #
        # Approximate: use cosine similarity of the query embedding
        # against stored embeddings (if we had them). Since we don't
        # store raw embeddings, use the fingerprint as a fixed target
        # and compute MSE between a learned projection of the query
        # and the positive fingerprint mean.
        pos_mean = pos_fps.mean(dim=0)
        neg_mean = neg_fps.mean(dim=0)

        # This is a simplified contrastive objective:
        # minimize distance to positive centroid, maximize to negative
        fp_dim = pos_mean.shape[0]
        q_proj = query_embedding[:fp_dim] if query_embedding.shape[0] >= fp_dim else torch.nn.functional.pad(query_embedding, (0, fp_dim - query_embedding.shape[0]))

        pos_dist = torch.nn.functional.mse_loss(q_proj, pos_mean)
        neg_dist = torch.nn.functional.mse_loss(q_proj, neg_mean)

        # Triplet-style: want pos_dist < neg_dist by a margin
        loss = torch.clamp(pos_dist - neg_dist + temperature, min=0.0)
        return loss

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

        # Dynamic config: modulate growth rates by prediction error.
        # High error → more wiring (the graph lacks structure).
        # Low error → less growth (the graph has learned this pattern).
        self._modulate_growth_rates()

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
        suppressed = self._apply_lateral_inhibition()

        # Competitive learning with anti-Hebbian suppression:
        # winner adapts toward input, suppressed nodes adapt away.
        # This drives node specialization without requiring sparse
        # initial connectivity.
        self._competitive_learning(input_tensor, suppressed_ids=suppressed)

        # Store original text for retrieval/verbalization
        step_num = step_result.get("global_step", len(self.text_store))
        if source_text is not None:
            self.text_store[step_num] = source_text
            # Cache BPE token IDs for token-overlap retrieval
            if self._tokenizer_fn is not None:
                self._token_cache[step_num] = set(
                    self._tokenizer_fn(source_text)
                )

        current_summary = self._get_activation_summary(step_result)

        # Store activation fingerprint for graph-driven retrieval
        if source_text is not None:
            self._activation_store[step_num] = self._get_node_fingerprint()
            # Index which nodes were active for this memory (for
            # topology-based retrieval)
            from soma.core.node import NodeType as _NT

            for node in self.soma.graph.all_nodes():
                if (
                    node.last_activation is not None
                    and node.last_activation.norm().item() > 1e-8
                    and node.node_type not in (_NT.SENSOR, _NT.OUTPUT)
                ):
                    if node.id not in self._node_memory_index:
                        self._node_memory_index[node.id] = set()
                    self._node_memory_index[node.id].add(step_num)

        # Train prediction head: re-predict from last summary,
        # compare to current summary, backprop.
        if self._last_summary is not None:
            # Direction 2: when projections are learnable, route the
            # prediction through a projection-averaged view so the
            # prediction loss trains BOTH the prediction head AND the
            # learnable projections. Each associator's projection
            # contributes a view of _last_summary; we mean them into
            # the prediction head's input. Gradient pathway:
            # pred_loss -> prediction_head -> projected_last
            #    -> mean(proj @ _last_summary for each proj)
            #    -> each projection's parameters.
            if (
                self.config.projection_mode == "learnable"
                and self._input_projections
            ):
                projected_views = []
                for proj in self._input_projections.values():
                    # Treat _last_summary as detached input (it was
                    # already detached on save) so gradient flows to
                    # proj but not back into SOMA's graph.
                    projected_views.append(proj @ self._last_summary)
                pred_input = torch.stack(projected_views).mean(dim=0)
            else:
                pred_input = self._last_summary
            predicted = self.prediction_head(pred_input)
            pred_loss = torch.nn.functional.mse_loss(
                predicted, current_summary.detach(),
            )
            self.prediction_error = pred_loss.item()

            # Direction 4a: optional LLM-distillation loss. When enabled
            # and a teacher + source_text are available, add an alpha-
            # weighted cosine-distance term between the projection-
            # averaged student view and the teacher embedding. Teacher
            # dim may differ from student dim (teacher 1024 for mxbai,
            # student sensor_output_dim); truncate/pad to align.
            distill_loss: torch.Tensor | None = None
            if (
                self.config.projection_distillation_target == "llm_embedding"
                and self._teacher is not None
                and source_text is not None
                and self.config.projection_mode == "learnable"
                and self._input_projections
            ):
                teacher_emb = self._teacher.embed(source_text).to(self.device)
                s_dim = pred_input.shape[0]
                if teacher_emb.shape[0] >= s_dim:
                    teacher_aligned = teacher_emb[:s_dim]
                else:
                    teacher_aligned = torch.zeros(s_dim, device=self.device)
                    teacher_aligned[: teacher_emb.shape[0]] = teacher_emb
                cos = torch.nn.functional.cosine_similarity(
                    pred_input.unsqueeze(0),
                    teacher_aligned.detach().unsqueeze(0),
                    dim=1,
                )
                distill_loss = (
                    1.0 - cos.squeeze()
                ) * self.config.projection_distillation_weight

            # Only update if error is still meaningful — prevent
            # over-convergence that collapses all fingerprints.
            # Biological analogy: synaptic plasticity decreases
            # for well-learned patterns but never reaches zero.
            if self.prediction_error > 1e-5 or distill_loss is not None:
                self._pred_optimizer.zero_grad()
                total_loss = pred_loss
                if distill_loss is not None:
                    total_loss = total_loss + distill_loss
                total_loss.backward()
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
