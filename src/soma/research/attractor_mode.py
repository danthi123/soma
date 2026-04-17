"""Configure SOMA as a recurrent attractor network for associative recall.

Research Direction D, Sub-phases D2 and D3.

The idea: SOMA's plastic graph with Hebbian learning generalises a Hopfield
network -- continuous activations, learned connectivity, structural growth +
homeostasis.  This module provides helpers that:

1. Construct a SOMA instance sized for a given pattern dimension.
2. **Store** patterns by running ``soma.step()`` with targets=input
   (driving Hebbian learning to encode the pattern).
3. **Recall** patterns by iterating ``soma.step()`` on a noisy probe
   with ``eval_mode=True`` (no weight updates, no growth) until the
   output converges or the iteration budget is spent.
4. **Output scaling** (D3): z-score normalisation and learned scaler to
   fix magnitude attenuation (SOMA outputs ~0.01-0.20 instead of +/-1).
5. **Structural plasticity helpers** (D3): trigger neurogenesis /
   synaptogenesis from outside ``SOMA.step()`` to grow capacity on
   saturation.

No SOMA internals are modified -- the module only manipulates the public
``step()`` / ``SOMAConfig`` API plus the standalone growth functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.growth.neurogenesis import neurogenesis
from soma.growth.synaptogenesis import synaptogenesis
from soma.system import SOMA


def make_attractor_config(
    pattern_dim: int,
    *,
    n_associators: int = 8,
    n_integrators: int = 4,
    hebbian_lr: float = 0.01,
    base_lr: float = 0.001,
    seed: int | None = 42,
) -> SOMAConfig:
    """Return a SOMAConfig tuned for attractor-mode experiments.

    The sensor and output nodes both use *pattern_dim* as their feature
    dimension, so a stored pattern can be fed in and read back out at the
    same size.

    Growth intervals are set very high (effectively disabled) so the
    graph topology stays fixed during store/recall.
    """
    return SOMAConfig(
        # Dimensions -- everything sized to pattern_dim
        sensor_output_dim=pattern_dim,
        associator_input_dim=pattern_dim,
        associator_hidden_dim=pattern_dim * 2,
        associator_output_dim=pattern_dim,
        integrator_input_dim=pattern_dim,
        integrator_hidden_dim=pattern_dim * 2,
        integrator_output_dim=pattern_dim,
        # Topology
        initial_associator_count=n_associators,
        initial_integrator_count=n_integrators,
        # Learning
        base_lr=base_lr,
        hebbian_lr=hebbian_lr,
        edge_weight_decay=1.0,  # no decay during attractor experiments
        activation_threshold=0.0,  # always count coactivation
        # Disable growth -- set intervals astronomically high
        synaptogenesis_interval=999_999_999,
        neurogenesis_interval=999_999_999,
        pruning_interval=999_999_999,
        consolidation_interval=999_999_999,
        # Homeostasis bounds
        gain_min=0.1,
        gain_max=10.0,
        # Memory systems (small -- not the focus)
        wm_slots=4,
        wm_dim=pattern_dim,
        episodic_capacity=100,
        key_dim=pattern_dim,
        value_dim=pattern_dim * 2,
        # Text-related defaults (needed by config validation)
        input_modalities=["text"],
        output_modalities=["text"],
        text_embed_dim=pattern_dim,
        # Misc
        seed=seed,
        use_batched_executor=True,
    )


def make_attractor_soma(
    pattern_dim: int,
    *,
    n_associators: int = 8,
    n_integrators: int = 4,
    hebbian_lr: float = 0.01,
    base_lr: float = 0.001,
    seed: int | None = 42,
    device: torch.device | str | None = None,
) -> SOMA:
    """Create a SOMA instance configured for attractor-mode experiments."""
    config = make_attractor_config(
        pattern_dim,
        n_associators=n_associators,
        n_integrators=n_integrators,
        hebbian_lr=hebbian_lr,
        base_lr=base_lr,
        seed=seed,
    )
    return SOMA(config, device=device)


def store_pattern(
    soma: SOMA,
    pattern: torch.Tensor,
    *,
    n_presentations: int = 1,
    modality: str = "text",
) -> list[float]:
    """Present a pattern to SOMA for storage via Hebbian learning.

    Feeds *pattern* as both input and target so the loss drives
    backprop + Hebbian updates toward reproducing the pattern.

    Returns the loss value for each presentation step.
    """
    losses: list[float] = []
    for _ in range(n_presentations):
        result = soma.step(
            inputs={modality: pattern},
            targets={modality: pattern},
        )
        loss_val = result.get("loss")
        losses.append(float(loss_val) if loss_val is not None else 0.0)
    return losses


@dataclass
class RecallResult:
    """Result of running SOMA recall for one probe."""

    final_output: torch.Tensor
    outputs_per_iter: list[torch.Tensor]
    converged: bool
    convergence_iter: int | None  # iteration at which convergence was declared


def recall_pattern(
    soma: SOMA,
    probe: torch.Tensor,
    *,
    max_iters: int = 100,
    convergence_tol: float = 1e-5,
    convergence_window: int = 5,
    modality: str = "text",
) -> RecallResult:
    """Recall a pattern by iterating SOMA in e-val mode on a noisy probe.

    Feeds *probe* as input, runs ``soma.step(eval_mode=True)`` for up to
    *max_iters* iterations, reading the output at each step.  Declares
    convergence when the L2 distance between consecutive outputs is below
    *convergence_tol* for *convergence_window* consecutive iterations.

    Returns a ``RecallResult`` with per-iteration outputs and convergence
    info.
    """
    outputs: list[torch.Tensor] = []
    consecutive_stable = 0
    converged = False
    convergence_iter: int | None = None

    for i in range(max_iters):
        # Feed the probe as input each iteration (clamped input, like
        # a Hopfield network clamping visible units).
        result = soma.step(
            inputs={modality: probe},
            eval_mode=True,
        )
        out_dict = result["outputs"]
        out_tensor = out_dict.get(modality)
        if out_tensor is None:
            # Output node was dormant -- use zeros
            out_tensor = torch.zeros_like(probe)
        outputs.append(out_tensor.detach().clone())

        # Convergence check
        if len(outputs) >= 2:
            diff = (outputs[-1] - outputs[-2]).norm().item()
            if diff < convergence_tol:
                consecutive_stable += 1
            else:
                consecutive_stable = 0
            if consecutive_stable >= convergence_window:
                converged = True
                convergence_iter = i + 1
                break

    return RecallResult(
        final_output=outputs[-1],
        outputs_per_iter=outputs,
        converged=converged,
        convergence_iter=convergence_iter,
    )


# ======================================================================
# D3: Output scaling — fix magnitude attenuation
# ======================================================================


@dataclass
class ZScoreScaler:
    """Per-dimension z-score normalisation fitted on SOMA recall outputs.

    Computes mean/std over the stored patterns' recall outputs, then
    normalises any new output to zero-mean, unit-variance before
    multiplying by the target standard deviation (1.0 for bipolar
    patterns whose per-dim std is 1).
    """

    mean: torch.Tensor  # (D,)
    std: torch.Tensor  # (D,)
    target_std: float = 1.0

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise *x* using the fitted statistics."""
        return (x - self.mean) / (self.std + 1e-8) * self.target_std


def fit_zscore_scaler(
    soma: SOMA,
    patterns: torch.Tensor,
    *,
    n_iters: int = 50,
    modality: str = "text",
) -> ZScoreScaler:
    """Fit a z-score scaler by recalling each stored pattern and
    collecting the raw SOMA outputs.

    Args:
        soma: a SOMA instance that has already stored the patterns.
        patterns: (N, D) tensor of the originally stored patterns.
        n_iters: recall iterations per pattern.
        modality: input/output modality key.

    Returns:
        A fitted ``ZScoreScaler``.
    """
    raw_outputs: list[torch.Tensor] = []
    for i in range(patterns.shape[0]):
        rr = recall_pattern(
            soma,
            patterns[i],
            max_iters=n_iters,
            modality=modality,
        )
        raw_outputs.append(rr.final_output.detach())
    stacked = torch.stack(raw_outputs)  # (N, D)
    return ZScoreScaler(
        mean=stacked.mean(dim=0),
        std=stacked.std(dim=0),
    )


class OutputScaler(nn.Module):
    """Learned per-dimension affine scaler: ``y = x * scale + bias``.

    Trained on (SOMA raw output, original pattern) pairs with MSE loss.
    Only 2*D parameters — trivially small.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


def fit_output_scaler(
    soma: SOMA,
    patterns: torch.Tensor,
    *,
    n_iters: int = 50,
    lr: float = 0.01,
    train_steps: int = 100,
    modality: str = "text",
) -> OutputScaler:
    """Fit a learned ``OutputScaler`` on (raw_output, target_pattern) pairs.

    Args:
        soma: SOMA instance with stored patterns.
        patterns: (N, D) the originally stored bipolar patterns.
        n_iters: recall iterations per pattern.
        lr: Adam learning rate for the scaler.
        train_steps: number of optimisation steps.
        modality: input/output modality key.

    Returns:
        A trained ``OutputScaler``.
    """
    raw_outputs: list[torch.Tensor] = []
    for i in range(patterns.shape[0]):
        rr = recall_pattern(soma, patterns[i], max_iters=n_iters, modality=modality)
        raw_outputs.append(rr.final_output.detach())
    X = torch.stack(raw_outputs)  # (N, D)
    Y = patterns.detach()  # (N, D)

    dim = patterns.shape[1]
    scaler = OutputScaler(dim)
    # Move scaler to same device as data
    scaler = scaler.to(X.device)

    optimizer = torch.optim.Adam(scaler.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(train_steps):
        optimizer.zero_grad()
        pred = scaler(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()

    scaler.train(False)
    return scaler


# ======================================================================
# D3: Structural plasticity helpers
# ======================================================================


def trigger_neurogenesis(
    soma: SOMA,
    *,
    num_new_nodes: int = 1,
    num_neighbors: int = 5,
    rng: torch.Generator | None = None,
) -> list[str]:
    """Force neurogenesis on a SOMA instance, bypassing the interval/
    threshold checks.

    Directly calls ``neurogenesis()`` with a fake error history that
    guarantees the threshold is exceeded.  Returns the IDs of newly
    created nodes (may be fewer than requested if ``max_nodes`` is hit).
    """
    new_ids: list[str] = []
    for _ in range(num_new_nodes):
        # Build a fake error history that will exceed the threshold.
        # neurogenesis checks recent_mean / baseline_mean > threshold.
        # With threshold=1.2, we use recent=10.0, baseline=1.0.
        fake_errors = [1.0] * 1000 + [10.0] * 100
        new_node = neurogenesis(
            soma.graph,
            fake_errors,
            step=soma.global_step,
            config=soma.config,
            num_neighbors=num_neighbors,
            rng=rng,
        )
        if new_node is not None:
            new_ids.append(new_node.id)
            # Move new node to the same device as the rest of the graph
            if soma.device is not None:
                new_node.to(soma.device)
                # Also move new edges
                for edge in soma.graph.get_incoming_edges(new_node.id):
                    edge.to(soma.device)
                for edge in soma.graph.get_outgoing_edges(new_node.id):
                    edge.to(soma.device)
    return new_ids


def trigger_synaptogenesis(
    soma: SOMA,
    activations: dict[str, torch.Tensor] | None = None,
    *,
    rng: torch.Generator | None = None,
) -> list[str]:
    """Force synaptogenesis on a SOMA instance.

    If *activations* is None, builds a fake activation dict with
    unit-magnitude activations for all nodes so every pair is
    considered.  Returns the IDs of newly created edges.
    """
    if activations is None:
        activations = {}
        for nid, node in soma.graph.nodes.items():
            activations[nid] = torch.ones(node.output_dim, device=soma.device) * 0.5

    new_edges = synaptogenesis(
        soma.graph,
        activations,
        step=soma.global_step,
        config=soma.config,
        rng=rng,
    )
    # Move new edges to device
    if soma.device is not None:
        for edge in new_edges:
            edge.to(soma.device)
    return [e.id for e in new_edges]


def measure_recall_accuracy(
    soma: SOMA,
    patterns: torch.Tensor,
    probes: torch.Tensor,
    *,
    n_iters: int = 50,
    modality: str = "text",
    scaler: OutputScaler | ZScoreScaler | None = None,
) -> dict[str, float]:
    """Measure sign-exact and nearest-pattern recall on a set of patterns.

    Returns dict with keys: exact_rate, nearest_rate, per_bit_acc.
    """
    n = patterns.shape[0]
    dim = patterns.shape[1]
    exact = 0
    nearest = 0
    bits_correct = 0

    for i in range(n):
        rr = recall_pattern(soma, probes[i], max_iters=n_iters, modality=modality)
        output = rr.final_output

        # Apply scaler if provided
        if scaler is not None:
            if isinstance(scaler, ZScoreScaler):
                output = scaler.transform(output)
            elif isinstance(scaler, OutputScaler):
                with torch.no_grad():
                    output = scaler(output)

        recalled = output.sign()
        original = patterns[i]

        if torch.equal(recalled, original):
            exact += 1

        sims = torch.cosine_similarity(output.unsqueeze(0), patterns, dim=1)
        if sims.argmax().item() == i:
            nearest += 1

        bits_correct += int((recalled == original).sum().item())

    return {
        "exact_rate": exact / max(n, 1),
        "nearest_rate": nearest / max(n, 1),
        "per_bit_acc": bits_correct / max(n * dim, 1),
    }


def count_nodes_by_type(soma: SOMA) -> dict[str, int]:
    """Count nodes in the SOMA graph by type."""
    counts: dict[str, int] = {}
    for nt in NodeType:
        nodes = soma.graph.nodes_by_type(nt)
        if nodes:
            counts[nt.name] = len(nodes)
    return counts
