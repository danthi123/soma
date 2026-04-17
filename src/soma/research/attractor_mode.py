"""Configure SOMA as a recurrent attractor network for associative recall.

Research Direction D, Sub-phase D2.

The idea: SOMA's plastic graph with Hebbian learning generalises a Hopfield
network -- continuous activations, learned connectivity, structural growth +
homeostasis.  This module provides helpers that:

1. Construct a SOMA instance sized for a given pattern dimension.
2. **Store** patterns by running ``soma.step()`` with targets=input
   (driving Hebbian learning to encode the pattern).
3. **Recall** patterns by iterating ``soma.step()`` on a noisy probe
   with ``eval_mode=True`` (no weight updates, no growth) until the
   output converges or the iteration budget is spent.

No SOMA internals are modified -- the module only manipulates the public
``step()`` / ``SOMAConfig`` API.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from soma.core.config import SOMAConfig
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
