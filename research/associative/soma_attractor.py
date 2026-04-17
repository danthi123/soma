"""D2 experiments: SOMA as a recurrent attractor network.

Evaluates SOMA in attractor mode on the same benchmarks used in D1:
  1. Random-binary pattern recall (capacity sweep, D=50)
  2. MNIST denoising (D=784, N=50)

Reports convergence behaviour, exact-recall rates, capacity curves,
and the interaction between homeostasis and attractor dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from research.associative.datasets import make_binary_patterns, make_mnist_recall
from soma.research.attractor_mode import (
    make_attractor_soma,
    recall_pattern,
    store_pattern,
)

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceInfo:
    """Convergence statistics for a set of recall trials."""

    n_trials: int = 0
    n_converged: int = 0
    mean_convergence_iter: float = 0.0
    convergence_rate: float = 0.0


@dataclass
class BinaryRecallResult:
    """Results for one (N, K) configuration on random-binary patterns."""

    n_patterns: int
    dim: int
    n_iters: int
    exact_match_rate: float
    per_bit_accuracy: float
    nearest_pattern_accuracy: float  # cosine-nearest stored pattern = correct?
    convergence: ConvergenceInfo
    store_losses: list[float] = field(default_factory=list)


@dataclass
class MNISTRecallResult:
    """Results for SOMA attractor mode on MNIST denoising."""

    noise_type: str
    n_patterns: int
    n_iters: int
    mse: float
    class_accuracy: float
    per_pixel_accuracy: float
    convergence: ConvergenceInfo


# ---------------------------------------------------------------------------
# Random-binary experiments
# ---------------------------------------------------------------------------


def run_binary_recall(
    n_patterns: int,
    dim: int = 50,
    n_iters: int = 50,
    mask_frac: float = 0.2,
    *,
    n_associators: int = 8,
    n_integrators: int = 4,
    n_presentations: int = 3,
    hebbian_lr: float = 0.01,
    base_lr: float = 0.001,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> BinaryRecallResult:
    """Store N random bipolar patterns in SOMA, recall from noisy probes."""
    ds = make_binary_patterns(
        n_patterns=n_patterns,
        dim=dim,
        mask_frac=mask_frac,
        device=device,
        seed=seed,
    )

    soma = make_attractor_soma(
        pattern_dim=dim,
        n_associators=n_associators,
        n_integrators=n_integrators,
        hebbian_lr=hebbian_lr,
        base_lr=base_lr,
        seed=seed,
        device=device,
    )

    # Store phase: present each pattern multiple times
    all_store_losses: list[float] = []
    for i in range(n_patterns):
        pattern = ds.patterns[i]
        losses = store_pattern(
            soma, pattern, n_presentations=n_presentations, modality="text"
        )
        all_store_losses.extend(losses)

    # Recall phase: for each pattern, feed noisy probe and iterate
    exact_matches = 0
    nearest_matches = 0
    total_bits_correct = 0
    total_bits = 0
    convergence_count = 0
    convergence_iters: list[int] = []

    for i in range(n_patterns):
        probe = ds.probes[i]
        rr = recall_pattern(
            soma,
            probe,
            max_iters=n_iters,
            convergence_tol=1e-5,
            convergence_window=5,
            modality="text",
        )

        # Binarize output via sign function
        recalled = rr.final_output.sign()
        original = ds.patterns[i]

        # Exact match (sign-based)
        if torch.equal(recalled, original):
            exact_matches += 1

        # Nearest stored pattern (cosine similarity)
        sims = torch.cosine_similarity(
            rr.final_output.unsqueeze(0), ds.patterns, dim=1
        )
        nearest_idx = sims.argmax().item()
        if nearest_idx == i:
            nearest_matches += 1

        # Per-bit accuracy
        bits_correct = (recalled == original).sum().item()
        total_bits_correct += bits_correct
        total_bits += dim

        # Convergence tracking
        if rr.converged:
            convergence_count += 1
            if rr.convergence_iter is not None:
                convergence_iters.append(rr.convergence_iter)

    exact_rate = exact_matches / max(n_patterns, 1)
    nearest_rate = nearest_matches / max(n_patterns, 1)
    per_bit_acc = total_bits_correct / max(total_bits, 1)
    conv_info = ConvergenceInfo(
        n_trials=n_patterns,
        n_converged=convergence_count,
        mean_convergence_iter=(
            sum(convergence_iters) / len(convergence_iters)
            if convergence_iters
            else 0.0
        ),
        convergence_rate=convergence_count / max(n_patterns, 1),
    )

    return BinaryRecallResult(
        n_patterns=n_patterns,
        dim=dim,
        n_iters=n_iters,
        exact_match_rate=exact_rate,
        per_bit_accuracy=per_bit_acc,
        nearest_pattern_accuracy=nearest_rate,
        convergence=conv_info,
        store_losses=all_store_losses,
    )


def run_binary_capacity_sweep(
    dim: int = 50,
    n_values: list[int] | None = None,
    k_values: list[int] | None = None,
    *,
    n_associators: int = 8,
    n_integrators: int = 4,
    n_presentations: int = 3,
    hebbian_lr: float = 0.01,
    base_lr: float = 0.001,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> list[dict]:
    """Sweep N (pattern count) and K (recall iterations) on random-binary."""
    if n_values is None:
        n_values = [1, 2, 3, 5, 7, 10, 12, 15, 20]
    if k_values is None:
        k_values = [1, 5, 10, 50, 100]

    results: list[dict] = []

    for n in n_values:
        row: dict = {"n_patterns": n, "dim": dim, "ratio": n / dim}
        for k in k_values:
            r = run_binary_recall(
                n_patterns=n,
                dim=dim,
                n_iters=k,
                n_associators=n_associators,
                n_integrators=n_integrators,
                n_presentations=n_presentations,
                hebbian_lr=hebbian_lr,
                base_lr=base_lr,
                seed=seed,
                device=device,
            )
            row[f"k{k}_exact"] = r.exact_match_rate
            row[f"k{k}_nearest"] = r.nearest_pattern_accuracy
            row[f"k{k}_bits"] = r.per_bit_accuracy
            row[f"k{k}_conv_rate"] = r.convergence.convergence_rate
            row[f"k{k}_conv_iter"] = r.convergence.mean_convergence_iter

        results.append(row)
        print(
            f"  N={n:3d}: "
            + " | ".join(
                f"K={k}: exact={row[f'k{k}_exact']:.2f}, "
                f"near={row[f'k{k}_nearest']:.2f}"
                for k in k_values
            ),
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# MNIST experiments
# ---------------------------------------------------------------------------


def run_mnist_recall(
    n_patterns: int = 50,
    noise_type: str = "gaussian",
    noise_level: float = 0.5,
    occlusion_frac: float = 0.3,
    n_iters: int = 50,
    *,
    n_associators: int = 16,
    n_integrators: int = 8,
    n_presentations: int = 3,
    hebbian_lr: float = 0.001,
    base_lr: float = 0.0001,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> MNISTRecallResult:
    """Store N MNIST digits in SOMA, recall from noisy probes."""
    ds = make_mnist_recall(
        n_patterns=n_patterns,
        noise_type=noise_type,  # type: ignore[arg-type]
        noise_level=noise_level,
        occlusion_frac=occlusion_frac,
        device=device,
        seed=seed,
    )

    soma = make_attractor_soma(
        pattern_dim=784,
        n_associators=n_associators,
        n_integrators=n_integrators,
        hebbian_lr=hebbian_lr,
        base_lr=base_lr,
        seed=seed,
        device=device,
    )

    # Store phase
    for i in range(n_patterns):
        store_pattern(soma, ds.patterns[i], n_presentations=n_presentations)

    # Recall phase
    total_mse = 0.0
    correct_class = 0
    total_pixel_acc = 0.0
    convergence_count = 0
    convergence_iters: list[int] = []

    for i in range(n_patterns):
        rr = recall_pattern(soma, ds.probes[i], max_iters=n_iters)

        recalled = rr.final_output
        original = ds.patterns[i]

        # MSE
        mse_i = ((recalled - original) ** 2).mean().item()
        total_mse += mse_i

        # Per-pixel accuracy (within tolerance)
        pixel_acc = (torch.abs(recalled - original) < 0.1).float().mean().item()
        total_pixel_acc += pixel_acc

        # Class accuracy: nearest stored pattern
        dists = torch.cdist(
            recalled.unsqueeze(0).unsqueeze(0),
            ds.patterns.unsqueeze(0),
        ).squeeze(0).squeeze(0)
        nearest_idx = dists.argmin().item()
        if ds.labels[nearest_idx] == ds.labels[i]:
            correct_class += 1

        if rr.converged:
            convergence_count += 1
            if rr.convergence_iter is not None:
                convergence_iters.append(rr.convergence_iter)

    n = max(n_patterns, 1)
    conv_info = ConvergenceInfo(
        n_trials=n_patterns,
        n_converged=convergence_count,
        mean_convergence_iter=(
            sum(convergence_iters) / len(convergence_iters)
            if convergence_iters
            else 0.0
        ),
        convergence_rate=convergence_count / n,
    )

    return MNISTRecallResult(
        noise_type=noise_type,
        n_patterns=n_patterns,
        n_iters=n_iters,
        mse=total_mse / n,
        class_accuracy=correct_class / n,
        per_pixel_accuracy=total_pixel_acc / n,
        convergence=conv_info,
    )


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------


def analyse_convergence(
    dim: int = 50,
    n_patterns: int = 5,
    max_iters: int = 100,
    *,
    n_associators: int = 8,
    n_integrators: int = 4,
    hebbian_lr: float = 0.01,
    base_lr: float = 0.001,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict:
    """Detailed convergence analysis: track output norm trajectory.

    Returns per-pattern output norms and convergence metrics.
    """
    ds = make_binary_patterns(
        n_patterns=n_patterns,
        dim=dim,
        mask_frac=0.2,
        device=device,
        seed=seed,
    )

    soma = make_attractor_soma(
        pattern_dim=dim,
        n_associators=n_associators,
        n_integrators=n_integrators,
        hebbian_lr=hebbian_lr,
        base_lr=base_lr,
        seed=seed,
        device=device,
    )

    # Store
    for i in range(n_patterns):
        store_pattern(soma, ds.patterns[i], n_presentations=3)

    # Track outputs per iteration for each pattern
    trajectories: list[dict] = []
    for i in range(n_patterns):
        rr = recall_pattern(
            soma, ds.probes[i], max_iters=max_iters, modality="text"
        )
        norms = [o.norm().item() for o in rr.outputs_per_iter]
        diffs = []
        for j in range(1, len(rr.outputs_per_iter)):
            d = (rr.outputs_per_iter[j] - rr.outputs_per_iter[j - 1]).norm().item()
            diffs.append(d)

        trajectories.append(
            {
                "pattern_idx": i,
                "norms": norms,
                "inter_step_diffs": diffs,
                "converged": rr.converged,
                "convergence_iter": rr.convergence_iter,
                "final_norm": norms[-1] if norms else 0.0,
            }
        )

    return {
        "n_patterns": n_patterns,
        "dim": dim,
        "max_iters": max_iters,
        "trajectories": trajectories,
    }


# ---------------------------------------------------------------------------
# Homeostasis ablation
# ---------------------------------------------------------------------------


def homeostasis_ablation(
    n_patterns: int = 10,
    dim: int = 50,
    n_iters: int = 50,
    *,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict:
    """Compare SOMA attractor with default vs. disabled homeostasis.

    'Disabled' means gain_min = gain_max = 1.0 (gain is clamped to 1).
    """
    results: dict = {}

    for label, _gain_range in [("default", (0.1, 10.0)), ("disabled", (1.0, 10.0))]:
        soma = make_attractor_soma(
            pattern_dim=dim,
            hebbian_lr=0.01,
            base_lr=0.001,
            seed=seed,
            device=device,
        )
        # Override gain bounds after construction for ablation
        if label == "disabled":
            for node in soma.graph.all_nodes():
                node.gain = 1.0
                node._gain_min = 1.0
                node._gain_max = 1.0

        ds = make_binary_patterns(
            n_patterns=n_patterns, dim=dim, mask_frac=0.2, device=device, seed=seed
        )

        # Store
        for i in range(n_patterns):
            store_pattern(soma, ds.patterns[i], n_presentations=3)

        # Recall
        exact = 0
        nearest = 0
        bit_acc_total = 0.0
        for i in range(n_patterns):
            rr = recall_pattern(soma, ds.probes[i], max_iters=n_iters)
            recalled = rr.final_output.sign()
            if torch.equal(recalled, ds.patterns[i]):
                exact += 1
            bit_acc_total += (recalled == ds.patterns[i]).float().mean().item()
            sims = torch.cosine_similarity(
                rr.final_output.unsqueeze(0), ds.patterns, dim=1
            )
            if sims.argmax().item() == i:
                nearest += 1

        results[label] = {
            "exact_match_rate": exact / n_patterns,
            "nearest_pattern_accuracy": nearest / n_patterns,
            "per_bit_accuracy": bit_acc_total / n_patterns,
        }

    return results
