"""Dataset generators for associative-memory benchmarks (D1).

Two benchmarks:
  1. Random binary patterns -- classical Hopfield capacity test.
  2. MNIST denoising -- store digit images, recall from noisy queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Random binary patterns
# ---------------------------------------------------------------------------

@dataclass
class BinaryPatternSet:
    """A set of bipolar (+/-1) random patterns plus noisy probes."""

    patterns: Tensor        # (N, D) bipolar
    probes: Tensor          # (N, D) noisy versions of *patterns*
    mask: Tensor            # (N, D) bool -- True where the probe was corrupted
    n_patterns: int
    dim: int


def make_binary_patterns(
    n_patterns: int,
    dim: int = 50,
    mask_frac: float = 0.2,
    *,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> BinaryPatternSet:
    """Generate *n_patterns* random bipolar vectors and partial probes.

    Each probe is a copy of the original pattern with *mask_frac* of the
    bits zeroed out (set to 0, which is ambiguous in bipolar space).
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    patterns = torch.randint(0, 2, (n_patterns, dim), generator=gen).float() * 2 - 1
    patterns = patterns.to(device)

    # Build mask: True where we zero-out
    mask = torch.rand(n_patterns, dim, generator=gen).to(device) < mask_frac

    probes = patterns.clone()
    probes[mask] = 0.0

    return BinaryPatternSet(
        patterns=patterns,
        probes=probes,
        mask=mask,
        n_patterns=n_patterns,
        dim=dim,
    )


# ---------------------------------------------------------------------------
# MNIST denoising
# ---------------------------------------------------------------------------

@dataclass
class MNISTRecallSet:
    """Stored MNIST patterns + noisy probes for denoising recall."""

    patterns: Tensor        # (N, 784) normalised to [-1, 1]
    labels: Tensor          # (N,) int class labels
    probes: Tensor          # (N, 784) corrupted versions
    noise_type: str
    n_patterns: int


def _load_mnist_flat(
    root: str = "research/associative/data/",
    train: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return (images, labels) with images flattened to (N, 784) in [-1, 1]."""
    from torchvision import datasets  # type: ignore[import-untyped]

    ds = datasets.MNIST(root=root, train=train, download=True)
    images = ds.data.float().reshape(-1, 784) / 255.0  # [0, 1]
    images = images * 2 - 1  # [-1, 1]
    return images, ds.targets


NoiseType = Literal["gaussian", "occlusion"]


def make_mnist_recall(
    n_patterns: int = 50,
    noise_type: NoiseType = "gaussian",
    noise_level: float = 0.5,
    occlusion_frac: float = 0.3,
    *,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    root: str = "research/associative/data/",
) -> MNISTRecallSet:
    """Select *n_patterns* MNIST images and create noisy probes.

    Parameters
    ----------
    noise_type:
        ``"gaussian"`` adds N(0, noise_level) noise.
        ``"occlusion"`` zeros out *occlusion_frac* of pixels.
    """
    images, targets = _load_mnist_flat(root=root)

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    idx = torch.randperm(len(images), generator=gen)[:n_patterns]
    patterns = images[idx].to(device)
    labels = targets[idx].to(device)

    if noise_type == "gaussian":
        noise = torch.randn_like(patterns) * noise_level
        probes = (patterns + noise).clamp(-1, 1)
    elif noise_type == "occlusion":
        mask = torch.rand_like(patterns) < occlusion_frac
        probes = patterns.clone()
        probes[mask] = 0.0
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return MNISTRecallSet(
        patterns=patterns,
        labels=labels,
        probes=probes,
        noise_type=noise_type,
        n_patterns=n_patterns,
    )
