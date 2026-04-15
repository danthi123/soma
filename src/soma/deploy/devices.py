"""Device + model-tier selection for SOMA consumer deployment.

Detects available CUDA VRAM, picks a sensible (device, dtype) pair, and
maps VRAM tier to an HF model identifier. Four tiers: tiny (CPU /
low-end GPU), small (8GB-class GPU), large (16GB+ Gemma-4-E4B-it),
xlarge (24GB+ Qwen3.5-9B).
"""

from __future__ import annotations

from typing import TypedDict

import torch


class ModelTierSpec(TypedDict):
    name: str
    hidden_dim: int
    min_vram_gb: int
    approx_fp16_gb: float


# Tier rationale (2026-04-14):
# - tiny: SmolLM2-360M-Instruct, the smallest sane chat-tuned model.
# - small: Qwen2.5-1.5B-Instruct, the canonical "fits in 4GB fp16" model.
# - large: Gemma-4-E4B-it. Matformer/Gemma-3n-lineage; effective 4B active
#   params with ~8B total weights, ~16GB fp16. Multimodal (vision+audio
#   via mmproj). Fits 24GB with comfortable headroom; chosen as the
#   default 16-23GB-band model because the smaller hidden_dim (2560 vs
#   Qwen3.5-9B's 4096) and active-param-sparse Matformer design give a
#   better quality-per-VRAM ratio in this band.
# - xlarge: Qwen3.5-9B, dense ~9B param multimodal model, ~18GB fp16.
#   Needs the full 24GB to leave room for KV cache + SOMA prefix tensors.
#
# The 16/24 GB boundary is a deliberate trade: anyone with a 16-23GB card
# (4080, A4000, 3090-Ti-laptop) gets the more-headroom Gemma; anyone with
# a true 24GB+ card (3090, 4090, A5000) defaults up to dense Qwen3.5-9B.
MODEL_TIERS: dict[str, ModelTierSpec] = {
    "tiny": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "hidden_dim": 960,
        "min_vram_gb": 0,
        "approx_fp16_gb": 0.72,
    },
    "small": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "min_vram_gb": 4,
        "approx_fp16_gb": 3.1,
    },
    "large": {
        "name": "google/gemma-4-E4B-it",
        "hidden_dim": 2560,
        "min_vram_gb": 12,
        "approx_fp16_gb": 16.0,
    },
    "xlarge": {
        "name": "Qwen/Qwen3.5-9B",
        "hidden_dim": 4096,
        "min_vram_gb": 24,
        "approx_fp16_gb": 18.0,
    },
}


def detect_cuda_vram() -> int | None:
    """Total VRAM of CUDA device 0, in GB. ``None`` when no CUDA.

    Returns the floor of the GB figure: a 7.94 GB card reports ``7``,
    a 23.64 GB RTX 3090 reports ``23``. Downstream ``auto_select_tier``
    no longer bakes in any cushion of its own; callers are expected to
    apply :func:`effective_vram_gb` (with ``SOMAConfig.vram_safety_factor``)
    before threshold comparison so the headroom policy is operator-tunable.
    """
    if not torch.cuda.is_available():
        return None
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return int(total_bytes / (1024**3))


def effective_vram_gb(*, vram_gb: int | None, safety_factor: float) -> int | None:
    """Apply the operator-tunable headroom factor to a raw VRAM figure.

    ``safety_factor=0.9`` (the SOMAConfig default) means "reserve 10% for
    OS/driver/KV-cache jitter": a 23 GB raw report becomes 20 effective GB.
    Pass ``safety_factor=1.0`` to disable headroom (use full VRAM).

    Returns ``None`` when ``vram_gb`` is ``None`` (no CUDA) so callers can
    keep the no-CUDA branch unchanged.
    """
    if vram_gb is None:
        return None
    return int(vram_gb * safety_factor)


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """CUDA+fp16 when available; CPU+fp32 otherwise.

    fp16 on CPU is skipped intentionally -- PyTorch's CPU fp16 matmul is
    typically slower than fp32 on consumer CPUs without AVX-512-fp16.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def auto_select_tier(*, vram_gb: int | None) -> str:
    """Pick the best tier that comfortably fits the available VRAM.

    Thresholds are applied to the *effective* VRAM (post safety-factor),
    not the raw card capacity. See :func:`effective_vram_gb`.
    """
    if vram_gb is None or vram_gb < 4:
        return "tiny"
    if vram_gb < 12:
        return "small"
    if vram_gb < 24:
        return "large"
    return "xlarge"
