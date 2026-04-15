"""Device + model-tier selection for SOMA consumer deployment.

Detects available CUDA VRAM, picks a sensible (device, dtype) pair, and
maps VRAM tier to an HF model identifier. Three tiers only -- tiny (CPU /
low-end GPU), small (8GB-class GPU), large (16GB+).
"""

from __future__ import annotations

from typing import TypedDict

import torch


class ModelTierSpec(TypedDict):
    name: str
    hidden_dim: int
    min_vram_gb: int
    approx_fp16_gb: float


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
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_dim": 3584,
        "min_vram_gb": 16,
        "approx_fp16_gb": 14.5,
    },
}


def detect_cuda_vram() -> int | None:
    """Total VRAM of CUDA device 0, in GB (integer). None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return int(total_bytes / (1024**3))


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """CUDA+fp16 when available; CPU+fp32 otherwise.

    fp16 on CPU is skipped intentionally -- PyTorch's CPU fp16 matmul is
    typically slower than fp32 on consumer CPUs without AVX-512-fp16.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def auto_select_tier(*, vram_gb: int | None) -> str:
    """Pick the best tier that comfortably fits the available VRAM."""
    if vram_gb is None or vram_gb < 4:
        return "tiny"
    if vram_gb < 16:
        return "small"
    return "large"
