"""Shared argparse glue for SOMA chat CLIs.

``chat_repl.py`` and ``train_verbalizer_bootstrap.py`` both accept the
same ``--tier`` / ``--device`` / ``--dtype`` / ``--llm-name`` flags with
identical semantics. Centralising the flag definitions and the resolution
logic here keeps the two scripts in sync and gives the unit tests a single
helper to exercise instead of two near-duplicate CLI blocks.

The resolver intentionally accepts an :class:`argparse.Namespace` rather
than the individual strings -- the scripts have more flags than just
these four, and passing the full namespace lets us add tier-related flags
later without churning both call sites.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Literal

import torch

from soma.deploy.devices import (
    MODEL_TIERS,
    auto_select_tier,
    detect_cuda_vram,
    select_device_and_dtype,
)

TIER_CHOICES = ["auto", "tiny", "small", "large"]
DTYPE_CHOICES = ["auto", "fp32", "fp16"]
Backend = Literal["hf", "gguf"]


def add_deploy_arguments(parser: argparse.ArgumentParser) -> None:
    """Register --llm-name / --tier / --device / --dtype on ``parser``.

    All flags default to ``None`` or ``"auto"`` so existing invocations
    that pass ``--llm-name foo`` keep working (backward compat), while a
    zero-config ``--tier auto`` path becomes available for new operators.
    """
    parser.add_argument(
        "--llm-name",
        type=str,
        default=None,
        help="HuggingFace model name. If omitted, --tier selects the model.",
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=TIER_CHOICES,
        default="auto",
        help=(
            "Model tier to use when --llm-name is not set. 'auto' picks based "
            "on detected CUDA VRAM (see soma.deploy.MODEL_TIERS)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Torch device string. 'auto' picks cuda if available, else cpu. "
            "Note: cuda now defaults to fp16 (Phase 7). Pass --dtype fp32 to "
            "restore the pre-Phase-7 fp32-on-cuda behavior."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=DTYPE_CHOICES,
        default="auto",
        help="Model dtype override. 'auto' pairs with --device auto (cuda->fp16, cpu->fp32).",
    )
    parser.add_argument(
        "--gguf-path",
        type=Path,
        default=None,
        help=(
            "Path to a local GGUF file (e.g., from LM Studio cache). Inference-"
            "only -- cannot be used with verbalizer training. Overrides --tier "
            "and --llm-name."
        ),
    )


def resolve_device_dtype_tier(
    args: argparse.Namespace,
) -> tuple[torch.device, torch.dtype, str, str | None]:
    """Turn parsed ``args`` into concrete ``(device, dtype, llm_name, tier)``.

    ``tier`` is ``None`` when the caller passed ``--llm-name`` explicitly
    (backward-compat path); otherwise it's the resolved tier key in
    :data:`MODEL_TIERS`. Explicit ``--llm-name`` wins over ``--tier``.
    """
    # Device resolution.
    if args.device == "auto":
        device, auto_dtype = select_device_and_dtype()
    else:
        device = torch.device(args.device)
        auto_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Dtype resolution.
    if args.dtype == "auto":
        dtype = auto_dtype
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # fp16 on CPU is typically SLOWER than fp32 (no AVX-512-fp16 on most
    # consumer CPUs). Warn operators who explicitly requested this combo
    # instead of silently accepting a footgun -- but don't reject, since
    # a benchmark or memory-shape test might legitimately want it.
    if dtype == torch.float16 and device.type == "cpu":
        warnings.warn(
            "--dtype fp16 with --device cpu is typically slower than fp32 on "
            "consumer CPUs that lack AVX-512-fp16. Prefer --dtype fp32 on CPU, "
            "or --device cuda for fp16.",
            stacklevel=2,
        )

    # Tier / llm-name resolution. Explicit --llm-name beats any --tier.
    if args.llm_name is not None:
        return device, dtype, args.llm_name, None

    tier = auto_select_tier(vram_gb=detect_cuda_vram()) if args.tier == "auto" else args.tier
    llm_name = MODEL_TIERS[tier]["name"]
    return device, dtype, llm_name, tier


def resolve_backend(args: argparse.Namespace) -> Backend:
    """Pick the inference backend implied by ``args``.

    Returns ``"gguf"`` when the operator passed ``--gguf-path``, else
    ``"hf"`` (the default Phase-7 path that downloads HF safetensors).

    The GGUF backend is inference-only and bypasses the tier registry --
    ``--gguf-path`` therefore wins over ``--tier`` and ``--llm-name``.
    Callers that need a training-capable LLM should refuse the ``"gguf"``
    backend explicitly (see ``GGUFChatHead.supports_gradients``).
    """
    if getattr(args, "gguf_path", None) is not None:
        return "gguf"
    return "hf"


def dtype_label(dtype: torch.dtype) -> str:
    """Short human label for a torch dtype ("fp16" / "fp32" / raw repr)."""
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


def print_selection(
    *,
    llm_name: str,
    tier: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """One-line report of the resolved tier/device/dtype, to stdout.

    Two forms: the tier-auto line ("Selected tier=small (Qwen/...), ...")
    and the explicit-llm-name line ("Using explicit --llm-name=..., ...").
    Kept small on purpose -- operators want to see it next to normal
    ``print()`` output, not buried in logs.
    """
    label = dtype_label(dtype)
    if tier is None:
        print(f"Using explicit --llm-name={llm_name}, device={device.type}, dtype={label}")
    else:
        print(f"Selected tier={tier} ({llm_name}), device={device.type}, dtype={label}")
