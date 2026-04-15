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

import torch

from soma.deploy.devices import (
    MODEL_TIERS,
    auto_select_tier,
    detect_cuda_vram,
    effective_vram_gb,
    select_device_and_dtype,
)

TIER_CHOICES = ["auto", "tiny", "small", "large", "xlarge"]
DTYPE_CHOICES = ["auto", "fp32", "fp16"]

# Default safety factor for the deploy CLI. Mirrors
# ``SOMAConfig.vram_safety_factor`` but is duplicated here because the
# CLI doesn't own a SOMAConfig (the verbalizer/REPL scripts construct
# their own config later). Keep this in sync with the dataclass default;
# the ``test_cli_default_vram_safety_factor_matches_config`` test guards
# it so a future config-default change can't silently drift.
DEFAULT_VRAM_SAFETY_FACTOR = 0.9


def _parse_vram_safety_factor(raw: str) -> float:
    """argparse type-converter for ``--vram-safety-factor``.

    Wraps the float() conversion so we can reject 0.0 / negative / >1.0
    values at parse time with a friendly ``argparse`` error instead of
    letting them through and crashing later in :func:`effective_vram_gb`.
    """
    try:
        value = float(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid float: {raw!r}") from e
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"--vram-safety-factor must be in (0, 1], got {value!r}")
    return value


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
        "--vram-safety-factor",
        type=_parse_vram_safety_factor,
        default=DEFAULT_VRAM_SAFETY_FACTOR,
        help=(
            "Headroom factor applied to detected VRAM before tier selection. "
            "0.9 (default) reserves 10%% for OS/driver/KV-cache jitter. Set to "
            "1.0 for no headroom; set lower (e.g. 0.7) when SOMA's own footprint "
            "is large. Must be in (0, 1]."
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

    if args.tier == "auto":
        # Apply the headroom factor before threshold comparison. Factor lives
        # on the namespace if the operator passed --vram-safety-factor;
        # otherwise we use DEFAULT_VRAM_SAFETY_FACTOR (which mirrors the
        # SOMAConfig default of 0.9). Older callers built before the flag
        # existed might not have it on their namespace -- fall back to
        # default in that case rather than erroring, so the resolver stays
        # backward-compatible with hand-built Namespace objects in tests.
        safety_factor = getattr(args, "vram_safety_factor", DEFAULT_VRAM_SAFETY_FACTOR)
        effective = effective_vram_gb(vram_gb=detect_cuda_vram(), safety_factor=safety_factor)
        tier = auto_select_tier(vram_gb=effective)
    else:
        tier = args.tier
    llm_name = MODEL_TIERS[tier]["name"]
    return device, dtype, llm_name, tier


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
