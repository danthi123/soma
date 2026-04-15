"""Factory for tier-selected ChatHeads with device/dtype handled."""

from __future__ import annotations

from typing import Any, Literal, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soma.deploy.devices import MODEL_TIERS
from soma.io.chat_head import ChatHead

# bitsandbytes is an OPTIONAL dependency (extras: ".[quant]"). Importing
# ``BitsAndBytesConfig`` from transformers does not itself require bnb to be
# installed, but using it at ``from_pretrained`` time will raise inside
# transformers when bnb is missing. We track availability via a real bnb
# import so we can give a friendlier error message at the factory boundary.
try:
    import bitsandbytes  # noqa: F401 -- presence check only

    _HAS_BITSANDBYTES = True
except ImportError:
    _HAS_BITSANDBYTES = False

QuantMode = Literal["none", "int8", "int4"]


def _build_bnb_config(quantization: QuantMode, compute_dtype: torch.dtype) -> Any:
    """Construct a ``BitsAndBytesConfig`` for the requested mode.

    Imported lazily so callers using ``quantization="none"`` never touch the
    bnb-aware code path. ``compute_dtype`` only matters for int4 (the dtype
    in which the dequantised weights are matmul'd); int8 always computes in
    fp16 / bf16 internally.
    """
    from transformers import BitsAndBytesConfig

    # transformers is mypy-ignored, so BitsAndBytesConfig resolves to a typed
    # stub via its __init__ rather than Any -- cast keeps the kwargs
    # untouched but escapes the strict-call check.
    bnb_config = cast(Any, BitsAndBytesConfig)
    if quantization == "int4":
        return bnb_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    # int8
    return bnb_config(load_in_8bit=True)


def build_chat_head(
    *,
    tier: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool = False,
    quantization: QuantMode = "none",
) -> ChatHead:
    """Download/load the tier-selected HF model and wrap in a ChatHead.

    For ``quantization="none"`` (default), moves the model to ``device`` with
    the requested ``dtype``. For int4 / int8, defers placement to bitsandbytes
    via ``device_map={"": device.type}`` -- bnb does its own CUDA placement
    during weight loading and a post-hoc ``.to(device)`` would either no-op
    or break the quantised buffers. Validates that the loaded model's
    ``config.hidden_size`` matches the tier's registered ``hidden_dim`` --
    guards against silent misconfiguration where the :data:`MODEL_TIERS`
    registry drifts from the real model's shape (e.g. a HF-side architecture
    change).

    Parameters
    ----------
    tier:
        One of :data:`MODEL_TIERS` keys (``"tiny"``, ``"small"``, ``"large"``).
    device:
        Target torch device; model is moved here via ``.to(device)`` after load
        (or via ``device_map`` when quantising).
    dtype:
        Target dtype; passed as ``torch_dtype=`` to ``from_pretrained`` so
        weights are materialised in the requested precision rather than
        upcast-then-downcast. With ``quantization="int4"`` this becomes the
        compute dtype for the dequantised matmuls.
    local_files_only:
        Forwarded to ``from_pretrained``. Useful for offline / air-gapped
        dev where re-downloading is unwanted.
    quantization:
        One of ``"none"``, ``"int8"``, ``"int4"``. The two non-default modes
        require the optional ``bitsandbytes`` dependency (install via
        ``pip install -e ".[quant]"``) and a CUDA device. int4 with the
        default ``nf4`` quant type fits a 7B model in ~5 GB of VRAM while
        keeping ``model.get_input_embeddings()`` in compute_dtype, so the
        verbalizer's gradient flow through input embeddings is preserved.

    Raises
    ------
    ValueError
        If ``tier`` is not in :data:`MODEL_TIERS`, if the loaded model's
        ``hidden_size`` does not match the registry entry, or if
        ``quantization != "none"`` is requested on a non-CUDA device.
    ImportError
        If ``quantization != "none"`` and ``bitsandbytes`` is not installed.
    """
    if tier not in MODEL_TIERS:
        raise ValueError(f"unknown tier {tier!r}. Valid: {sorted(MODEL_TIERS.keys())}")

    if quantization != "none":
        if device.type == "cpu":
            raise ValueError(
                f"quantization={quantization!r} requires a CUDA device "
                "(bitsandbytes is CUDA-only). Use device='cuda' or "
                "quantization='none' for CPU inference."
            )
        if not _HAS_BITSANDBYTES:
            raise ImportError(
                f"quantization={quantization!r} requires the optional "
                "'bitsandbytes' dependency. Install via "
                '`pip install -e ".[quant]"`. Note: Windows wheels are '
                "historically flaky -- WSL is the recommended path."
            )

    spec = MODEL_TIERS[tier]
    model_name = spec["name"]
    expected_hidden = spec["hidden_dim"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )

    from_pretrained_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if quantization == "none":
        model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs),
        ).to(device)
    else:
        # bnb places weights itself during load; do NOT call .to(device) afterward.
        # device_map={"": device.type} pins everything to a single device (the
        # alternative "auto" can shard across GPUs which we do not want here).
        from_pretrained_kwargs["quantization_config"] = _build_bnb_config(quantization, dtype)
        from_pretrained_kwargs["device_map"] = {"": device.type}
        model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs),
        )

    actual_hidden = int(model.config.hidden_size)
    if actual_hidden != expected_hidden:
        raise ValueError(
            f"hidden_dim mismatch for tier {tier!r}: registered "
            f"{expected_hidden}, loaded model reports {actual_hidden}. "
            "The MODEL_TIERS registry has drifted -- update devices.py."
        )

    return ChatHead(model=model, tokenizer=tokenizer)
