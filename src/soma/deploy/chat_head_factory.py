"""Factory for tier-selected ChatHeads with device/dtype handled."""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soma.deploy.devices import MODEL_TIERS
from soma.io.chat_head import ChatHead


def build_chat_head(
    *,
    tier: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool = False,
) -> ChatHead:
    """Download/load the tier-selected HF model and wrap in a ChatHead.

    Moves the model to ``device`` with the requested ``dtype``. Validates
    that the loaded model's ``config.hidden_size`` matches the tier's
    registered ``hidden_dim`` -- guards against silent misconfiguration
    where the :data:`MODEL_TIERS` registry drifts from the real model's
    shape (e.g. a HF-side architecture change).

    Parameters
    ----------
    tier:
        One of :data:`MODEL_TIERS` keys (``"tiny"``, ``"small"``, ``"large"``).
    device:
        Target torch device; model is moved here via ``.to(device)`` after load.
    dtype:
        Target dtype; passed as ``torch_dtype=`` to ``from_pretrained`` so
        weights are materialised in the requested precision rather than
        upcast-then-downcast.
    local_files_only:
        Forwarded to ``from_pretrained``. Useful for offline / air-gapped
        dev where re-downloading is unwanted.

    Raises
    ------
    ValueError
        If ``tier`` is not in :data:`MODEL_TIERS`, or if the loaded model's
        ``hidden_size`` does not match the registry entry.
    """
    if tier not in MODEL_TIERS:
        raise ValueError(f"unknown tier {tier!r}. Valid: {sorted(MODEL_TIERS.keys())}")

    spec = MODEL_TIERS[tier]
    model_name = spec["name"]
    expected_hidden = spec["hidden_dim"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ),
    ).to(device)

    actual_hidden = int(model.config.hidden_size)
    if actual_hidden != expected_hidden:
        raise ValueError(
            f"hidden_dim mismatch for tier {tier!r}: registered "
            f"{expected_hidden}, loaded model reports {actual_hidden}. "
            "The MODEL_TIERS registry has drifted -- update devices.py."
        )

    return ChatHead(model=model, tokenizer=tokenizer)
