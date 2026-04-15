"""Unit tests for the shared deploy-CLI resolver.

Covers the extracted helpers in :mod:`soma.deploy.cli` and the argparse
wiring in both scripts: defaults, explicit-llm-name backward compat,
explicit tier selection, dtype overrides, and rejection of bad values.

No HF downloads -- :func:`resolve_device_dtype_tier` is a pure dict
lookup once device/dtype are chosen, and :data:`MODEL_TIERS` is used as
ground truth for the tier -> model-name mapping.
"""

from __future__ import annotations

import argparse

import pytest
import torch

from soma.deploy.cli import (
    DTYPE_CHOICES,
    QUANT_CHOICES,
    TIER_CHOICES,
    add_deploy_arguments,
    dtype_label,
    print_selection,
    resolve_device_dtype_tier,
    resolve_quantization,
)
from soma.deploy.devices import MODEL_TIERS


def _build_test_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_deploy_arguments(parser)
    return parser


# ----- choices advertised ---------------------------------------------------


def test_tier_choices_match_model_tiers_plus_auto() -> None:
    assert set(TIER_CHOICES) == set(MODEL_TIERS.keys()) | {"auto"}


def test_dtype_choices_are_auto_fp32_fp16() -> None:
    assert set(DTYPE_CHOICES) == {"auto", "fp32", "fp16"}


def test_quant_choices_are_none_int8_int4() -> None:
    assert set(QUANT_CHOICES) == {"none", "int8", "int4"}


# ----- argparse defaults ----------------------------------------------------


def test_parser_defaults_are_all_auto_or_none() -> None:
    args = _build_test_parser().parse_args([])
    assert args.llm_name is None
    assert args.tier == "auto"
    assert args.device == "auto"
    assert args.dtype == "auto"
    assert args.quantization == "none"


def test_parser_accepts_explicit_tier() -> None:
    args = _build_test_parser().parse_args(["--tier", "small"])
    assert args.tier == "small"


def test_parser_accepts_explicit_dtype_fp16() -> None:
    args = _build_test_parser().parse_args(["--dtype", "fp16"])
    assert args.dtype == "fp16"


def test_parser_accepts_explicit_llm_name() -> None:
    args = _build_test_parser().parse_args(["--llm-name", "foo/bar"])
    assert args.llm_name == "foo/bar"


def test_parser_rejects_unknown_tier() -> None:
    with pytest.raises(SystemExit):
        _build_test_parser().parse_args(["--tier", "nonsense"])


def test_parser_rejects_unknown_dtype() -> None:
    with pytest.raises(SystemExit):
        _build_test_parser().parse_args(["--dtype", "bf16"])


def test_parser_accepts_explicit_quantization_int4() -> None:
    args = _build_test_parser().parse_args(["--quantization", "int4"])
    assert args.quantization == "int4"
    assert resolve_quantization(args) == "int4"


def test_parser_accepts_explicit_quantization_int8() -> None:
    args = _build_test_parser().parse_args(["--quantization", "int8"])
    assert args.quantization == "int8"
    assert resolve_quantization(args) == "int8"


def test_parser_rejects_unknown_quantization() -> None:
    with pytest.raises(SystemExit):
        _build_test_parser().parse_args(["--quantization", "fp4"])


def test_resolve_quantization_default_is_none() -> None:
    args = _build_test_parser().parse_args([])
    assert resolve_quantization(args) == "none"


# ----- resolver: device / dtype pairings -----------------------------------


def test_resolve_device_auto_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """--device auto with no CUDA -> cpu + fp32."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = _build_test_parser().parse_args([])
    device, dtype, llm_name, tier = resolve_device_dtype_tier(args)
    assert device.type == "cpu"
    assert dtype == torch.float32
    # auto-tier + no CUDA -> tiny
    assert tier == "tiny"
    assert llm_name == MODEL_TIERS["tiny"]["name"]


def test_resolve_device_explicit_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """--device cpu keeps working even when CUDA is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = _build_test_parser().parse_args(["--device", "cpu"])
    device, dtype, _llm_name, _tier = resolve_device_dtype_tier(args)
    assert device.type == "cpu"
    # Device is cpu, so auto-dtype is fp32 even though CUDA exists.
    assert dtype == torch.float32


def test_resolve_dtype_override_fp32_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator forces fp32 on GPU for debugging: dtype overrides auto-fp16."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = _build_test_parser().parse_args(["--device", "cuda", "--dtype", "fp32"])
    device, dtype, _llm_name, _tier = resolve_device_dtype_tier(args)
    assert device.type == "cuda"
    assert dtype == torch.float32


def test_resolve_dtype_override_fp16_on_cpu_warns() -> None:
    """fp16+cpu is accepted but must warn -- it's a real footgun on most
    consumer CPUs (fp16 matmul without AVX-512-fp16 is slower than fp32),
    so silent acceptance would hide a performance trap from benchmarks."""
    args = _build_test_parser().parse_args(["--device", "cpu", "--dtype", "fp16"])
    with pytest.warns(UserWarning, match="fp16.*cpu"):
        _device, dtype, _llm_name, _tier = resolve_device_dtype_tier(args)
    assert dtype == torch.float16


# ----- resolver: tier / llm-name pairings ----------------------------------


def test_resolve_explicit_tier_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = _build_test_parser().parse_args(["--tier", "small"])
    _device, _dtype, llm_name, tier = resolve_device_dtype_tier(args)
    assert tier == "small"
    assert llm_name == MODEL_TIERS["small"]["name"]


def test_resolve_explicit_llm_name_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit --llm-name should win even when --tier is also set."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = _build_test_parser().parse_args(["--tier", "small", "--llm-name", "my-org/my-model"])
    _device, _dtype, llm_name, tier = resolve_device_dtype_tier(args)
    assert tier is None  # sentinel: explicit-name path
    assert llm_name == "my-org/my-model"


def test_resolve_auto_tier_with_no_cuda_picks_tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = _build_test_parser().parse_args([])
    _device, _dtype, llm_name, tier = resolve_device_dtype_tier(args)
    assert tier == "tiny"
    assert llm_name == MODEL_TIERS["tiny"]["name"]


# ----- tiny helpers --------------------------------------------------------


def test_dtype_label_known_values() -> None:
    assert dtype_label(torch.float16) == "fp16"
    assert dtype_label(torch.float32) == "fp32"


def test_print_selection_explicit_name(capsys: pytest.CaptureFixture[str]) -> None:
    print_selection(
        llm_name="foo/bar",
        tier=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    out = capsys.readouterr().out
    assert "Using explicit --llm-name=foo/bar" in out
    assert "device=cpu" in out
    assert "dtype=fp32" in out


def test_print_selection_tier(capsys: pytest.CaptureFixture[str]) -> None:
    print_selection(
        llm_name=MODEL_TIERS["tiny"]["name"],
        tier="tiny",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    out = capsys.readouterr().out
    assert "Selected tier=tiny" in out
    assert MODEL_TIERS["tiny"]["name"] in out
    # quantization defaults to "none" -> suppressed from output
    assert "quant=" not in out


def test_print_selection_with_quantization(capsys: pytest.CaptureFixture[str]) -> None:
    """Operators must see ``quant=int4`` so it's clear bnb is active."""
    print_selection(
        llm_name=MODEL_TIERS["large"]["name"],
        tier="large",
        device=torch.device("cuda"),
        dtype=torch.float16,
        quantization="int4",
    )
    out = capsys.readouterr().out
    assert "quant=int4" in out


# ----- script-level argparse wiring ----------------------------------------


def test_chat_repl_parser_defaults() -> None:
    """chat_repl --soma-checkpoint X -> --tier auto, --device auto, --dtype auto."""
    from scripts.chat_repl import _build_parser

    args = _build_parser().parse_args(["--soma-checkpoint", "some/bundle"])
    assert args.llm_name is None
    assert args.tier == "auto"
    assert args.device == "auto"
    assert args.dtype == "auto"


def test_chat_repl_parser_rejects_unknown_tier() -> None:
    from scripts.chat_repl import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--soma-checkpoint", "some/bundle", "--tier", "nonsense"])


def test_train_verbalizer_bootstrap_parser_defaults() -> None:
    from scripts.train_verbalizer_bootstrap import _build_parser

    args = _build_parser().parse_args(
        [
            "--soma-checkpoint",
            "some/bundle",
            "--corpus",
            "some.txt",
            "--out-dir",
            "some/out",
        ]
    )
    assert args.llm_name is None
    assert args.tier == "auto"
    assert args.device == "auto"
    assert args.dtype == "auto"


def test_train_verbalizer_bootstrap_parser_rejects_unknown_dtype() -> None:
    from scripts.train_verbalizer_bootstrap import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "--soma-checkpoint",
                "some/bundle",
                "--corpus",
                "some.txt",
                "--out-dir",
                "some/out",
                "--dtype",
                "bf16",
            ]
        )
