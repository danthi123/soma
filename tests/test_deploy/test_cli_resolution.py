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

from soma.core.config import SOMAConfig
from soma.deploy.cli import (
    DEFAULT_VRAM_SAFETY_FACTOR,
    DTYPE_CHOICES,
    TIER_CHOICES,
    add_deploy_arguments,
    dtype_label,
    print_selection,
    resolve_device_dtype_tier,
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


# ----- argparse defaults ----------------------------------------------------


def test_parser_defaults_are_all_auto_or_none() -> None:
    args = _build_test_parser().parse_args([])
    assert args.llm_name is None
    assert args.tier == "auto"
    assert args.device == "auto"
    assert args.dtype == "auto"


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


# ----- vram_safety_factor flag --------------------------------------------


def test_cli_default_vram_safety_factor_matches_config() -> None:
    """The CLI default and the SOMAConfig default must stay in sync.

    The CLI duplicates the value (DEFAULT_VRAM_SAFETY_FACTOR) because it
    doesn't own a SOMAConfig at parse time. This test catches drift if
    one default changes without the other -- otherwise operators would
    see different headroom behavior depending on whether they go via
    chat_repl (CLI default) or load a config (SOMAConfig default)."""
    assert SOMAConfig().vram_safety_factor == pytest.approx(DEFAULT_VRAM_SAFETY_FACTOR)


def test_parser_default_vram_safety_factor() -> None:
    args = _build_test_parser().parse_args([])
    assert args.vram_safety_factor == pytest.approx(DEFAULT_VRAM_SAFETY_FACTOR)


def test_parser_accepts_explicit_vram_safety_factor() -> None:
    args = _build_test_parser().parse_args(["--vram-safety-factor", "0.7"])
    assert args.vram_safety_factor == pytest.approx(0.7)


def test_parser_accepts_factor_one() -> None:
    """factor=1.0 is valid (means 'use full VRAM')."""
    args = _build_test_parser().parse_args(["--vram-safety-factor", "1.0"])
    assert args.vram_safety_factor == pytest.approx(1.0)


@pytest.mark.parametrize("bad", ["0.0", "-0.1", "1.5", "2.0"])
def test_parser_rejects_out_of_range_vram_safety_factor(bad: str) -> None:
    """Out-of-range values must error at argparse time, not propagate."""
    with pytest.raises(SystemExit):
        _build_test_parser().parse_args(["--vram-safety-factor", bad])


def test_parser_rejects_non_numeric_vram_safety_factor() -> None:
    with pytest.raises(SystemExit):
        _build_test_parser().parse_args(["--vram-safety-factor", "garbage"])


def test_resolve_applies_vram_safety_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Headroom factor must be applied to detected VRAM before tier
    selection. A 23 GB raw card with factor=0.9 -> 20 GB effective ->
    'large' tier, NOT xlarge (which needs 24 effective GB)."""

    class _FakeProps:
        total_memory = int(23.64 * 1024**3)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: _FakeProps())
    args = _build_test_parser().parse_args(["--device", "cuda"])
    _device, _dtype, llm_name, tier = resolve_device_dtype_tier(args)
    assert tier == "large"
    assert llm_name == MODEL_TIERS["large"]["name"]


def test_resolve_factor_one_picks_xlarge_at_24gb(monkeypatch: pytest.MonkeyPatch) -> None:
    """factor=1.0 disables headroom: 24 GB raw -> 24 effective -> xlarge."""

    class _FakeProps:
        total_memory = 24 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: _FakeProps())
    args = _build_test_parser().parse_args(["--device", "cuda", "--vram-safety-factor", "1.0"])
    _device, _dtype, llm_name, tier = resolve_device_dtype_tier(args)
    assert tier == "xlarge"
    assert llm_name == MODEL_TIERS["xlarge"]["name"]


def test_resolve_aggressive_factor_demotes_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """factor=0.5 on a 24 GB card -> 12 effective -> still 'large' (12 is the
    boundary). Operator can demote further with a smaller factor or by
    forcing --tier explicitly."""

    class _FakeProps:
        total_memory = 24 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: _FakeProps())
    args = _build_test_parser().parse_args(["--device", "cuda", "--vram-safety-factor", "0.5"])
    _device, _dtype, _llm_name, tier = resolve_device_dtype_tier(args)
    assert tier == "large"


def test_resolve_explicit_tier_ignores_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """When --tier is explicit, the factor is irrelevant -- the operator
    has overridden tier selection entirely."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = _build_test_parser().parse_args(["--tier", "xlarge", "--vram-safety-factor", "0.1"])
    _device, _dtype, llm_name, tier = resolve_device_dtype_tier(args)
    assert tier == "xlarge"
    assert llm_name == MODEL_TIERS["xlarge"]["name"]


def test_resolve_works_without_vram_safety_factor_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compat: a hand-built Namespace without vram_safety_factor
    must still resolve. Tests the getattr() default fallback in
    resolve_device_dtype_tier."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = argparse.Namespace(llm_name=None, tier="auto", device="auto", dtype="auto")
    _device, _dtype, llm_name, tier = resolve_device_dtype_tier(args)
    # No CUDA -> tiny regardless of factor. The point is the resolver
    # didn't crash on a missing attribute.
    assert tier == "tiny"
    assert llm_name == MODEL_TIERS["tiny"]["name"]
