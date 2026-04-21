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
from pathlib import Path

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.deploy.cli import (
    DEFAULT_VRAM_SAFETY_FACTOR,
    DTYPE_CHOICES,
    QUANT_CHOICES,
    TIER_CHOICES,
    add_deploy_arguments,
    dtype_label,
    print_selection,
    resolve_backend,
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
    # The default ``--tier auto`` path calls ``detect_cuda_vram()`` which
    # hits ``torch.cuda.get_device_properties`` — that goes through to the
    # real driver even when ``is_available`` is monkeypatched, so the test
    # needs to stub it too. Returning a fake "24 GB" lets the tier resolver
    # complete without touching the GPU.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _idx: type("FakeProps", (), {"total_memory": 24 * 1024**3})(),
    )
    args = _build_test_parser().parse_args(["--device", "cpu"])
    device, dtype, _llm_name, _tier = resolve_device_dtype_tier(args)
    assert device.type == "cpu"
    # Device is cpu, so auto-dtype is fp32 even though CUDA exists.
    assert dtype == torch.float32


def test_resolve_dtype_override_fp32_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operator forces fp32 on GPU for debugging: dtype overrides auto-fp16."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # Same reason as ``test_resolve_device_explicit_cpu``: the resolver's
    # auto-tier branch wants ``get_device_properties`` to answer, even
    # though this test only cares about device+dtype resolution.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _idx: type("FakeProps", (), {"total_memory": 24 * 1024**3})(),
    )
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


# ----- gguf backend flag ---------------------------------------------------


def test_parser_default_gguf_path_is_none() -> None:
    """No --gguf-path flag -> args.gguf_path is None (HF backend default)."""
    args = _build_test_parser().parse_args([])
    assert args.gguf_path is None


def test_parser_accepts_gguf_path_as_path() -> None:
    """--gguf-path arg is parsed into a pathlib.Path, not a raw str."""
    args = _build_test_parser().parse_args(["--gguf-path", "/some/file.gguf"])
    assert isinstance(args.gguf_path, Path)
    assert args.gguf_path == Path("/some/file.gguf")


def test_resolve_backend_default_is_hf() -> None:
    """No --gguf-path -> 'hf' backend (preserves Phase-7 default behavior)."""
    args = _build_test_parser().parse_args([])
    assert resolve_backend(args) == "hf"


def test_resolve_backend_with_gguf_path_is_gguf() -> None:
    """--gguf-path X -> 'gguf' backend, regardless of other flags."""
    args = _build_test_parser().parse_args(["--gguf-path", "/some/file.gguf"])
    assert resolve_backend(args) == "gguf"


def test_resolve_backend_gguf_overrides_tier_and_llm_name() -> None:
    """--gguf-path wins even when the operator also passed --tier/--llm-name.

    Documents the precedence rule: the GGUF path is the most-specific
    backend selector, so it short-circuits the registry lookup. We don't
    raise on the combination -- operators may want to keep the tier flag
    set to its default while overriding the actual backend for one run.
    """
    args = _build_test_parser().parse_args(
        [
            "--tier",
            "small",
            "--llm-name",
            "foo/bar",
            "--gguf-path",
            "/some/file.gguf",
        ]
    )
    assert resolve_backend(args) == "gguf"


def test_resolve_backend_does_not_check_file_existence() -> None:
    """The resolver is a pure flag inspector -- it does NOT touch the disk.

    File-existence checking is the GGUF factory's job (FileNotFoundError).
    Keeping the resolver pure lets argparse-level tests run without
    creating temp files.
    """
    args = _build_test_parser().parse_args(["--gguf-path", "/nonexistent/file.gguf"])
    assert resolve_backend(args) == "gguf"
