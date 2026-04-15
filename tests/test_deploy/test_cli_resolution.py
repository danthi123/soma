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

from soma.deploy.cli import (
    DTYPE_CHOICES,
    TIER_CHOICES,
    add_deploy_arguments,
    dtype_label,
    print_selection,
    resolve_backend,
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
