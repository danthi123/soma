import pytest
import torch

from soma.deploy.devices import (
    MODEL_TIERS,
    auto_select_tier,
    detect_cuda_vram,
    effective_vram_gb,
    select_device_and_dtype,
)


def test_model_tiers_has_four_entries():
    assert set(MODEL_TIERS.keys()) == {"tiny", "small", "large", "xlarge"}
    for _tier_name, spec in MODEL_TIERS.items():
        assert "name" in spec
        assert "hidden_dim" in spec
        assert "min_vram_gb" in spec
        assert "approx_fp16_gb" in spec


def test_model_tiers_vram_monotonic():
    """Larger tiers should require more VRAM."""
    tiers = ["tiny", "small", "large", "xlarge"]
    vram = [MODEL_TIERS[t]["min_vram_gb"] for t in tiers]
    assert vram == sorted(vram)


def test_model_tiers_hidden_dims_match_hf_configs():
    """Hidden dims must match HF config.json -- guards against silent
    typo'd registry changes that would cause the verbalizer projector to
    be built with the wrong shape and crash at first forward pass."""
    # SmolLM2-360M-Instruct
    assert MODEL_TIERS["tiny"]["hidden_dim"] == 960
    # Qwen2.5-1.5B-Instruct
    assert MODEL_TIERS["small"]["hidden_dim"] == 1536
    # Gemma-4-E4B-it text_config.hidden_size
    assert MODEL_TIERS["large"]["hidden_dim"] == 2560
    # Qwen3.5-9B text_config.hidden_size
    assert MODEL_TIERS["xlarge"]["hidden_dim"] == 4096


def test_detect_cuda_vram_returns_int_or_none():
    result = detect_cuda_vram()
    if torch.cuda.is_available():
        assert isinstance(result, int)
        assert result > 0
    else:
        assert result is None


def test_detect_cuda_vram_reports_gb_floor(monkeypatch):
    """When CUDA claims 24*1024**3 bytes, detect reports 24 GB (floor)."""

    class _FakeProps:
        total_memory = 24 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: _FakeProps())
    assert detect_cuda_vram() == 24


def test_detect_cuda_vram_floors_fractional_gb(monkeypatch):
    """A card reporting 7.94 GB of bytes must floor to 7, not round to 8."""

    class _FakeProps:
        total_memory = int(7.94 * 1024**3)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: _FakeProps())
    assert detect_cuda_vram() == 7


def test_detect_cuda_vram_3090_reports_23(monkeypatch):
    """RTX 3090 advertises 25.4B bytes (~23.64 GB); floor is 23. Document
    that this is intentional -- the safety factor compensates, not detect."""

    class _FakeProps:
        total_memory = int(23.64 * 1024**3)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: _FakeProps())
    assert detect_cuda_vram() == 23


def test_select_device_and_dtype_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, dtype = select_device_and_dtype()
    assert device.type == "cpu"
    assert dtype == torch.float32


def test_select_device_and_dtype_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, dtype = select_device_and_dtype()
    assert device.type == "cuda"
    assert dtype == torch.float16


# ----- effective_vram_gb ---------------------------------------------------


def test_effective_vram_gb_passes_through_none():
    """No CUDA -> None in, None out, regardless of factor."""
    assert effective_vram_gb(vram_gb=None, safety_factor=0.9) is None
    assert effective_vram_gb(vram_gb=None, safety_factor=1.0) is None
    assert effective_vram_gb(vram_gb=None, safety_factor=0.5) is None


def test_effective_vram_gb_default_factor_on_3090():
    """RTX 3090 (raw 23 GB) at the default 0.9 factor -> 20 GB effective.
    Picks the 'large' tier (12 <= 20 < 24) instead of getting bumped down
    to small by floor jitter."""
    assert effective_vram_gb(vram_gb=23, safety_factor=0.9) == 20


def test_effective_vram_gb_factor_one_means_no_headroom():
    """factor=1.0 -> identity. Operators who know exactly what they're
    doing can disable headroom this way."""
    assert effective_vram_gb(vram_gb=24, safety_factor=1.0) == 24
    assert effective_vram_gb(vram_gb=8, safety_factor=1.0) == 8


def test_effective_vram_gb_factor_half_aggressive_headroom():
    """factor=0.5 -> reserve half. Useful when SOMA itself is big (KV
    cache or long context blowing up the margin)."""
    assert effective_vram_gb(vram_gb=24, safety_factor=0.5) == 12
    assert effective_vram_gb(vram_gb=16, safety_factor=0.5) == 8


def test_effective_vram_gb_floors_fractional_result():
    """int() truncates toward zero -- 23 * 0.9 = 20.7 -> 20, not 21."""
    assert effective_vram_gb(vram_gb=23, safety_factor=0.9) == 20
    # 7 * 0.9 = 6.3 -> 6, not 7.
    assert effective_vram_gb(vram_gb=7, safety_factor=0.9) == 6


# ----- auto_select_tier thresholds (post-Phase-7 modernization) -----------


def test_auto_select_tier_no_vram_returns_tiny():
    assert auto_select_tier(vram_gb=None) == "tiny"


def test_auto_select_tier_below_4_returns_tiny():
    assert auto_select_tier(vram_gb=0) == "tiny"
    assert auto_select_tier(vram_gb=2) == "tiny"
    assert auto_select_tier(vram_gb=3) == "tiny"


def test_auto_select_tier_4_returns_small():
    """4 is the small tier floor."""
    assert auto_select_tier(vram_gb=4) == "small"


def test_auto_select_tier_8_returns_small():
    assert auto_select_tier(vram_gb=8) == "small"


def test_auto_select_tier_11_returns_small():
    """11 is just below the new large threshold (was 16, now 12)."""
    assert auto_select_tier(vram_gb=11) == "small"


def test_auto_select_tier_12_returns_large():
    """12 is the new large threshold (lowered from 16 in Phase 7 Track A
    to fit Gemma-4-E4B-it's smaller VRAM footprint)."""
    assert auto_select_tier(vram_gb=12) == "large"


def test_auto_select_tier_16_returns_large():
    """16 GB is comfortably in the large band now."""
    assert auto_select_tier(vram_gb=16) == "large"


def test_auto_select_tier_20_returns_large():
    """RTX 3090 (post safety_factor 0.9: 23 -> 20) lands in large."""
    assert auto_select_tier(vram_gb=20) == "large"


def test_auto_select_tier_23_returns_large():
    """23 (raw 3090 floor without safety factor) is still large, not xlarge.
    The xlarge threshold is 24 -- you need a true 24GB+ card after
    headroom adjustment to get the dense Qwen3.5-9B."""
    assert auto_select_tier(vram_gb=23) == "large"


def test_auto_select_tier_24_returns_xlarge():
    """24 GB effective -> xlarge (Qwen3.5-9B dense)."""
    assert auto_select_tier(vram_gb=24) == "xlarge"


def test_auto_select_tier_48_returns_xlarge():
    """A6000-class cards still pick xlarge -- there's no even-larger tier
    yet, and Qwen3.5-9B is the operator-blessed default 24GB+ choice."""
    assert auto_select_tier(vram_gb=48) == "xlarge"


@pytest.mark.parametrize(
    "vram, expected",
    [
        (None, "tiny"),
        (0, "tiny"),
        (3, "tiny"),
        (4, "small"),
        (11, "small"),
        (12, "large"),
        (16, "large"),
        (23, "large"),
        (24, "xlarge"),
        (80, "xlarge"),
    ],
)
def test_auto_select_tier_full_threshold_table(vram, expected):
    """Single source-of-truth table for the threshold spec."""
    assert auto_select_tier(vram_gb=vram) == expected
