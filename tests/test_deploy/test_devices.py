import torch

from soma.deploy.devices import (
    MODEL_TIERS,
    auto_select_tier,
    detect_cuda_vram,
    select_device_and_dtype,
)


def test_model_tiers_has_three_entries():
    assert set(MODEL_TIERS.keys()) == {"tiny", "small", "large"}
    for _tier_name, spec in MODEL_TIERS.items():
        assert "name" in spec
        assert "hidden_dim" in spec
        assert "min_vram_gb" in spec
        assert "approx_fp16_gb" in spec


def test_model_tiers_vram_monotonic():
    """Larger tiers should require more VRAM."""
    tiers = ["tiny", "small", "large"]
    vram = [MODEL_TIERS[t]["min_vram_gb"] for t in tiers]
    assert vram == sorted(vram)


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


def test_auto_select_tier_no_vram_returns_tiny():
    assert auto_select_tier(vram_gb=None) == "tiny"


def test_auto_select_tier_small_vram_returns_tiny():
    assert auto_select_tier(vram_gb=2) == "tiny"


def test_auto_select_tier_8gb_returns_small():
    assert auto_select_tier(vram_gb=8) == "small"


def test_auto_select_tier_24gb_returns_large():
    assert auto_select_tier(vram_gb=24) == "large"


def test_auto_select_tier_16gb_returns_large():
    """16GB is the large threshold."""
    assert auto_select_tier(vram_gb=16) == "large"


def test_auto_select_tier_15gb_returns_small():
    """Just below large threshold -> small."""
    assert auto_select_tier(vram_gb=15) == "small"
