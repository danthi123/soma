"""Tests for ``soma.io.audio_encoder.AudioEncoder``."""

from __future__ import annotations

import pytest
import torch

from soma.io.audio_encoder import AudioEncoder, _build_mel_filter_bank


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def encoder() -> AudioEncoder:
    """Small encoder sized so tests finish quickly on CPU."""
    return AudioEncoder(
        patch_size=4,
        embed_dim=16,
        sample_rate=8000,
        n_mels=16,
        n_fft=128,
        hop_length=64,
    )


def _synthetic_waveform(encoder: AudioEncoder, duration_patches: int = 4) -> torch.Tensor:
    """Generate a sine-wave waveform long enough to produce ``duration_patches`` patches."""
    # We want at least (duration_patches * patch_size) STFT frames.
    target_frames = duration_patches * encoder.patch_size
    num_samples = encoder.n_fft + (target_frames - 1) * encoder.hop_length + 1
    t = torch.linspace(0.0, num_samples / encoder.sample_rate, num_samples)
    return 0.3 * torch.sin(2 * torch.pi * 440.0 * t)


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------
class TestConstruction:
    def test_rejects_non_multiple_n_mels(self) -> None:
        with pytest.raises(ValueError, match="n_mels"):
            AudioEncoder(patch_size=4, n_mels=10)

    def test_rejects_non_power_of_two_n_fft(self) -> None:
        with pytest.raises(ValueError, match="power of two"):
            AudioEncoder(n_fft=300)

    def test_rejects_non_positive_args(self) -> None:
        with pytest.raises(ValueError):
            AudioEncoder(patch_size=0)
        with pytest.raises(ValueError):
            AudioEncoder(embed_dim=0)
        with pytest.raises(ValueError):
            AudioEncoder(sample_rate=0)
        with pytest.raises(ValueError):
            AudioEncoder(hop_length=0)


# ----------------------------------------------------------------------
# Mel filter bank
# ----------------------------------------------------------------------
class TestMelFilterBank:
    def test_shape(self) -> None:
        fb = _build_mel_filter_bank(n_mels=16, n_fft=128, sample_rate=8000)
        assert fb.shape == (16, 128 // 2 + 1)

    def test_triangles_sum_up_reasonably(self) -> None:
        fb = _build_mel_filter_bank(n_mels=16, n_fft=128, sample_rate=8000)
        # Each row is a triangle; peak is in (0, 1].
        assert torch.all(fb.max(dim=1).values > 0)
        assert torch.all(fb.max(dim=1).values <= 1.0)


# ----------------------------------------------------------------------
# Spectrogram
# ----------------------------------------------------------------------
class TestLogMelSpectrogram:
    def test_shape_for_mono_input(self, encoder: AudioEncoder) -> None:
        wav = _synthetic_waveform(encoder, duration_patches=3)
        spec = encoder.log_mel_spectrogram(wav)
        assert spec.ndim == 2
        assert spec.shape[0] == encoder.n_mels
        assert spec.shape[1] > 0

    def test_handles_stereo_by_averaging(self, encoder: AudioEncoder) -> None:
        mono = _synthetic_waveform(encoder)
        stereo = torch.stack([mono, mono])
        assert encoder.log_mel_spectrogram(mono).shape == (
            encoder.n_mels,
            encoder.num_frames_for(mono.numel()),
        )
        assert encoder.log_mel_spectrogram(stereo).shape == (
            encoder.n_mels,
            encoder.num_frames_for(mono.numel()),
        )

    def test_rejects_3d_input(self, encoder: AudioEncoder) -> None:
        with pytest.raises(ValueError, match=r"waveform"):
            encoder.log_mel_spectrogram(torch.randn(2, 3, 1000))


# ----------------------------------------------------------------------
# Encode
# ----------------------------------------------------------------------
class TestEncode:
    def test_encode_batch_shape(self, encoder: AudioEncoder) -> None:
        wav = _synthetic_waveform(encoder, duration_patches=3)
        patches = encoder.encode_batch(wav)
        assert patches.ndim == 2
        assert patches.shape[1] == encoder.embed_dim
        assert patches.shape[0] > 0

    def test_encode_returns_list_matching_batch(self, encoder: AudioEncoder) -> None:
        wav = _synthetic_waveform(encoder)
        patches_list = encoder.encode(wav)
        patches_batch = encoder.encode_batch(wav)
        assert len(patches_list) == patches_batch.shape[0]
        for vec in patches_list:
            assert vec.shape == (encoder.embed_dim,)

    def test_too_short_waveform_raises(self, encoder: AudioEncoder) -> None:
        short = torch.randn(encoder.n_fft)  # produces < patch_size frames
        with pytest.raises(ValueError, match="too short"):
            encoder.encode_batch(short)

    def test_num_patches_for_matches_encode_length(self, encoder: AudioEncoder) -> None:
        wav = _synthetic_waveform(encoder, duration_patches=3)
        expected = encoder.num_patches_for(wav.numel())
        assert encoder.encode_batch(wav).shape[0] == expected


# ----------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------
class TestGradient:
    def test_patch_proj_receives_grad(self, encoder: AudioEncoder) -> None:
        wav = _synthetic_waveform(encoder)
        wav.requires_grad_(False)  # gradient flows through conv weights only
        patches = encoder.encode_batch(wav)
        loss = patches.sum()
        loss.backward()
        assert encoder.patch_proj.weight.grad is not None
        assert torch.any(encoder.patch_proj.weight.grad != 0)


# ----------------------------------------------------------------------
# Device portability
# ----------------------------------------------------------------------
class TestDevice:
    def test_buffers_move_with_module(self, encoder: AudioEncoder) -> None:
        # No CUDA? Just confirm buffers are accessible and on cpu.
        assert encoder._mel_fb().device.type == "cpu"
        assert encoder._window().device.type == "cpu"
