"""Audio input encoder: waveform -> log-mel spectrogram patches -> embeddings.

Whitepaper Section 8.1 (audio analog).

Mirrors ``ImageEncoder`` but for audio. The pipeline is:

1. Convert a raw waveform ``(C, T)`` (or ``(T,)`` mono) into a log-mel
   spectrogram ``(n_mels, T')`` using ``torchaudio`` when available, or a
   pure-``torch`` STFT fallback so tests run without torchaudio installed.
2. Treat the spectrogram as a 2D image with one channel and apply
   patch projection (same Conv2d trick as ``ImageEncoder``) to emit a
   sequence of embedding vectors, one per time-frequency patch.

Design notes:
- Torchaudio is an optional dependency (already in the ``[multimodal]``
  extra). We degrade gracefully to a torch-native STFT + mel filter bank
  if torchaudio is missing so the encoder is always instantiable.
- ``encode(waveform)`` returns a list of ``(embed_dim,)`` tensors, matching
  ``ImageEncoder.encode`` / ``TextEncoder.encode``.
- The mel filter bank is computed once at construction and registered as
  a buffer so it moves with the module across devices.
"""

from __future__ import annotations

import torch
from torch import nn


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _build_mel_filter_bank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Triangular mel filter bank with shape ``(n_mels, n_fft // 2 + 1)``."""
    if f_max is None:
        f_max = sample_rate / 2.0
    mel_min = _hz_to_mel(torch.tensor(f_min))
    mel_max = _hz_to_mel(torch.tensor(f_max))
    mel_points = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    # Convert Hz to FFT bin indices.
    bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    filter_bank = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        left, center, right = int(bin_points[m - 1]), int(bin_points[m]), int(bin_points[m + 1])
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for k in range(left, center):
            if 0 <= k < filter_bank.shape[1]:
                filter_bank[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            if 0 <= k < filter_bank.shape[1]:
                filter_bank[m - 1, k] = (right - k) / max(1, right - center)
    return filter_bank


class AudioEncoder(nn.Module):
    """Project an audio waveform into a sequence of patch embeddings.

    Parameters
    ----------
    patch_size:
        Size of the square patch taken from the ``(n_mels, time)``
        log-mel spectrogram. Mel and time dims must both be divisible by
        ``patch_size``.
    embed_dim:
        Dimensionality of each output patch embedding.
    sample_rate:
        Expected sample rate of the incoming waveform (Hz).
    n_mels:
        Number of mel bins in the spectrogram. Must be a positive multiple
        of ``patch_size``.
    n_fft:
        FFT size used for the STFT.
    hop_length:
        STFT hop length (samples between consecutive frames). Smaller
        values give finer time resolution but more frames.
    log_offset:
        Small constant added inside the log to avoid ``log(0)``.
    """

    def __init__(
        self,
        patch_size: int = 8,
        embed_dim: int = 64,
        *,
        sample_rate: int = 16_000,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 256,
        log_offset: float = 1e-6,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")
        if n_mels % patch_size != 0:
            raise ValueError(f"n_mels ({n_mels}) must be divisible by patch_size ({patch_size})")
        if n_fft <= 0 or (n_fft & (n_fft - 1)) != 0:
            raise ValueError(f"n_fft must be a positive power of two, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_offset = log_offset

        mel_fb = _build_mel_filter_bank(n_mels, n_fft, sample_rate)
        self.register_buffer("mel_filter_bank", mel_fb)
        self.register_buffer("hann_window", torch.hann_window(n_fft))

        # Patch projection — 1-channel spectrogram as an image.
        self.patch_proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Spectrogram
    # ------------------------------------------------------------------
    def _mel_fb(self) -> torch.Tensor:
        fb = self.mel_filter_bank
        assert isinstance(fb, torch.Tensor)
        return fb

    def _window(self) -> torch.Tensor:
        win = self.hann_window
        assert isinstance(win, torch.Tensor)
        return win

    def log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute a log-mel spectrogram from ``waveform``.

        Input shape: ``(T,)`` mono or ``(C, T)`` multi-channel — we average
        channels before STFT.
        Output shape: ``(n_mels, num_frames)``.
        """
        if waveform.ndim == 2:
            # Mix down to mono.
            wav = waveform.mean(dim=0)
        elif waveform.ndim == 1:
            wav = waveform
        else:
            raise ValueError(f"Expected (T,) or (C, T) waveform, got shape {tuple(waveform.shape)}")
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window().to(wav.device),
            return_complex=True,
            center=False,
        )
        power = stft.real.pow(2) + stft.imag.pow(2)
        mel = self._mel_fb().to(wav.device) @ power
        return torch.log(mel + self.log_offset)

    def num_frames_for(self, num_samples: int) -> int:
        """Number of STFT frames produced for a waveform of ``num_samples`` samples."""
        if num_samples < self.n_fft:
            return 0
        return 1 + (num_samples - self.n_fft) // self.hop_length

    def num_patches_for(self, num_samples: int) -> int:
        """Number of patch embeddings produced for a waveform of given length."""
        frames = self.num_frames_for(num_samples)
        return (self.n_mels // self.patch_size) * (frames // self.patch_size)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return a ``(num_patches, embed_dim)`` tensor for one waveform."""
        spec = self.log_mel_spectrogram(waveform)
        # Ensure both dims divide patch_size.
        n_mels, frames = spec.shape
        if frames < self.patch_size:
            raise ValueError(
                f"Audio too short: got {frames} frames (< patch_size={self.patch_size}); "
                f"waveform must span at least "
                f"{self.n_fft + (self.patch_size - 1) * self.hop_length} samples"
            )
        usable_frames = (frames // self.patch_size) * self.patch_size
        spec = spec[:, :usable_frames]
        # (1, 1, n_mels, frames) for Conv2d.
        patches: torch.Tensor = self.patch_proj(spec.unsqueeze(0).unsqueeze(0))
        patches = patches.flatten(2)  # (1, embed_dim, num_patches)
        patches = patches.squeeze(0).transpose(0, 1)  # (num_patches, embed_dim)
        return patches.contiguous()

    def encode(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of per-patch embeddings."""
        return list(self.encode_batch(waveform).unbind(dim=0))

    def min_samples_for_one_patch(self) -> int:
        """Minimum waveform length (in samples) required to produce at least one patch."""
        return self.n_fft + (self.patch_size - 1) * self.hop_length + 1
