"""Image input encoder: patch projection -> per-patch embedding.

Whitepaper Section 8.1.

A ``Conv2d`` with ``kernel_size == stride == patch_size`` turns a
``(C, H, W)`` image into a grid of patches, each projected to an
``embed_dim`` vector. ``encode`` returns a flat list of patch embeddings
suitable for feeding into a SENSOR node or a multimodal curriculum.

We deliberately don't couple this module to ``torchvision`` — callers
supply already-normalized ``torch.Tensor`` images. The whitepaper uses
3 input channels; we expose ``in_channels`` so callers can feed greyscale
or multichannel data.
"""

from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Project an image into a sequence of patch embeddings."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 64,
        *,
        in_channels: int = 3,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        self.patch_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if device is not None:
            self.to(device)

    def num_patches(self, height: int, width: int) -> int:
        """Return the number of patches produced for a ``(H, W)`` image."""
        return (height // self.patch_size) * (width // self.patch_size)

    def encode_batch(self, image: torch.Tensor) -> torch.Tensor:
        """Return a ``(num_patches, embed_dim)`` tensor for one image.

        ``image`` must have shape ``(C, H, W)`` and its H, W must be
        divisible by ``patch_size`` (we raise otherwise — callers should
        resize/crop upstream).
        """
        if image.ndim != 3:
            raise ValueError(f"encode_batch expects (C, H, W), got {tuple(image.shape)}")
        channels, height, width = image.shape
        if channels != self.in_channels:
            raise ValueError(f"ImageEncoder expected {self.in_channels} channels, got {channels}")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Image dims ({height}, {width}) must be divisible by patch_size={self.patch_size}"
            )

        # Conv2d expects (N, C, H, W); add batch dim then project.
        patches: torch.Tensor = self.patch_proj(image.unsqueeze(0))  # (1, embed_dim, H', W')
        # Flatten spatial dims and drop the batch dim.
        patches = patches.flatten(2)  # (1, embed_dim, num_patches)
        patches = patches.squeeze(0).transpose(0, 1)  # (num_patches, embed_dim)
        return patches.contiguous()

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of per-patch embeddings (matches whitepaper API)."""
        return list(self.encode_batch(image).unbind(dim=0))
