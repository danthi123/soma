"""Tests for ``soma.io.image_encoder.ImageEncoder``."""

from __future__ import annotations

import pytest
import torch

from soma.io.image_encoder import ImageEncoder


class TestConstruction:
    def test_defaults(self) -> None:
        enc = ImageEncoder()
        assert enc.patch_size == 16
        assert enc.embed_dim == 64
        assert enc.in_channels == 3

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"patch_size": 0}, "patch_size"),
            ({"embed_dim": 0}, "embed_dim"),
            ({"in_channels": 0}, "in_channels"),
        ],
    )
    def test_validation(self, kwargs: dict[str, int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ImageEncoder(**kwargs)

    def test_learnable_params(self) -> None:
        enc = ImageEncoder(patch_size=4, embed_dim=8, in_channels=3)
        names = {n for n, _ in enc.named_parameters()}
        assert "patch_proj.weight" in names
        assert "patch_proj.bias" in names


class TestNumPatches:
    def test_num_patches(self) -> None:
        enc = ImageEncoder(patch_size=4)
        assert enc.num_patches(height=16, width=16) == 16
        assert enc.num_patches(height=16, width=8) == 8


class TestEncodeBatch:
    def test_shape(self) -> None:
        enc = ImageEncoder(patch_size=4, embed_dim=8, in_channels=3)
        image = torch.randn(3, 16, 16)  # 4x4 = 16 patches
        patches = enc.encode_batch(image)
        assert patches.shape == (16, 8)

    def test_rejects_wrong_ndim(self) -> None:
        enc = ImageEncoder(patch_size=4, embed_dim=8, in_channels=3)
        with pytest.raises(ValueError, match="C, H, W"):
            enc.encode_batch(torch.randn(3, 16))  # 2D

    def test_rejects_wrong_channels(self) -> None:
        enc = ImageEncoder(patch_size=4, embed_dim=8, in_channels=3)
        with pytest.raises(ValueError, match="channels"):
            enc.encode_batch(torch.randn(1, 16, 16))

    def test_rejects_non_divisible_size(self) -> None:
        enc = ImageEncoder(patch_size=4, embed_dim=8, in_channels=3)
        with pytest.raises(ValueError, match="divisible"):
            enc.encode_batch(torch.randn(3, 15, 16))

    def test_gradients_flow(self) -> None:
        enc = ImageEncoder(patch_size=4, embed_dim=8, in_channels=3)
        image = torch.randn(3, 8, 8, requires_grad=True)
        patches = enc.encode_batch(image)
        patches.sum().backward()
        assert image.grad is not None


class TestEncodeList:
    def test_returns_list_of_per_patch_tensors(self) -> None:
        enc = ImageEncoder(patch_size=4, embed_dim=8, in_channels=3)
        image = torch.randn(3, 8, 8)
        patches = enc.encode(image)
        assert len(patches) == 4  # 2x2 = 4 patches
        for p in patches:
            assert p.shape == (8,)
