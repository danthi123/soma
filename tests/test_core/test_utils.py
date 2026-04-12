"""Tests for ``soma.core.utils``."""

from __future__ import annotations

import re

import pytest
import torch

from soma.core.utils import create_projection_if_needed, generate_uuid


class TestGenerateUuid:
    def test_returns_hex_string(self) -> None:
        uid = generate_uuid()
        assert isinstance(uid, str)
        assert len(uid) == 32
        assert re.fullmatch(r"[0-9a-f]{32}", uid)

    def test_ids_are_unique(self) -> None:
        ids = {generate_uuid() for _ in range(1000)}
        assert len(ids) == 1000


class TestCreateProjectionIfNeeded:
    def test_returns_none_when_dims_match(self) -> None:
        assert create_projection_if_needed(64, 64) is None

    def test_returns_linear_when_dims_differ(self) -> None:
        proj = create_projection_if_needed(64, 128)
        assert proj is not None
        assert proj.in_features == 64
        assert proj.out_features == 128
        assert proj.bias is None  # bias=False

    def test_projection_is_applicable(self) -> None:
        proj = create_projection_if_needed(8, 4)
        assert proj is not None
        x = torch.randn(8)
        y = proj(x)
        assert y.shape == (4,)

    def test_rejects_non_positive_dims(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            create_projection_if_needed(0, 64)
        with pytest.raises(ValueError, match="positive"):
            create_projection_if_needed(64, -1)

    def test_honors_device(self) -> None:
        proj = create_projection_if_needed(4, 8, device="cpu")
        assert proj is not None
        assert proj.weight.device.type == "cpu"
