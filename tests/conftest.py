"""Shared test fixtures for SOMA test suite."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device() -> torch.device:
    """Return CUDA device, skip if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")
