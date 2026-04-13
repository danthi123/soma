"""Shared test fixtures for SOMA test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Expose repo root so tests can `from scripts.X import ...` — scripts/ has its
# own __init__.py but pyproject.toml only puts src/ on pythonpath.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
