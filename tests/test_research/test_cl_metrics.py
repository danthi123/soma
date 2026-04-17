"""Unit tests for CL metrics (ACC, BWT, FWT)."""

from __future__ import annotations

import numpy as np
import pytest

from research.cl.metrics import acc, bwt, fwt


def test_acc_perfect() -> None:
    """Perfect accuracy on all tasks -> ACC = 1.0."""
    A = np.ones((3, 3))
    assert acc(A) == pytest.approx(1.0)


def test_acc_last_row() -> None:
    """ACC is the mean of the last row."""
    A = np.array([
        [0.9, 0.0, 0.0],
        [0.5, 0.8, 0.0],
        [0.3, 0.4, 0.7],
    ])
    assert acc(A) == pytest.approx((0.3 + 0.4 + 0.7) / 3)


def test_bwt_no_forgetting() -> None:
    """If final accuracy on each task equals its diagonal, BWT = 0."""
    A = np.array([
        [0.9, 0.0],
        [0.9, 0.8],
    ])
    assert bwt(A) == pytest.approx(0.0)


def test_bwt_forgetting() -> None:
    """BWT is negative when accuracy drops after training later tasks."""
    A = np.array([
        [0.9, 0.0, 0.0],
        [0.5, 0.8, 0.0],
        [0.3, 0.4, 0.7],
    ])
    # BWT = mean( A[2,0]-A[0,0], A[2,1]-A[1,1] ) = mean(-0.6, -0.4) = -0.5
    assert bwt(A) == pytest.approx(-0.5)


def test_bwt_single_task() -> None:
    """Single task -> BWT = 0 (no previous tasks to forget)."""
    A = np.array([[0.9]])
    assert bwt(A) == pytest.approx(0.0)


def test_fwt_no_transfer() -> None:
    """FWT is 0 when pre-training accuracy equals random baseline."""
    A = np.array([
        [0.9, 0.1],
        [0.5, 0.8],
    ])
    baseline = np.array([0.1, 0.1])
    assert fwt(A, random_baseline=baseline) == pytest.approx(0.0)


def test_fwt_positive() -> None:
    """Positive FWT when pre-training accuracy exceeds random baseline."""
    A = np.array([
        [0.9, 0.3, 0.0],
        [0.5, 0.8, 0.2],
        [0.3, 0.4, 0.7],
    ])
    baseline = np.full(3, 0.1)
    # FWT = mean( A[0,1]-0.1, A[1,2]-0.1 ) = mean(0.2, 0.1) = 0.15
    assert fwt(A, random_baseline=baseline) == pytest.approx(0.15)


def test_fwt_default_baseline() -> None:
    """Default baseline is 0.1 (10-class)."""
    A = np.array([
        [0.9, 0.1],
        [0.5, 0.8],
    ])
    assert fwt(A) == pytest.approx(0.0)
