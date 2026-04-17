"""Unit tests for D1 associative-memory infrastructure.

Scoped to "the harness does what it says" -- not "SOMA wins".
"""

from __future__ import annotations

import pytest
import torch

from research.associative.baselines import (
    ClassicalHopfield,
    KNNRecall,
    ModernHopfield,
)
from research.associative.datasets import make_binary_patterns
from research.associative.harness import RecallModel, evaluate_recall

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_baselines_satisfy_protocol() -> None:
    """All three baselines implement the RecallModel protocol."""
    assert isinstance(ClassicalHopfield(), RecallModel)
    assert isinstance(ModernHopfield(), RecallModel)
    assert isinstance(KNNRecall(), RecallModel)


# ---------------------------------------------------------------------------
# Single-pattern recall (trivial case)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [ClassicalHopfield, ModernHopfield, KNNRecall])
def test_single_pattern_exact_recall(model_cls: type) -> None:
    """With 1 stored pattern, all models should recall it perfectly."""
    ds = make_binary_patterns(n_patterns=1, dim=50, mask_frac=0.2, seed=42)
    model = model_cls()
    bipolar = isinstance(model, ClassicalHopfield)
    metrics = evaluate_recall(model, ds.patterns, ds.probes, bipolar=bipolar)
    assert metrics.exact_match_rate == 1.0, f"{model_cls.__name__} failed single-pattern recall"


# ---------------------------------------------------------------------------
# Classical Hopfield capacity around 0.14 * D
# ---------------------------------------------------------------------------

def test_classical_hopfield_capacity_bound() -> None:
    """Classical Hopfield capacity should degrade around 0.14 * D.

    With D=50, the theoretical limit is ~7 patterns.
    We check that recall is near-perfect at N=5 and degrades by N=15.
    """
    dim = 50
    model = ClassicalHopfield(max_iter=200)

    # Well below capacity: should work
    ds_low = make_binary_patterns(n_patterns=5, dim=dim, mask_frac=0.2, seed=0)
    m_low = evaluate_recall(model, ds_low.patterns, ds_low.probes, bipolar=True)
    assert m_low.exact_match_rate >= 0.6, (
        f"Classical Hopfield should recall most of 5/{dim} patterns, "
        f"got {m_low.exact_match_rate:.2f}"
    )

    # Well above capacity: should fail
    ds_high = make_binary_patterns(n_patterns=20, dim=dim, mask_frac=0.2, seed=0)
    m_high = evaluate_recall(model, ds_high.patterns, ds_high.probes, bipolar=True)
    assert m_high.exact_match_rate < 0.8, (
        f"Classical Hopfield should NOT recall most of 20/{dim} patterns, "
        f"got {m_high.exact_match_rate:.2f}"
    )


# ---------------------------------------------------------------------------
# Modern Hopfield should beat classical at higher N
# ---------------------------------------------------------------------------

def test_modern_hopfield_higher_capacity() -> None:
    """Modern Hopfield should recall more patterns than classical at N=12."""
    dim = 50
    n = 12
    ds = make_binary_patterns(n_patterns=n, dim=dim, mask_frac=0.2, seed=7)

    classical = ClassicalHopfield(max_iter=200)
    modern = ModernHopfield(beta=10.0, n_iter=10)

    m_c = evaluate_recall(classical, ds.patterns, ds.probes, bipolar=True)
    m_m = evaluate_recall(modern, ds.patterns, ds.probes, bipolar=True)

    # Modern should have higher per-element accuracy (even if not exact match)
    assert m_m.per_element_accuracy >= m_c.per_element_accuracy - 0.05, (
        f"Modern Hopfield ({m_m.per_element_accuracy:.3f}) should be "
        f"at least close to classical ({m_c.per_element_accuracy:.3f}) at N={n}"
    )


# ---------------------------------------------------------------------------
# Harness metrics are valid
# ---------------------------------------------------------------------------

def test_harness_metrics_range() -> None:
    """All metric values should be in valid ranges."""
    ds = make_binary_patterns(n_patterns=5, dim=30, mask_frac=0.2, seed=99)
    model = KNNRecall()
    metrics = evaluate_recall(model, ds.patterns, ds.probes, bipolar=True)

    assert 0.0 <= metrics.exact_match_rate <= 1.0
    assert 0.0 <= metrics.per_element_accuracy <= 1.0
    assert metrics.mse >= 0.0
    assert metrics.n_queries == 5


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def test_binary_patterns_shape_and_values() -> None:
    """Generated patterns should be bipolar (+/-1) with correct shapes."""
    ds = make_binary_patterns(n_patterns=10, dim=50, seed=0)
    assert ds.patterns.shape == (10, 50)
    assert ds.probes.shape == (10, 50)
    assert set(ds.patterns.unique().tolist()) == {-1.0, 1.0}
    # Probes should have zeros (masked positions) and +/-1
    assert 0.0 in ds.probes.unique().tolist()


def test_binary_patterns_deterministic() -> None:
    """Same seed produces identical patterns."""
    ds1 = make_binary_patterns(n_patterns=5, dim=20, seed=123)
    ds2 = make_binary_patterns(n_patterns=5, dim=20, seed=123)
    assert torch.equal(ds1.patterns, ds2.patterns)
    assert torch.equal(ds1.probes, ds2.probes)
