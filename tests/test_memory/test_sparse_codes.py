"""Tests for `soma.memory.sparse_codes` — Path B Phase 2 TDD.

Design reference: docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md § Phase 2.

Tests expose the four required behaviors:
1. kwta() produces exactly k active dims (or min(k, available) on small inputs)
2. kwta() is deterministic given a projection seed
3. pattern_separate() reduces pairwise overlap above threshold
4. code_similarity() (Jaccard) is bounded in [0, 1]
5. pattern_separate() is monotone in strength (more strength → less overlap)
"""

from __future__ import annotations

import numpy as np
import pytest

from soma.memory.sparse_codes import (
    SparseCode,
    code_similarity,
    kwta,
    pattern_separate,
)


class TestKWTA:
    def test_kwta_produces_k_active_dims(self):
        """kwta should return a SparseCode with exactly k active dimensions
        when the dense input has at least k non-zero projected activations."""
        rng = np.random.default_rng(0)
        dense = rng.standard_normal(128).astype(np.float32)
        code = kwta(dense, k=32, dim=4096, seed=42)
        assert isinstance(code, SparseCode)
        assert code.k == 32
        assert code.active.shape == (32,)
        # Active indices must be sorted, unique, and within [0, dim).
        assert np.all(np.diff(code.active) > 0), "indices not sorted/unique"
        assert code.active.min() >= 0
        assert code.active.max() < 4096

    def test_kwta_deterministic_given_seed(self):
        """Two kwta() calls with the same seed and same dense input must
        produce identical codes. Different seeds should (generally) produce
        different codes."""
        rng = np.random.default_rng(0)
        dense = rng.standard_normal(128).astype(np.float32)
        code_a = kwta(dense, k=16, dim=512, seed=7)
        code_b = kwta(dense, k=16, dim=512, seed=7)
        code_c = kwta(dense, k=16, dim=512, seed=99)
        assert np.array_equal(code_a.active, code_b.active), "seed-7 calls differ"
        # seed mismatch: overwhelmingly likely to differ at random-normal proj
        assert not np.array_equal(code_a.active, code_c.active), (
            "different seeds produced same code — projection not seed-dependent"
        )

    def test_kwta_reuses_external_projection(self):
        """If a projection matrix is passed explicitly, kwta uses it and
        ignores the seed. Two calls with the same projection must agree
        regardless of seed argument."""
        rng = np.random.default_rng(5)
        dense = rng.standard_normal(64).astype(np.float32)
        proj = rng.standard_normal((2048, 64)).astype(np.float32)
        code_a = kwta(dense, k=20, dim=2048, projection=proj, seed=1)
        code_b = kwta(dense, k=20, dim=2048, projection=proj, seed=2)
        assert np.array_equal(code_a.active, code_b.active)


class TestCodeSimilarity:
    def test_code_similarity_jaccard_bounded_0_1(self):
        """Jaccard similarity is always in [0, 1] across reasonable inputs."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = SparseCode(
                active=np.sort(rng.choice(256, size=16, replace=False)).astype(np.int32),
                dim=256,
            )
            b = SparseCode(
                active=np.sort(rng.choice(256, size=16, replace=False)).astype(np.int32),
                dim=256,
            )
            s = code_similarity(a, b)
            assert 0.0 <= s <= 1.0

    def test_code_similarity_identical_returns_one(self):
        idx = np.array([1, 4, 9, 16], dtype=np.int32)
        a = SparseCode(active=idx, dim=32)
        b = SparseCode(active=idx, dim=32)
        assert code_similarity(a, b) == 1.0

    def test_code_similarity_disjoint_returns_zero(self):
        a = SparseCode(active=np.array([0, 1, 2], dtype=np.int32), dim=32)
        b = SparseCode(active=np.array([3, 4, 5], dtype=np.int32), dim=32)
        assert code_similarity(a, b) == 0.0


class TestPatternSeparate:
    def test_pattern_separate_reduces_overlap_above_threshold(self):
        """Given two codes that share > threshold fraction of active dims,
        pattern_separate should produce codes with lower pairwise similarity
        than the input codes had."""
        # Two codes that share 12 of 16 active dims (Jaccard = 12/20 = 0.60).
        dim = 256
        shared = np.arange(12, dtype=np.int32)
        a_only = np.arange(100, 104, dtype=np.int32)
        b_only = np.arange(200, 204, dtype=np.int32)
        a = SparseCode(active=np.sort(np.concatenate([shared, a_only])), dim=dim)
        b = SparseCode(active=np.sort(np.concatenate([shared, b_only])), dim=dim)

        before = code_similarity(a, b)
        [a_sep, b_sep] = pattern_separate([a, b], strength=0.5)
        after = code_similarity(a_sep, b_sep)
        assert after < before, f"separation did not reduce overlap: {before:.3f} -> {after:.3f}"

    def test_pattern_separate_monotone_in_strength(self):
        """Higher strength should produce <= overlap than lower strength.
        (Monotone non-increasing, not strictly decreasing — strength=0 is
        a no-op.)"""
        dim = 256
        shared = np.arange(10, dtype=np.int32)
        a = SparseCode(active=np.sort(np.concatenate([shared, np.array([100, 101])])), dim=dim)
        b = SparseCode(active=np.sort(np.concatenate([shared, np.array([200, 201])])), dim=dim)

        overlaps = []
        for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
            codes = pattern_separate([a, b], strength=strength)
            overlaps.append(code_similarity(codes[0], codes[1]))

        # Non-increasing sequence.
        for prev, curr in zip(overlaps[:-1], overlaps[1:]):
            assert curr <= prev + 1e-9, f"overlap increased at higher strength: {overlaps}"

    def test_pattern_separate_preserves_dim(self):
        """The SparseCode's ambient dim should not change after separation."""
        a = SparseCode(active=np.array([1, 2, 3], dtype=np.int32), dim=64)
        b = SparseCode(active=np.array([2, 3, 4], dtype=np.int32), dim=64)
        separated = pattern_separate([a, b], strength=0.5)
        for c in separated:
            assert c.dim == 64

    def test_pattern_separate_strength_zero_is_identity(self):
        """strength=0 should leave codes unchanged."""
        a = SparseCode(active=np.array([1, 2, 5], dtype=np.int32), dim=32)
        b = SparseCode(active=np.array([1, 2, 8], dtype=np.int32), dim=32)
        [a_out, b_out] = pattern_separate([a, b], strength=0.0)
        assert np.array_equal(a_out.active, a.active)
        assert np.array_equal(b_out.active, b.active)


class TestSparseCodeValidation:
    def test_sparse_code_rejects_unsorted_active(self):
        """Active indices must be sorted for efficient intersection."""
        with pytest.raises(ValueError):
            SparseCode(active=np.array([3, 1, 2], dtype=np.int32), dim=16)

    def test_sparse_code_rejects_duplicate_active(self):
        with pytest.raises(ValueError):
            SparseCode(active=np.array([1, 2, 2, 3], dtype=np.int32), dim=16)

    def test_sparse_code_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            SparseCode(active=np.array([1, 20], dtype=np.int32), dim=16)
        with pytest.raises(ValueError):
            SparseCode(active=np.array([-1, 2], dtype=np.int32), dim=16)

    def test_sparse_code_k_property(self):
        c = SparseCode(active=np.array([0, 5, 10], dtype=np.int32), dim=16)
        assert c.k == 3
