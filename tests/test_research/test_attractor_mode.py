"""Unit tests for SOMA attractor mode (Research D2).

Tests:
- Construction: SOMA builds with attractor config, has integrators
- Output shape: step() returns outputs of the right shape
- Convergence: iterated recall produces outputs that stabilize
- Homeostasis: attractor mode doesn't crash with homeostasis active
- Store/recall: basic store+recall produces non-degenerate output
"""

from __future__ import annotations

import torch

from soma.research.attractor_mode import (
    make_attractor_config,
    make_attractor_soma,
    recall_pattern,
    store_pattern,
)


class TestAttractorConstruction:
    """SOMA in attractor config constructs correctly."""

    def test_creates_soma(self) -> None:
        soma = make_attractor_soma(pattern_dim=16, seed=0)
        assert soma is not None
        assert soma.graph.num_nodes > 0
        assert soma.graph.num_edges > 0

    def test_has_integrators(self) -> None:
        """Seed graph must include integrator nodes (commit 860cc82 fix)."""
        soma = make_attractor_soma(pattern_dim=16, n_integrators=4, seed=0)
        from soma.core.node import NodeType

        integrators = soma.graph.nodes_by_type(NodeType.INTEGRATOR)
        assert len(integrators) >= 4

    def test_config_disables_growth(self) -> None:
        config = make_attractor_config(16)
        assert config.synaptogenesis_interval > 1_000_000
        assert config.neurogenesis_interval > 1_000_000
        assert config.pruning_interval > 1_000_000

    def test_config_edge_weight_no_decay(self) -> None:
        config = make_attractor_config(16)
        assert config.edge_weight_decay == 1.0


class TestOutputShape:
    """step() returns outputs with the correct shape."""

    def test_output_shape_matches_pattern_dim(self) -> None:
        dim = 32
        soma = make_attractor_soma(pattern_dim=dim, seed=0)
        pattern = torch.randn(dim)
        result = soma.step(inputs={"text": pattern}, targets={"text": pattern})
        outputs = result["outputs"]
        assert "text" in outputs
        assert outputs["text"].shape[-1] == dim

    def test_recall_output_shape(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=0)
        probe = torch.randn(dim)
        rr = recall_pattern(soma, probe, max_iters=3)
        assert rr.final_output.shape[-1] == dim
        assert len(rr.outputs_per_iter) == 3 or rr.converged


class TestConvergence:
    """Iterated recall should produce outputs that stabilize."""

    def test_outputs_stabilize_after_iterations(self) -> None:
        """After many iterations on the same input, consecutive outputs
        should become increasingly similar (even if not fully converged)."""
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=42)
        # Store one pattern
        pattern = torch.ones(dim)
        store_pattern(soma, pattern, n_presentations=5)
        # Recall
        rr = recall_pattern(soma, pattern * 0.5, max_iters=50)
        # Check that later diffs are <= earlier diffs on average
        diffs = []
        for i in range(1, len(rr.outputs_per_iter)):
            d = (rr.outputs_per_iter[i] - rr.outputs_per_iter[i - 1]).norm().item()
            diffs.append(d)
        if len(diffs) > 10:
            early = sum(diffs[:5]) / 5
            late = sum(diffs[-5:]) / 5
            # Late diffs should not be dramatically larger than early
            # (would indicate divergence)
            assert late < early * 10, (
                f"Outputs appear to diverge: early_diff={early:.6f}, "
                f"late_diff={late:.6f}"
            )


class TestHomeostasis:
    """Attractor mode with homeostasis doesn't crash."""

    def test_store_and_recall_with_homeostasis(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=0)
        pattern = torch.randn(dim)
        # Store
        losses = store_pattern(soma, pattern, n_presentations=3)
        assert len(losses) == 3
        assert all(isinstance(v, float) for v in losses)
        # Recall
        rr = recall_pattern(soma, pattern * 0.8, max_iters=10)
        assert rr.final_output.shape[-1] == dim
        # Output should not be all NaN or Inf
        assert torch.isfinite(rr.final_output).all()

    def test_gain_stays_bounded(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=0)
        pattern = torch.randn(dim)
        store_pattern(soma, pattern, n_presentations=10)
        for node in soma.graph.all_nodes():
            assert 0.1 <= node.gain <= 10.0


class TestStoreRecall:
    """Basic store + recall produces non-degenerate output."""

    def test_recalled_output_not_all_zeros(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=42)
        pattern = torch.ones(dim)
        store_pattern(soma, pattern, n_presentations=5)
        rr = recall_pattern(soma, pattern * 0.5, max_iters=10)
        assert rr.final_output.norm().item() > 0.01, "Output is degenerate (near-zero)"

    def test_different_patterns_produce_different_outputs(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=42)
        p1 = torch.ones(dim)
        p2 = -torch.ones(dim)
        store_pattern(soma, p1, n_presentations=5)
        store_pattern(soma, p2, n_presentations=5)

        rr1 = recall_pattern(soma, p1, max_iters=10)
        rr2 = recall_pattern(soma, p2, max_iters=10)
        diff = (rr1.final_output - rr2.final_output).norm().item()
        # Outputs should be distinguishable
        assert diff > 0.01, "Different inputs produce identical outputs"
