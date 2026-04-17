"""Unit tests for SOMA attractor mode (Research D2 + D3).

Tests:
- Construction: SOMA builds with attractor config, has integrators
- Output shape: step() returns outputs of the right shape
- Convergence: iterated recall produces outputs that stabilize
- Homeostasis: attractor mode doesn't crash with homeostasis active
- Store/recall: basic store+recall produces non-degenerate output
- D3 output scaler: z-score and learned scaler fix magnitude attenuation
- D3 structural plasticity: trigger_neurogenesis / trigger_synaptogenesis
"""

from __future__ import annotations

import torch

from soma.research.attractor_mode import (
    OutputScaler,
    ZScoreScaler,
    count_nodes_by_type,
    fit_output_scaler,
    fit_zscore_scaler,
    make_attractor_config,
    make_attractor_soma,
    measure_recall_accuracy,
    recall_pattern,
    store_pattern,
    trigger_neurogenesis,
    trigger_synaptogenesis,
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


# ======================================================================
# D3: Output scaler tests
# ======================================================================


class TestZScoreScaler:
    """Z-score scaler normalisation."""

    def test_zscore_scaler_construction(self) -> None:
        scaler = ZScoreScaler(
            mean=torch.zeros(16),
            std=torch.ones(16),
        )
        x = torch.randn(16)
        out = scaler.transform(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-5)

    def test_zscore_scaler_rescales(self) -> None:
        # If input has mean=2, std=0.5, scaler should normalise to 0-mean, 1-std
        mean = torch.full((8,), 2.0)
        std = torch.full((8,), 0.5)
        scaler = ZScoreScaler(mean=mean, std=std)
        x = torch.full((8,), 2.5)  # (2.5 - 2.0) / 0.5 = 1.0
        out = scaler.transform(x)
        assert torch.allclose(out, torch.ones(8), atol=1e-5)

    def test_fit_zscore_on_soma(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=42)
        patterns = torch.sign(torch.randn(3, dim))
        for i in range(3):
            store_pattern(soma, patterns[i], n_presentations=3)
        scaler = fit_zscore_scaler(soma, patterns, n_iters=10)
        assert scaler.mean.shape == (dim,)
        assert scaler.std.shape == (dim,)
        # Std should be positive
        assert (scaler.std > 0).all()


class TestOutputScaler:
    """Learned affine output scaler."""

    def test_output_scaler_forward(self) -> None:
        scaler = OutputScaler(dim=16)
        x = torch.randn(16)
        out = scaler(x)
        assert out.shape == (16,)
        # Initially scale=1, bias=0 => identity
        assert torch.allclose(out, x, atol=1e-5)

    def test_output_scaler_learns(self) -> None:
        """Scaler should learn to rescale small inputs to +/-1."""
        dim = 8
        # Simulate: SOMA outputs are 0.1 * pattern
        patterns = torch.sign(torch.randn(5, dim))
        raw = patterns * 0.1

        scaler = OutputScaler(dim)
        optimizer = torch.optim.Adam(scaler.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()
        for _ in range(200):
            optimizer.zero_grad()
            pred = scaler(raw)
            loss = loss_fn(pred, patterns)
            loss.backward()
            optimizer.step()

        scaler.train(False)
        with torch.no_grad():
            pred = scaler(raw)
        # Should be close to the original patterns
        assert ((pred.sign() == patterns).float().mean().item()) > 0.8

    def test_fit_output_scaler_on_soma(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=42)
        patterns = torch.sign(torch.randn(3, dim))
        for i in range(3):
            store_pattern(soma, patterns[i], n_presentations=3)
        scaler = fit_output_scaler(soma, patterns, n_iters=10, train_steps=50)
        assert isinstance(scaler, OutputScaler)
        # Scaler should have learned parameters
        assert scaler.scale.shape == (dim,)


# ======================================================================
# D3: Structural plasticity tests
# ======================================================================


class TestStructuralPlasticity:
    """Trigger neurogenesis / synaptogenesis from outside SOMA.step()."""

    def test_trigger_neurogenesis_adds_nodes(self) -> None:
        dim = 16
        soma = make_attractor_soma(
            pattern_dim=dim, n_associators=4, n_integrators=2, seed=42
        )
        before = soma.graph.num_nodes
        new_ids = trigger_neurogenesis(soma, num_new_nodes=2)
        after = soma.graph.num_nodes
        assert after > before
        assert len(new_ids) == 2
        for nid in new_ids:
            assert nid in soma.graph.nodes

    def test_trigger_synaptogenesis_adds_edges(self) -> None:
        dim = 16
        soma = make_attractor_soma(
            pattern_dim=dim, n_associators=4, n_integrators=2, seed=42
        )
        before = soma.graph.num_edges
        new_ids = trigger_synaptogenesis(soma)
        after = soma.graph.num_edges
        # May or may not add edges depending on random draws, but should not crash
        assert after >= before
        assert isinstance(new_ids, list)

    def test_neurogenesis_preserves_functionality(self) -> None:
        """SOMA still works after adding nodes."""
        dim = 16
        soma = make_attractor_soma(
            pattern_dim=dim, n_associators=4, n_integrators=2, seed=42
        )
        pattern = torch.sign(torch.randn(dim))
        store_pattern(soma, pattern, n_presentations=3)
        trigger_neurogenesis(soma, num_new_nodes=2)
        # Should still be able to recall
        rr = recall_pattern(soma, pattern, max_iters=10)
        assert torch.isfinite(rr.final_output).all()

    def test_count_nodes_by_type(self) -> None:
        dim = 16
        soma = make_attractor_soma(
            pattern_dim=dim, n_associators=4, n_integrators=2, seed=42
        )
        counts = count_nodes_by_type(soma)
        assert counts.get("ASSOCIATOR", 0) >= 4
        assert counts.get("INTEGRATOR", 0) >= 2
        assert counts.get("SENSOR", 0) >= 1
        assert counts.get("OUTPUT", 0) >= 1


class TestMeasureRecallAccuracy:
    """measure_recall_accuracy helper."""

    def test_returns_metrics(self) -> None:
        dim = 16
        soma = make_attractor_soma(pattern_dim=dim, seed=42)
        patterns = torch.sign(torch.randn(3, dim))
        probes = patterns.clone()
        for i in range(3):
            store_pattern(soma, patterns[i], n_presentations=3)
        metrics = measure_recall_accuracy(soma, patterns, probes, n_iters=10)
        assert "exact_rate" in metrics
        assert "nearest_rate" in metrics
        assert "per_bit_acc" in metrics
        assert 0.0 <= metrics["exact_rate"] <= 1.0
        assert 0.0 <= metrics["nearest_rate"] <= 1.0
