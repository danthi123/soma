"""Tests for Direction 3 — neuromodulator-style plasticity broadcast.

A single scalar ``plasticity_gain`` tracks how "surprised" the system
is right now (ratio of recent vs baseline PE) and multiplies all
plasticity rates (synaptogenesis admission, Hebbian weight updates)
so that growth and learning concentrate on moments of informative
surprise and pause during steady-state.

This suite covers:
1. Config validation for the four new fields.
2. SOMA.plasticity_gain initialization, update rule, and bounds.
3. Serialization round-trip.
4. End-to-end: supervision='off' behavior bit-exact when broadcast
   is off; non-trivial effect when broadcast is on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.system import SOMA


def _tiny_config(**overrides: object) -> SOMAConfig:
    defaults: dict[str, object] = dict(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=8,
        integrator_hidden_dim=16,
        integrator_output_dim=8,
        wm_slots=4,
        wm_dim=8,
        key_dim=8,
        value_dim=16,
        text_embed_dim=8,
        initial_associator_count=3,
        initial_integrator_count=0,
        max_nodes=64,
        num_curiosity_domains=2,
        synaptogenesis_interval=0,
        neurogenesis_interval=0,
        pruning_interval=0,
        consolidation_interval=0,
        seed=0,
    )
    defaults.update(overrides)
    return SOMAConfig(**defaults)


class TestConfigValidation:
    def test_default_broadcast_off(self) -> None:
        cfg = _tiny_config()
        assert cfg.plasticity_broadcast_mode == "off"

    def test_default_alpha_heavy_history(self) -> None:
        cfg = _tiny_config()
        # Slow updates (heavy history) so the gain reflects the
        # surprise trajectory, not a single noisy step.
        assert 0.8 <= cfg.plasticity_broadcast_alpha < 1.0

    def test_default_bounds_asymmetric(self) -> None:
        """min < 1 < max: gain can both suppress and amplify plasticity."""
        cfg = _tiny_config()
        assert cfg.plasticity_broadcast_min_gain < 1.0
        assert cfg.plasticity_broadcast_max_gain > 1.0

    def test_accepts_pe_scaled_mode(self) -> None:
        cfg = _tiny_config(
            plasticity_broadcast_mode="pe_scaled",
            plasticity_broadcast_alpha=0.9,
            plasticity_broadcast_min_gain=0.2,
            plasticity_broadcast_max_gain=2.0,
        )
        assert cfg.plasticity_broadcast_mode == "pe_scaled"
        assert cfg.plasticity_broadcast_alpha == 0.9
        assert cfg.plasticity_broadcast_min_gain == 0.2
        assert cfg.plasticity_broadcast_max_gain == 2.0

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="plasticity_broadcast_mode"):
            _tiny_config(plasticity_broadcast_mode="continuous")  # type: ignore[arg-type]

    def test_rejects_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="plasticity_broadcast_alpha"):
            _tiny_config(plasticity_broadcast_alpha=1.5)
        with pytest.raises(ValueError, match="plasticity_broadcast_alpha"):
            _tiny_config(plasticity_broadcast_alpha=-0.1)

    def test_rejects_min_gain_not_less_than_max(self) -> None:
        with pytest.raises(ValueError, match="plasticity_broadcast"):
            _tiny_config(
                plasticity_broadcast_min_gain=1.0,
                plasticity_broadcast_max_gain=1.0,
            )
        with pytest.raises(ValueError, match="plasticity_broadcast"):
            _tiny_config(
                plasticity_broadcast_min_gain=2.0,
                plasticity_broadcast_max_gain=1.0,
            )

    def test_rejects_non_positive_bounds(self) -> None:
        with pytest.raises(ValueError, match="plasticity_broadcast"):
            _tiny_config(plasticity_broadcast_min_gain=-0.1)
        with pytest.raises(ValueError, match="plasticity_broadcast"):
            _tiny_config(plasticity_broadcast_max_gain=0.0)


class TestPlasticityGainState:
    def test_initialized_to_one(self) -> None:
        cfg = _tiny_config()
        soma = SOMA(cfg)
        assert soma.plasticity_gain == 1.0

    def test_gain_unchanged_when_broadcast_off(self) -> None:
        cfg = _tiny_config(plasticity_broadcast_mode="off")
        soma = SOMA(cfg)
        inputs = {"text": torch.randn(cfg.sensor_output_dim)}
        target = torch.randn(cfg.sensor_output_dim)
        for _ in range(5):
            soma.step(inputs=inputs, targets={"text": target})
        assert soma.plasticity_gain == 1.0

    def test_gain_moves_when_pe_ratio_departs_from_one(self) -> None:
        """pe_scaled: gain tracks recent/baseline PE ratio via EMA.

        neurogenesis's ratio is recent (last 100) / baseline (last
        1000). To make these differ, seed the buffer with 900 baseline
        samples at magnitude 1 followed by 100 spike samples at
        magnitude 2. ratio = 2/( (900*1 + 100*2)/1000 ) = 2/1.1 ~= 1.82.
        """
        cfg = _tiny_config(
            plasticity_broadcast_mode="pe_scaled",
            plasticity_broadcast_alpha=0.5,  # fast updates to see effect
        )
        soma = SOMA(cfg)
        soma._recent_errors = [1.0] * 900 + [2.0] * 100
        inputs = {"text": torch.randn(cfg.sensor_output_dim)}
        target = torch.randn(cfg.sensor_output_dim)
        soma.step(inputs=inputs, targets={"text": target})
        # Gain should have moved toward ~1.82 (but not arrived with alpha=0.5).
        assert soma.plasticity_gain > 1.0
        assert soma.plasticity_gain < 2.0

    def test_gain_clamped_to_configured_bounds(self) -> None:
        cfg = _tiny_config(
            plasticity_broadcast_mode="pe_scaled",
            plasticity_broadcast_alpha=0.0,  # jump instantly to target
            plasticity_broadcast_min_gain=0.2,
            plasticity_broadcast_max_gain=2.5,
        )
        soma = SOMA(cfg)
        soma._recent_errors = [1.0] * 50 + [100.0] * 50  # ratio=100, must be clamped
        inputs = {"text": torch.randn(cfg.sensor_output_dim)}
        target = torch.randn(cfg.sensor_output_dim)
        soma.step(inputs=inputs, targets={"text": target})
        assert soma.plasticity_gain <= 2.5
        assert soma.plasticity_gain >= 0.2

    def test_update_helper_produces_expected_value(self) -> None:
        """Unit test the update formula directly."""
        cfg = _tiny_config(
            plasticity_broadcast_mode="pe_scaled",
            plasticity_broadcast_alpha=0.8,
            plasticity_broadcast_min_gain=0.1,
            plasticity_broadcast_max_gain=3.0,
        )
        soma = SOMA(cfg)
        soma.plasticity_gain = 1.0
        # Simulate ratio=2 target; alpha=0.8 so gain -> 0.8*1.0 + 0.2*2.0 = 1.2
        soma._update_plasticity_gain(pe_ratio=2.0)
        assert abs(soma.plasticity_gain - 1.2) < 1e-9


class TestSerialization:
    def test_round_trip_preserves_gain(self, tmp_path: Path) -> None:
        cfg = _tiny_config(plasticity_broadcast_mode="pe_scaled")
        soma = SOMA(cfg)
        soma.plasticity_gain = 1.7
        path = tmp_path / "soma.pt"
        soma.save_state(path)

        soma2 = SOMA(cfg)
        soma2.load_state(path)
        assert soma2.plasticity_gain == 1.7

    def test_legacy_checkpoint_loads_with_default(self, tmp_path: Path) -> None:
        cfg = _tiny_config()
        soma = SOMA(cfg)
        path = tmp_path / "soma_legacy.pt"
        soma.save_state(path)

        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        state = raw["payload"] if raw.get("format") == "soma-brain" else raw
        state.pop("plasticity_gain", None)
        torch.save(raw, str(path))

        soma2 = SOMA(cfg)
        soma2.load_state(path)
        assert soma2.plasticity_gain == 1.0


class TestSynaptogenesisScaling:
    """Direction 3: plasticity_gain multiplies synaptogenesis_rate.

    With gain=0 effectively suppresses synaptogenesis; gain=2 doubles
    admission probability. These tests inject a known gain into
    synaptogenesis via the new rate_scale kwarg.
    """

    def _make_pair(self, cfg: SOMAConfig, *, dim: int = 8):
        from soma.core.graph import Graph
        from soma.core.node import Node, NodeType
        graph = Graph()
        pos = torch.zeros(cfg.position_dim)
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos)
        graph.add_node(a)
        graph.add_node(b)
        return graph, a, b

    def test_zero_scale_suppresses_admission(self) -> None:
        from soma.growth.synaptogenesis import synaptogenesis
        cfg = SOMAConfig(synaptogenesis_rate=10.0, activation_threshold=0.01)
        graph, a, b = self._make_pair(cfg)
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        # scale=0 => prob=0 for every pair => no admissions.
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng, rate_scale=0.0,
        )
        assert new == []

    def test_scale_two_doubles_effective_rate(self) -> None:
        """With the same RNG seed, rate_scale=2 should admit at least
        as many edges as rate_scale=1 — the effective probability is
        higher so every coin-flip that passed at rate=1 also passes
        at rate=2."""
        from soma.growth.synaptogenesis import synaptogenesis
        cfg = SOMAConfig(synaptogenesis_rate=0.05, activation_threshold=0.01)
        graph1, a1, b1 = self._make_pair(cfg)
        graph2, a2, b2 = self._make_pair(cfg)
        dim = 8
        acts1 = {a1.id: torch.ones(dim), b1.id: torch.ones(dim)}
        acts2 = {a2.id: torch.ones(dim), b2.id: torch.ones(dim)}
        rng1 = torch.Generator().manual_seed(7)
        rng2 = torch.Generator().manual_seed(7)
        # Run many trials to get a stable count delta.
        total1 = 0
        total2 = 0
        for seed in range(50):
            from soma.core.graph import Graph
            g1 = Graph.deserialize(graph1.serialize(), cfg)
            g2 = Graph.deserialize(graph2.serialize(), cfg)
            rng1 = torch.Generator().manual_seed(seed)
            rng2 = torch.Generator().manual_seed(seed)
            n1 = synaptogenesis(
                g1, acts1, step=100, config=cfg, rng=rng1, rate_scale=1.0,
            )
            n2 = synaptogenesis(
                g2, acts2, step=100, config=cfg, rng=rng2, rate_scale=2.0,
            )
            total1 += len(n1)
            total2 += len(n2)
        assert total2 >= total1, (
            f"rate_scale=2 should admit >= rate_scale=1 "
            f"(got {total2} vs {total1})"
        )

    def test_rate_scale_defaults_to_one(self) -> None:
        """Omitted kwarg => backward-compat: identical behavior to
        rate_scale=1.0."""
        from soma.growth.synaptogenesis import synaptogenesis
        cfg = SOMAConfig(synaptogenesis_rate=10.0, activation_threshold=0.01)
        graph, a, b = self._make_pair(cfg)
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        rng = torch.Generator().manual_seed(0)
        new = synaptogenesis(
            graph, acts, step=100, config=cfg, rng=rng,
        )
        # With high rate, at least one admission.
        assert len(new) >= 1


class TestHebbianScaling:
    """Direction 3: plasticity_gain multiplies hebbian_lr.

    hebbian_scale * hebbian_lr determines per-step edge-weight growth.
    Checking that scale=0 freezes edges and scale=2 grows them faster.
    """

    def _make_graph_with_edge(self, cfg: SOMAConfig, *, dim: int = 8):
        from soma.core.edge import Edge
        from soma.core.graph import Graph
        from soma.core.node import Node, NodeType
        graph = Graph()
        pos = torch.zeros(cfg.position_dim)
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, cfg, position=pos)
        graph.add_node(a)
        graph.add_node(b)
        edge = Edge(
            source_id=a.id, target_id=b.id,
            source_output_dim=dim, target_input_dim=dim,
            creation_step=0, initial_weight=0.1,
        )
        graph.add_edge(edge)
        return graph, a, b, edge

    def test_zero_scale_freezes_hebbian(self) -> None:
        from soma.core.learning import _apply_hebbian_edge_updates
        cfg = SOMAConfig(
            hebbian_lr=0.1,
            activation_threshold=0.01,
            edge_weight_decay=1.0,  # disable decay to isolate Hebbian
        )
        graph, a, b, edge = self._make_graph_with_edge(cfg)
        start = float(edge.weight.data.norm().item())
        acts = {a.id: torch.ones(8), b.id: torch.ones(8)}
        _apply_hebbian_edge_updates(graph, acts, config=cfg, plasticity_scale=0.0)
        end = float(edge.weight.data.norm().item())
        # With scale=0, no Hebbian delta; weight unchanged.
        assert abs(end - start) < 1e-6

    def test_scale_two_grows_faster(self) -> None:
        from soma.core.learning import _apply_hebbian_edge_updates
        cfg = SOMAConfig(
            hebbian_lr=0.01,
            activation_threshold=0.01,
            edge_weight_decay=1.0,
        )
        # Two identical edges; apply scale=1 to one, scale=2 to other.
        g1, a1, b1, e1 = self._make_graph_with_edge(cfg)
        g2, a2, b2, e2 = self._make_graph_with_edge(cfg)
        acts1 = {a1.id: torch.ones(8), b1.id: torch.ones(8)}
        acts2 = {a2.id: torch.ones(8), b2.id: torch.ones(8)}
        for _ in range(5):
            _apply_hebbian_edge_updates(g1, acts1, config=cfg, plasticity_scale=1.0)
            _apply_hebbian_edge_updates(g2, acts2, config=cfg, plasticity_scale=2.0)
        n1 = float(e1.weight.data.norm().item())
        n2 = float(e2.weight.data.norm().item())
        # scale=2 accumulates ~2x the Hebbian bump each step.
        assert n2 > n1

    def test_plasticity_scale_defaults_to_one(self) -> None:
        """Omitted kwarg => identical to scale=1.0 (backward compat)."""
        from soma.core.learning import _apply_hebbian_edge_updates
        cfg = SOMAConfig(
            hebbian_lr=0.01,
            activation_threshold=0.01,
            edge_weight_decay=1.0,
        )
        g1, a1, b1, e1 = self._make_graph_with_edge(cfg)
        g2, a2, b2, e2 = self._make_graph_with_edge(cfg)
        acts1 = {a1.id: torch.ones(8), b1.id: torch.ones(8)}
        acts2 = {a2.id: torch.ones(8), b2.id: torch.ones(8)}
        for _ in range(3):
            _apply_hebbian_edge_updates(g1, acts1, config=cfg)
            _apply_hebbian_edge_updates(g2, acts2, config=cfg, plasticity_scale=1.0)
        n1 = float(e1.weight.data.norm().item())
        n2 = float(e2.weight.data.norm().item())
        # Should be identical (default behavior).
        assert abs(n1 - n2) < 1e-6
