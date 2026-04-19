"""Regression tests for SomaAdapter config flow-through.

Added after the adapter was extended with synap_locality / synap_interval /
synap_rate / seed kwargs during the 2026-04-19 locality investigation.
Strict TDD would have produced these tests first; they're added post-hoc
to protect the intended behavior going forward, not as TDD.
"""

from __future__ import annotations

import pytest

from benchmarks.harness.adapters.soma import SomaAdapter


@pytest.fixture
def attached_adapter():
    """Build an adapter with attach_soma=True and prepare it."""
    def _build(**overrides):
        adapter = SomaAdapter(
            use_sbert=False,
            attach_soma=True,
            **overrides,
        )
        adapter.prepare()
        return adapter
    return _build


class TestSynapLocalityFlowThrough:
    def test_default_locality_is_zero_disabled(self, attached_adapter) -> None:
        """Without explicit synap_locality, the filter is disabled."""
        a = attached_adapter()
        soma = a._mem._soma
        assert soma.config.synaptogenesis_max_distance == 0.0

    def test_custom_locality_propagates_to_config(self, attached_adapter) -> None:
        """synap_locality=0.5 reaches the attached SOMA's config."""
        a = attached_adapter(synap_locality=0.5)
        soma = a._mem._soma
        assert soma.config.synaptogenesis_max_distance == 0.5


class TestActiveGrowthOverride:
    """When synap_interval or synap_rate is set, adapter uses
    SOMAConfig.developmental() as the base so active growth is
    actually effective (activation_threshold=0.005 vs whitepaper's 0.1)."""

    def test_default_uses_whitepaper_config(self, attached_adapter) -> None:
        """Without growth overrides, use plain SOMAConfig defaults."""
        a = attached_adapter()
        soma = a._mem._soma
        # Plain SOMAConfig default: activation_threshold=0.1
        assert soma.config.activation_threshold == pytest.approx(0.1)
        # Plain SOMAConfig default: synaptogenesis_interval=100
        assert soma.config.synaptogenesis_interval == 100

    def test_synap_interval_override_triggers_developmental(
        self, attached_adapter,
    ) -> None:
        """Setting synap_interval switches base config to developmental()."""
        a = attached_adapter(synap_interval=10)
        soma = a._mem._soma
        assert soma.config.synaptogenesis_interval == 10
        # developmental() sets activation_threshold=0.005
        assert soma.config.activation_threshold == pytest.approx(0.005)

    def test_synap_rate_override_triggers_developmental(
        self, attached_adapter,
    ) -> None:
        """Setting synap_rate switches base config to developmental()."""
        a = attached_adapter(synap_rate=2.0)
        soma = a._mem._soma
        assert soma.config.synaptogenesis_rate == pytest.approx(2.0)
        assert soma.config.activation_threshold == pytest.approx(0.005)

    def test_both_overrides_compose(self, attached_adapter) -> None:
        """Setting both interval and rate works correctly."""
        a = attached_adapter(synap_interval=10, synap_rate=2.0)
        soma = a._mem._soma
        assert soma.config.synaptogenesis_interval == 10
        assert soma.config.synaptogenesis_rate == pytest.approx(2.0)

    def test_locality_plus_growth_overrides_compose(
        self, attached_adapter,
    ) -> None:
        """All four overrides (locality + interval + rate + seed) propagate."""
        a = attached_adapter(
            synap_locality=0.5,
            synap_interval=10,
            synap_rate=2.0,
            seed=42,
        )
        soma = a._mem._soma
        assert soma.config.synaptogenesis_max_distance == 0.5
        assert soma.config.synaptogenesis_interval == 10
        assert soma.config.synaptogenesis_rate == pytest.approx(2.0)
        assert soma.config.seed == 42


class TestSeedReproducibility:
    def test_same_seed_produces_same_initial_graph_state(
        self, attached_adapter,
    ) -> None:
        """Two adapters with the same seed must produce identical initial
        graphs (node count, edge count). This is the paired-comparison
        guarantee that locality-on vs locality-off experiments depend on."""
        a0 = attached_adapter(seed=7)
        a1 = attached_adapter(seed=7)
        s0 = a0._mem._soma
        s1 = a1._mem._soma
        assert s0.graph.num_nodes == s1.graph.num_nodes
        assert s0.graph.num_edges == s1.graph.num_edges

    def test_different_seeds_can_produce_different_states(
        self, attached_adapter,
    ) -> None:
        """Different seeds should potentially yield different state.
        (Initial graphs may happen to coincide in size — we don't require
        them to differ, only that seed is actually being used.)"""
        a0 = attached_adapter(seed=0)
        a1 = attached_adapter(seed=999)
        assert a0._mem._soma.config.seed == 0
        assert a1._mem._soma.config.seed == 999


class TestDistillationFlowThrough:
    """Direction 4a: distillation params propagate to SOMAConfig."""

    def test_default_distillation_target_is_none(self, attached_adapter) -> None:
        a = attached_adapter()
        soma = a._mem._soma
        assert soma.config.projection_distillation_target == "none"

    def test_distillation_target_propagates(self, attached_adapter) -> None:
        a = attached_adapter(
            projection_mode="learnable",
            projection_distillation_target="llm_embedding",
        )
        soma = a._mem._soma
        assert soma.config.projection_distillation_target == "llm_embedding"
        assert soma.config.projection_mode == "learnable"

    def test_distillation_model_propagates(self, attached_adapter) -> None:
        a = attached_adapter(
            projection_mode="learnable",
            projection_distillation_target="llm_embedding",
            projection_distillation_model="nomic-embed-text",
        )
        soma = a._mem._soma
        assert soma.config.projection_distillation_model == "nomic-embed-text"

    def test_distillation_weight_propagates(self, attached_adapter) -> None:
        a = attached_adapter(
            projection_mode="learnable",
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=0.3,
        )
        soma = a._mem._soma
        assert soma.config.projection_distillation_weight == pytest.approx(0.3)
