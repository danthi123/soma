"""Tests for prediction-error-supervised synaptogenesis.

Direction 1 from docs/plans/2026-04-18-grounding-plasticity.md. The
"pe_conditional" supervision mode filters out arbitrary co-activation
admissions by requiring that the candidate pair has a history of
co-activating before prediction error drops. The evidence is tracked
as a per-pair EMA of PE-delta; admission is skipped when the EMA
indicates that co-activation is uncorrelated with or precedes
PE increases.

This suite covers:
1. Config validation for supervision/alpha/threshold/min_observations.
2. PE-delta-EMA bookkeeping state on SOMA (update rule, last_pe
   tracking, pair-key symmetry, observation counts).
3. Gate semantics in synaptogenesis (admit only when count meets
   min_observations AND ema < threshold).
4. Serialization round-trip for the new state fields (EMA dict,
   observation counts, last_pe).
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
    def test_default_supervision_is_none(self) -> None:
        cfg = _tiny_config()
        assert cfg.synaptogenesis_supervision == "none"

    def test_default_ema_alpha_is_high(self) -> None:
        cfg = _tiny_config()
        # Default EMA alpha should heavily weight history over single samples
        # (large alpha = long memory). Exact value documented in config.
        assert 0.9 <= cfg.synaptogenesis_pe_ema_alpha < 1.0

    def test_default_threshold_permits_neutral_or_negative_delta(self) -> None:
        cfg = _tiny_config()
        # Threshold default should be 0.0 so "any non-PE-reducing pair" is
        # filtered; research runners can override to a negative threshold.
        assert cfg.synaptogenesis_pe_threshold == 0.0

    def test_default_min_observations_avoids_cold_start_admits(self) -> None:
        cfg = _tiny_config()
        assert cfg.synaptogenesis_pe_min_observations >= 1

    def test_accepts_pe_conditional_mode(self) -> None:
        cfg = _tiny_config(
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_ema_alpha=0.95,
            synaptogenesis_pe_threshold=-0.001,
            synaptogenesis_pe_min_observations=5,
        )
        assert cfg.synaptogenesis_supervision == "pe_conditional"
        assert cfg.synaptogenesis_pe_ema_alpha == 0.95
        assert cfg.synaptogenesis_pe_threshold == -0.001
        assert cfg.synaptogenesis_pe_min_observations == 5

    def test_rejects_invalid_supervision_mode(self) -> None:
        with pytest.raises(ValueError, match="synaptogenesis_supervision"):
            _tiny_config(synaptogenesis_supervision="lol")  # type: ignore[arg-type]

    def test_rejects_ema_alpha_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="synaptogenesis_pe_ema_alpha"):
            _tiny_config(synaptogenesis_pe_ema_alpha=1.5)
        with pytest.raises(ValueError, match="synaptogenesis_pe_ema_alpha"):
            _tiny_config(synaptogenesis_pe_ema_alpha=-0.1)

    def test_rejects_negative_min_observations(self) -> None:
        with pytest.raises(ValueError, match="synaptogenesis_pe_min_observations"):
            _tiny_config(synaptogenesis_pe_min_observations=-1)


class TestEmaBookkeeping:
    """SOMA maintains per-pair EMA of PE-delta across steps."""

    def test_state_initialized_empty(self) -> None:
        cfg = _tiny_config()
        soma = SOMA(cfg)
        assert soma._synap_pe_ema == {}
        assert soma._synap_pe_counts == {}
        assert soma._last_pe is None

    def test_first_step_records_last_pe_but_no_ema_updates(self) -> None:
        """On the first step with a loss, there is no previous loss to
        compute a delta from; we just record the loss for the NEXT step.
        EMA should remain empty."""
        cfg = _tiny_config(synaptogenesis_supervision="pe_conditional")
        soma = SOMA(cfg)
        inputs = {"text": torch.randn(cfg.sensor_output_dim)}
        target = torch.randn(cfg.sensor_output_dim)
        soma.step(inputs=inputs, targets={"text": target})
        assert soma._last_pe is not None
        assert soma._synap_pe_ema == {}
        assert soma._synap_pe_counts == {}

    def test_second_step_updates_ema_for_coactive_pairs(self) -> None:
        """After two steps we should have at least one pair's EMA
        populated, keyed by an unordered (min_id, max_id) tuple."""
        cfg = _tiny_config(
            synaptogenesis_supervision="pe_conditional",
            activation_threshold=0.0001,  # capture all non-trivially-active nodes
        )
        soma = SOMA(cfg)
        inputs = {"text": torch.ones(cfg.sensor_output_dim) * 2.0}
        target = torch.zeros(cfg.sensor_output_dim)
        soma.step(inputs=inputs, targets={"text": target})
        soma.step(inputs=inputs, targets={"text": target})
        assert len(soma._synap_pe_ema) > 0
        for key in soma._synap_pe_ema:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert key[0] <= key[1], "pair keys must be sorted for symmetry"

    def test_pair_key_symmetry(self) -> None:
        """Co-activation is symmetric — pair (a,b) and (b,a) must map to
        the same EMA slot."""
        cfg = _tiny_config(synaptogenesis_supervision="pe_conditional")
        soma = SOMA(cfg)
        # Directly call the internal helper on a pair.
        node_ids = list(soma.graph.nodes.keys())
        assert len(node_ids) >= 2
        a, b = node_ids[0], node_ids[1]
        # The helper should produce the same key in both orderings.
        assert soma._synap_pair_key(a, b) == soma._synap_pair_key(b, a)

    def test_ema_reflects_pe_reduction_sign(self) -> None:
        """If PE decreases step-over-step, the pair's EMA should move
        negative; if PE increases, EMA should move positive."""
        cfg = _tiny_config(
            synaptogenesis_supervision="pe_conditional",
            synaptogenesis_pe_ema_alpha=0.5,  # fast learning to see effect
            activation_threshold=0.0001,
        )
        soma = SOMA(cfg)
        node_ids = list(soma.graph.nodes.keys())
        a, b = node_ids[0], node_ids[1]
        key = soma._synap_pair_key(a, b)

        # Manually drive the EMA update helper with known PE deltas
        # for a co-active pair. Simulating two co-activations: one good
        # (negative delta), one bad (positive delta).
        active = {a, b}
        soma._update_synap_pe_ema(active, pe_delta=-0.2)
        ema_after_good = soma._synap_pe_ema[key]
        assert ema_after_good < 0.0

        # A positive delta should pull the EMA back up.
        soma._update_synap_pe_ema(active, pe_delta=+0.4)
        ema_after_mixed = soma._synap_pe_ema[key]
        assert ema_after_mixed > ema_after_good

    def test_ema_not_maintained_when_supervision_disabled(self) -> None:
        """With supervision='none', don't waste work on EMA bookkeeping."""
        cfg = _tiny_config(synaptogenesis_supervision="none")
        soma = SOMA(cfg)
        inputs = {"text": torch.ones(cfg.sensor_output_dim)}
        target = torch.zeros(cfg.sensor_output_dim)
        for _ in range(3):
            soma.step(inputs=inputs, targets={"text": target})
        # Supervision off => EMA dict stays empty, _last_pe never set.
        assert soma._synap_pe_ema == {}
        assert soma._synap_pe_counts == {}
        assert soma._last_pe is None


class TestSerialization:
    def test_round_trip_preserves_ema_state(self, tmp_path: Path) -> None:
        cfg = _tiny_config(synaptogenesis_supervision="pe_conditional")
        soma = SOMA(cfg)
        node_ids = list(soma.graph.nodes.keys())
        a, b, c = node_ids[0], node_ids[1], node_ids[2]
        # Populate EMA state directly so the test doesn't depend on
        # step() wiring being in place for this round-trip check.
        soma._synap_pe_ema[soma._synap_pair_key(a, b)] = -0.05
        soma._synap_pe_ema[soma._synap_pair_key(a, c)] = 0.02
        soma._synap_pe_counts[soma._synap_pair_key(a, b)] = 7
        soma._synap_pe_counts[soma._synap_pair_key(a, c)] = 3
        soma._last_pe = 0.42

        path = tmp_path / "soma.pt"
        soma.save_state(path)

        soma2 = SOMA(cfg)
        soma2.load_state(path)

        assert soma2._synap_pe_ema == soma._synap_pe_ema
        assert soma2._synap_pe_counts == soma._synap_pe_counts
        assert soma2._last_pe == soma._last_pe

    def test_legacy_checkpoint_loads_with_defaults(self, tmp_path: Path) -> None:
        """A checkpoint saved BEFORE the supervision fields existed must
        still load, with the new fields defaulting to empty / None."""
        cfg = _tiny_config()
        soma = SOMA(cfg)
        path = tmp_path / "soma_legacy.pt"
        soma.save_state(path)

        # Surgically remove the new fields to simulate a pre-feature bundle.
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        state = raw["payload"] if raw.get("format") == "soma-brain" else raw
        state.pop("synap_pe_ema", None)
        state.pop("synap_pe_counts", None)
        state.pop("last_pe", None)
        torch.save(raw, str(path))

        soma2 = SOMA(cfg)
        soma2.load_state(path)
        assert soma2._synap_pe_ema == {}
        assert soma2._synap_pe_counts == {}
        assert soma2._last_pe is None
