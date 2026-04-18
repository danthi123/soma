"""Tests for prediction-error-gated neurogenesis.

The "pe_gated" mode replaces interval-based polling with
every-step eligibility subject to a cooldown. This suite covers:
1. config validation for neurogenesis_mode and cooldown
2. gate semantics (interval vs pe_gated)
3. cooldown enforcement in pe_gated mode
4. homeostasis still overrides (allow_neurogenesis=False blocks both modes)
5. cooldown state round-trips through save/load
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
        neurogenesis_interval=200,
        pruning_interval=0,
        consolidation_interval=0,
        seed=0,
    )
    defaults.update(overrides)
    return SOMAConfig(**defaults)


class TestConfigValidation:
    def test_default_mode_is_interval(self) -> None:
        cfg = _tiny_config()
        assert cfg.neurogenesis_mode == "interval"
        assert cfg.neurogenesis_cooldown == 200

    def test_accepts_pe_gated_mode(self) -> None:
        cfg = _tiny_config(neurogenesis_mode="pe_gated", neurogenesis_cooldown=50)
        assert cfg.neurogenesis_mode == "pe_gated"
        assert cfg.neurogenesis_cooldown == 50

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="neurogenesis_mode"):
            _tiny_config(neurogenesis_mode="continuous")  # type: ignore[arg-type]

    def test_rejects_negative_cooldown(self) -> None:
        with pytest.raises(ValueError, match="neurogenesis_cooldown"):
            _tiny_config(neurogenesis_cooldown=-1)


class TestGateSemantics:
    def test_interval_mode_only_fires_on_aligned_steps(self) -> None:
        cfg = _tiny_config(neurogenesis_mode="interval", neurogenesis_interval=50)
        soma = SOMA(cfg)
        # Off-aligned step: gate closed.
        soma.global_step = 49
        assert soma._should_attempt_neurogenesis() is False
        # Aligned step: gate open (homeostasis lets it through by default).
        soma.global_step = 50
        assert soma._should_attempt_neurogenesis() is True
        soma.global_step = 100
        assert soma._should_attempt_neurogenesis() is True

    def test_pe_gated_mode_open_every_step_past_cooldown(self) -> None:
        cfg = _tiny_config(neurogenesis_mode="pe_gated", neurogenesis_cooldown=20)
        soma = SOMA(cfg)
        # Sentinel last-fire is very negative → first step eligible.
        soma.global_step = 1
        assert soma._should_attempt_neurogenesis() is True
        # After a firing, cooldown blocks the next few steps.
        soma._last_neurogenesis_step = 10
        soma.global_step = 15
        assert soma._should_attempt_neurogenesis() is False
        soma.global_step = 29
        assert soma._should_attempt_neurogenesis() is False
        # Step 30 exactly clears cooldown (diff == cooldown).
        soma.global_step = 30
        assert soma._should_attempt_neurogenesis() is True

    def test_pe_gated_ignores_interval_alignment(self) -> None:
        cfg = _tiny_config(
            neurogenesis_mode="pe_gated",
            neurogenesis_interval=50,
            neurogenesis_cooldown=5,
        )
        soma = SOMA(cfg)
        # Step 7 is not aligned on 50, but pe_gated still opens the gate.
        soma._last_neurogenesis_step = 1
        soma.global_step = 7
        assert soma._should_attempt_neurogenesis() is True

    def test_interval_zero_closes_gate_in_interval_mode(self) -> None:
        cfg = _tiny_config(neurogenesis_mode="interval", neurogenesis_interval=0)
        soma = SOMA(cfg)
        soma.global_step = 0
        assert soma._should_attempt_neurogenesis() is False
        soma.global_step = 100
        assert soma._should_attempt_neurogenesis() is False


class TestHomeostasisOverride:
    def test_homeostasis_blocks_both_modes(self) -> None:
        for mode in ("interval", "pe_gated"):
            cfg = _tiny_config(
                neurogenesis_mode=mode,
                neurogenesis_interval=1,
                neurogenesis_cooldown=1,
            )
            soma = SOMA(cfg)
            soma.homeostasis.allow_neurogenesis = False  # force-block
            soma.global_step = 10
            assert soma._should_attempt_neurogenesis() is False, (
                f"homeostasis should veto {mode} mode"
            )


class TestSerialization:
    def test_last_neurogenesis_step_survives_roundtrip(
        self, tmp_path: Path,
    ) -> None:
        cfg = _tiny_config(neurogenesis_mode="pe_gated", neurogenesis_cooldown=50)
        soma = SOMA(cfg)
        soma._last_neurogenesis_step = 777
        soma.global_step = 1000
        path = tmp_path / "brain.pt"
        soma.save_state(path)

        restored = SOMA(cfg)
        restored.load_state(path)
        assert restored._last_neurogenesis_step == 777

    def test_legacy_checkpoint_gets_sentinel_default(
        self, tmp_path: Path,
    ) -> None:
        # Simulate a checkpoint that pre-dates the field by saving, stripping
        # the key, and reloading. The restored SOMA should treat the first
        # step after load as cooldown-eligible.
        cfg = _tiny_config(neurogenesis_mode="pe_gated", neurogenesis_cooldown=50)
        soma = SOMA(cfg)
        soma.global_step = 500
        path = tmp_path / "brain.pt"
        soma.save_state(path)

        from soma.core.brain_bundle import unwrap_payload, wrap_payload
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        payload, _meta = unwrap_payload(raw)
        payload.pop("last_neurogenesis_step", None)
        stripped = wrap_payload(payload, soma_version="test")
        torch.save(stripped, str(path))

        restored = SOMA(cfg)
        restored.load_state(path)
        # Sentinel large negative: any step clears cooldown immediately.
        restored.global_step = 501
        assert restored._should_attempt_neurogenesis() is True
