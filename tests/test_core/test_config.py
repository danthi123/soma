"""Tests for ``soma.core.config.SOMAConfig``."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from soma.core.config import SOMAConfig


class TestDefaults:
    def test_defaults_match_whitepaper(self) -> None:
        config = SOMAConfig()
        # Spot-check whitepaper Section 10 constants.
        assert config.sensor_output_dim == 64
        assert config.associator_hidden_dim == 128
        assert config.integrator_hidden_dim == 256
        assert config.wm_slots == 32
        assert config.wm_dim == 128
        assert config.episodic_capacity == 10000
        assert config.vocab_size == 8192
        assert config.max_nodes == 50000
        assert config.base_lr == pytest.approx(0.001)
        assert config.hebbian_lr == pytest.approx(0.0001)
        assert config.max_edge_weight == pytest.approx(5.0)
        assert config.gain_min == pytest.approx(0.1)
        assert config.gain_max == pytest.approx(10.0)

    def test_default_modalities_independent(self) -> None:
        """Every ``SOMAConfig()`` must get its own list (no shared defaults)."""
        a = SOMAConfig()
        b = SOMAConfig()
        a.input_modalities.append("image")
        assert b.input_modalities == ["text"]

    def test_use_batched_executor_default_on(self) -> None:
        cfg = SOMAConfig()
        assert cfg.use_batched_executor is True

    def test_use_batched_executor_can_be_disabled(self) -> None:
        cfg = SOMAConfig(use_batched_executor=False)
        assert cfg.use_batched_executor is False


class TestFromDict:
    def test_override_subset(self) -> None:
        config = SOMAConfig.from_dict({"base_lr": 0.01, "wm_slots": 64})
        assert config.base_lr == pytest.approx(0.01)
        assert config.wm_slots == 64
        # Unchanged fields keep defaults.
        assert config.hebbian_lr == pytest.approx(0.0001)

    def test_drops_unknown_keys_with_warning(self) -> None:
        """Unknown keys (e.g. from older/newer checkpoints) warn, not crash."""
        d = SOMAConfig().to_dict()
        d["totally_new_field_from_future"] = 42
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = SOMAConfig.from_dict(d)
        # Sanity: still constructs.
        assert cfg.seed == SOMAConfig().seed
        # Warning was emitted naming the bad key.
        assert any("totally_new_field_from_future" in str(msg.message) for msg in w)

    def test_accepts_empty_dict(self) -> None:
        config = SOMAConfig.from_dict({})
        # Same as defaults.
        assert config.to_dict() == SOMAConfig().to_dict()


class TestFromYaml:
    def test_loads_default_yaml(self) -> None:
        """The shipped ``configs/default.yaml`` must parse cleanly."""
        yaml_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        config = SOMAConfig.from_yaml(yaml_path)
        # YAML must not diverge from dataclass defaults without reason;
        # spot-check a few values.
        assert config.sensor_output_dim == 64
        assert config.wm_slots == 32
        assert config.base_lr == pytest.approx(0.001)

    def test_custom_yaml_round_trip(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "custom.yaml"
        yaml_path.write_text(
            "base_lr: 0.005\nwm_slots: 16\ninput_modalities:\n  - text\n  - image\n",
            encoding="utf-8",
        )
        config = SOMAConfig.from_yaml(yaml_path)
        assert config.base_lr == pytest.approx(0.005)
        assert config.wm_slots == 16
        assert config.input_modalities == ["text", "image"]

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "list.yaml"
        yaml_path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            SOMAConfig.from_yaml(yaml_path)

    def test_accepts_empty_yaml(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("", encoding="utf-8")
        config = SOMAConfig.from_yaml(yaml_path)
        assert config.to_dict() == SOMAConfig().to_dict()


class TestToYaml:
    def test_round_trip(self, tmp_path: Path) -> None:
        original = SOMAConfig(base_lr=0.002, wm_slots=8, input_modalities=["text", "audio"])
        yaml_path = tmp_path / "round_trip.yaml"
        original.to_yaml(yaml_path)
        recovered = SOMAConfig.from_yaml(yaml_path)
        assert recovered.to_dict() == original.to_dict()

    def test_to_dict_is_independent(self) -> None:
        config = SOMAConfig()
        data = config.to_dict()
        data["input_modalities"].append("image")
        assert config.input_modalities == ["text"]


class TestValidation:
    @pytest.mark.parametrize(
        "field_name, bad_value",
        [
            ("sensor_output_dim", 0),
            ("wm_slots", -1),
            ("episodic_capacity", 0),
            ("max_nodes", -10),
            ("vocab_size", 0),
        ],
    )
    def test_positive_int_fields_reject_non_positive(self, field_name: str, bad_value: int) -> None:
        with pytest.raises(ValueError, match=field_name):
            SOMAConfig(**{field_name: bad_value})

    def test_rejects_negative_initial_counts(self) -> None:
        with pytest.raises(ValueError, match="initial_associator_count"):
            SOMAConfig(initial_associator_count=-1)

    def test_rejects_bad_decay_rate(self) -> None:
        with pytest.raises(ValueError, match="wm_decay_rate"):
            SOMAConfig(wm_decay_rate=0.0)
        with pytest.raises(ValueError, match="wm_decay_rate"):
            SOMAConfig(wm_decay_rate=1.5)

    def test_rejects_inverted_gain_range(self) -> None:
        with pytest.raises(ValueError, match="gain_min"):
            SOMAConfig(gain_min=5.0, gain_max=1.0)

    def test_rejects_empty_modalities(self) -> None:
        with pytest.raises(ValueError, match="input_modalities"):
            SOMAConfig(input_modalities=[])
        with pytest.raises(ValueError, match="output_modalities"):
            SOMAConfig(output_modalities=[])


class TestStabilityFields:
    def test_new_stability_fields_have_sane_defaults(self) -> None:
        cfg = SOMAConfig()
        assert 0.0 < cfg.edge_weight_decay <= 1.0
        assert cfg.edge_weight_decay > 0.99
        assert cfg.grad_clip_max_norm > 0.0
        assert cfg.max_consecutive_skipped_steps >= 1

    def test_edge_weight_decay_validated(self) -> None:
        with pytest.raises(ValueError, match="edge_weight_decay"):
            SOMAConfig(edge_weight_decay=0.0)
        with pytest.raises(ValueError, match="edge_weight_decay"):
            SOMAConfig(edge_weight_decay=1.5)

    def test_grad_clip_max_norm_validated(self) -> None:
        with pytest.raises(ValueError, match="grad_clip_max_norm"):
            SOMAConfig(grad_clip_max_norm=0.0)
        with pytest.raises(ValueError, match="grad_clip_max_norm"):
            SOMAConfig(grad_clip_max_norm=-1.0)

    def test_max_consecutive_skipped_steps_validated(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_skipped_steps"):
            SOMAConfig(max_consecutive_skipped_steps=0)


class TestYamlConsistency:
    def test_default_yaml_matches_dataclass_defaults(self) -> None:
        """The shipped ``configs/default.yaml`` should match the dataclass.

        If this test starts failing, either update the YAML, or intentionally
        override in the test. The two should stay in sync by default.
        """
        yaml_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        with yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        yaml_config = SOMAConfig.from_dict(raw)
        default_config = SOMAConfig()
        assert yaml_config.to_dict() == default_config.to_dict()
