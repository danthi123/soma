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

    def test_verbalizer_bootstrap_defaults(self) -> None:
        config = SOMAConfig()
        assert config.verbalizer_lr == pytest.approx(1e-4)
        assert config.verbalizer_checkpoint_interval == 500
        assert config.bootstrap_sample_tokens == 64
        assert config.bootstrap_max_steps == 5000

    def test_online_verbalizer_defaults(self) -> None:
        config = SOMAConfig()
        assert config.online_verbalizer_lr == pytest.approx(1e-5)
        assert config.online_batch_size == 4
        assert config.replay_buffer_capacity == 64
        assert config.divergence_window == 20
        assert config.divergence_threshold == pytest.approx(1.0)

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

    def test_verbalizer_bootstrap_fields_round_trip(self, tmp_path: Path) -> None:
        original = SOMAConfig(
            verbalizer_lr=3e-5,
            verbalizer_checkpoint_interval=100,
            bootstrap_sample_tokens=32,
            bootstrap_max_steps=1000,
        )
        yaml_path = tmp_path / "bootstrap.yaml"
        original.to_yaml(yaml_path)
        recovered = SOMAConfig.from_yaml(yaml_path)
        assert recovered.verbalizer_lr == pytest.approx(3e-5)
        assert recovered.verbalizer_checkpoint_interval == 100
        assert recovered.bootstrap_sample_tokens == 32
        assert recovered.bootstrap_max_steps == 1000

    def test_online_verbalizer_fields_round_trip(self, tmp_path: Path) -> None:
        original = SOMAConfig(
            online_verbalizer_lr=5e-6,
            online_batch_size=8,
            replay_buffer_capacity=128,
            divergence_window=50,
            divergence_threshold=2.0,
        )
        yaml_path = tmp_path / "online.yaml"
        original.to_yaml(yaml_path)
        recovered = SOMAConfig.from_yaml(yaml_path)
        assert recovered.online_verbalizer_lr == pytest.approx(5e-6)
        assert recovered.online_batch_size == 8
        assert recovered.replay_buffer_capacity == 128
        assert recovered.divergence_window == 50
        assert recovered.divergence_threshold == pytest.approx(2.0)

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


class TestVramSafetyFactor:
    def test_default_is_zero_point_nine(self) -> None:
        """Default 0.9 = reserve 10% headroom for OS/driver/KV-cache jitter.
        See devices.effective_vram_gb for the application site."""
        cfg = SOMAConfig()
        assert cfg.vram_safety_factor == pytest.approx(0.9)

    def test_round_trip_yaml(self, tmp_path: Path) -> None:
        original = SOMAConfig(vram_safety_factor=0.7)
        yaml_path = tmp_path / "vram.yaml"
        original.to_yaml(yaml_path)
        recovered = SOMAConfig.from_yaml(yaml_path)
        assert recovered.vram_safety_factor == pytest.approx(0.7)

    def test_round_trip_dict(self) -> None:
        original = SOMAConfig(vram_safety_factor=0.5)
        recovered = SOMAConfig.from_dict(original.to_dict())
        assert recovered.vram_safety_factor == pytest.approx(0.5)

    def test_factor_one_allowed(self) -> None:
        """1.0 means 'use full VRAM' -- valid, not the default but legal."""
        cfg = SOMAConfig(vram_safety_factor=1.0)
        assert cfg.vram_safety_factor == pytest.approx(1.0)

    def test_factor_just_above_zero_allowed(self) -> None:
        """Lower bound is exclusive zero -- tiny positive values allowed."""
        cfg = SOMAConfig(vram_safety_factor=0.01)
        assert cfg.vram_safety_factor == pytest.approx(0.01)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0, -1.0])
    def test_rejects_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="vram_safety_factor"):
            SOMAConfig(vram_safety_factor=bad)


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


class TestMemoryLayerPreset:
    """SOMAConfig.memory_layer() — product-tuned preset for the
    agent-memory use case. Derived from the v0.5 grounding-plasticity
    research (commits 2ba566b, c2a456b): enables positional-locality
    filter at the validated cutoff=0.5, uses moderate growth cadence
    suitable for accumulating user memories over time, and keeps
    activation_threshold tight so synap actually fires on small
    graphs typical of agent-memory workloads.
    """

    def test_returns_soma_config(self) -> None:
        cfg = SOMAConfig.memory_layer()
        assert isinstance(cfg, SOMAConfig)

    def test_enables_locality_filter_at_validated_cutoff(self) -> None:
        """The whole point of this preset: locality is ON by default."""
        cfg = SOMAConfig.memory_layer()
        assert cfg.synaptogenesis_max_distance == pytest.approx(0.5)

    def test_has_moderate_growth_cadence(self) -> None:
        """synap_interval should be tighter than whitepaper default (100)
        but not as aggressive as developmental (10). Memory workloads
        accumulate memories over thousands of interactions and need
        gradual structure formation, not burst growth."""
        cfg = SOMAConfig.memory_layer()
        assert 10 <= cfg.synaptogenesis_interval <= 50

    def test_activation_threshold_is_tight(self) -> None:
        """Tight threshold (not whitepaper's 0.1) so synap actually fires
        on small-graph workloads where activations rarely clear 0.1."""
        cfg = SOMAConfig.memory_layer()
        assert cfg.activation_threshold < 0.05

    def test_accepts_overrides(self) -> None:
        """Callers can still override any field."""
        cfg = SOMAConfig.memory_layer(synaptogenesis_max_distance=0.3, seed=7)
        assert cfg.synaptogenesis_max_distance == pytest.approx(0.3)
        assert cfg.seed == 7

    def test_override_to_disable_locality(self) -> None:
        """Callers who want vanilla behavior can turn locality off."""
        cfg = SOMAConfig.memory_layer(synaptogenesis_max_distance=0.0)
        assert cfg.synaptogenesis_max_distance == 0.0


class TestDistillationConfig:
    """Direction 4a: LLM-distilled projections config fields."""

    def test_default_distillation_target_is_none(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_target == "none"

    def test_accepts_llm_embedding_target(self) -> None:
        cfg = SOMAConfig(projection_distillation_target="llm_embedding")
        assert cfg.projection_distillation_target == "llm_embedding"

    def test_rejects_invalid_distillation_target(self) -> None:
        with pytest.raises(ValueError, match="projection_distillation_target"):
            SOMAConfig(projection_distillation_target="bogus")  # type: ignore[arg-type]

    def test_default_distillation_model_is_mxbai(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_model == "mxbai-embed-large"

    def test_default_distillation_base_url_is_local_ollama(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_base_url == "http://localhost:11434"

    def test_default_distillation_weight_is_one(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_weight == 1.0

    def test_rejects_negative_distillation_weight(self) -> None:
        with pytest.raises(ValueError, match="projection_distillation_weight"):
            SOMAConfig(projection_distillation_weight=-0.1)

    def test_accepts_zero_distillation_weight(self) -> None:
        # Zero weight is legal (disables the loss contribution without
        # touching the target mode — useful for ablations).
        cfg = SOMAConfig(projection_distillation_weight=0.0)
        assert cfg.projection_distillation_weight == 0.0
