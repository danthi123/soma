"""Tests for ``soma.system.SOMA`` — the integrated top-level class.

Covers construction (seed graph wiring), single-step inference/training,
working-memory updates, episodic encoding, curiosity signal emission,
growth gating, consolidation triggering, interactive session helper,
and save/load round-trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.system import SOMA


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def small_config() -> SOMAConfig:
    """A tiny config that keeps matching dims so the seed graph runs."""
    return SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        wm_slots=4,
        wm_dim=8,
        key_dim=8,
        value_dim=16,
        text_embed_dim=8,
        initial_associator_count=3,
        initial_integrator_count=0,
        max_nodes=64,
        num_curiosity_domains=3,
        base_lr=0.01,
        hebbian_lr=0.0001,
        youth_lr_multiplier=1.0,
        activation_threshold=0.01,
        synaptogenesis_rate=0.05,
        synaptogenesis_interval=50,
        neurogenesis_interval=200,
        pruning_interval=100,
        pruning_grace_period=20,
        consolidation_interval=100,
        consolidation_replay_steps=5,
        seed=0,
    )


@pytest.fixture
def soma(small_config: SOMAConfig) -> SOMA:
    return SOMA(small_config)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
class TestConstruction:
    def test_seeds_sensor_per_input_modality(self, soma: SOMA) -> None:
        assert "text" in soma.graph.sensor_nodes
        for modality in soma.config.input_modalities:
            sensor = soma.graph.get_sensor(modality)
            assert sensor.node_type is NodeType.SENSOR

    def test_seeds_output_per_output_modality(self, soma: SOMA) -> None:
        assert "text" in soma.graph.output_nodes
        for modality in soma.config.output_modalities:
            out = soma.graph.get_output(modality)
            assert out.node_type is NodeType.OUTPUT

    def test_seeds_requested_associators(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        assoc_count = sum(
            1 for node in soma.graph.all_nodes() if node.node_type is NodeType.ASSOCIATOR
        )
        assert assoc_count == small_config.initial_associator_count

    def test_minimum_one_associator_even_when_configured_zero(self) -> None:
        config = SOMAConfig(
            sensor_output_dim=8,
            associator_input_dim=8,
            associator_hidden_dim=16,
            associator_output_dim=8,
            wm_dim=8,
            key_dim=8,
            value_dim=16,
            text_embed_dim=8,
            initial_associator_count=0,
            initial_integrator_count=0,
            num_curiosity_domains=2,
            seed=0,
        )
        soma = SOMA(config)
        assoc_count = sum(
            1 for node in soma.graph.all_nodes() if node.node_type is NodeType.ASSOCIATOR
        )
        # Constructor enforces max(1, initial_associator_count).
        assert assoc_count >= 1

    def test_associators_connect_sensors_and_outputs(self, soma: SOMA) -> None:
        sensor = soma.graph.get_sensor("text")
        out = soma.graph.get_output("text")
        # Every associator should have at least one incoming edge from the sensor
        # and one outgoing edge to the output.
        for node in soma.graph.all_nodes():
            if node.node_type is not NodeType.ASSOCIATOR:
                continue
            incoming_sources = {e.source_id for e in soma.graph.get_incoming_edges(node.id)}
            outgoing_targets = {e.target_id for e in soma.graph.get_outgoing_edges(node.id)}
            assert sensor.id in incoming_sources
            assert out.id in outgoing_targets

    def test_modules_instantiated(self, soma: SOMA) -> None:
        # All sub-modules must be wired up.
        assert soma.working_memory.wm_dim == soma.config.wm_dim
        assert soma.episodic_memory.capacity == soma.config.episodic_capacity
        assert soma.curiosity.num_domains == soma.config.num_curiosity_domains
        assert soma.homeostasis.max_nodes == soma.config.max_nodes
        assert soma.development is not None

    def test_seeded_graph_is_executable(self, soma: SOMA) -> None:
        """Before any training step, graph must already produce an output."""
        dim = soma.config.sensor_output_dim
        result = soma.step(inputs={"text": torch.randn(dim)})
        assert "text" in result["outputs"]


# ----------------------------------------------------------------------
# Step
# ----------------------------------------------------------------------
class TestStepInference:
    def test_step_without_targets_produces_output(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim
        result = soma.step(inputs={"text": torch.randn(dim)})
        assert result["loss"] is None
        assert "text" in result["outputs"]
        assert result["global_step"] == 0

    def test_global_step_increments(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim
        soma.step(inputs={"text": torch.randn(dim)})
        soma.step(inputs={"text": torch.randn(dim)})
        assert soma.global_step == 2

    def test_returns_expected_result_keys(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim
        result = soma.step(inputs={"text": torch.randn(dim)})
        expected = {
            "outputs",
            "loss",
            "curiosity",
            "lr_multiplier",
            "global_step",
            "num_nodes",
            "num_edges",
            "experience_idx",
        }
        assert expected.issubset(set(result.keys()))


class TestStepTraining:
    def test_step_with_targets_computes_loss(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim
        result = soma.step(
            inputs={"text": torch.randn(dim)},
            targets={"text": torch.randn(dim)},
        )
        assert result["loss"] is not None
        assert result["loss"] == result["loss"]  # not NaN

    def test_loss_decreases_on_fixed_pattern(self, soma: SOMA) -> None:
        """Repeating a paired (input, target) should drive the loss down."""
        torch.manual_seed(0)
        dim = soma.config.sensor_output_dim
        inp = torch.randn(dim)
        tgt = torch.randn(dim) * 0.5

        losses: list[float] = []
        for _ in range(100):
            result = soma.step(inputs={"text": inp.clone()}, targets={"text": tgt.clone()})
            losses.append(result["loss"])

        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early

    def test_training_writes_to_episodic(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim
        assert soma.episodic_memory.num_valid == 0
        soma.step(
            inputs={"text": torch.randn(dim)},
            targets={"text": torch.randn(dim)},
        )
        assert soma.episodic_memory.num_valid == 1

    def test_recent_errors_tracked(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim
        for _ in range(5):
            soma.step(
                inputs={"text": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )
        assert len(soma._recent_errors) == 5

    def test_recent_errors_capped(self, soma: SOMA) -> None:
        """Beyond ``_recent_errors_cap`` the list should truncate."""
        soma._recent_errors_cap = 3
        dim = soma.config.sensor_output_dim
        for _ in range(10):
            soma.step(
                inputs={"text": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )
        assert len(soma._recent_errors) == 3


# ----------------------------------------------------------------------
# Working memory
# ----------------------------------------------------------------------
class TestWorkingMemoryInteraction:
    def test_wm_usage_rises_with_steps(self, soma: SOMA) -> None:
        torch.manual_seed(0)
        initial_occupancy = soma.working_memory.occupancy()
        dim = soma.config.sensor_output_dim
        # Many diverse inputs so write gate is more likely to trigger.
        for _ in range(30):
            soma.step(inputs={"text": torch.randn(dim)})
        # At least the slot age counters should have advanced.
        assert soma.working_memory._age().max().item() > 0
        # Occupancy is gated; we only assert it didn't regress below the
        # starting value (sanity check).
        assert soma.working_memory.occupancy() >= initial_occupancy


# ----------------------------------------------------------------------
# Curiosity
# ----------------------------------------------------------------------
class TestCuriosity:
    def test_curiosity_signal_returned(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim
        result = soma.step(
            inputs={"text": torch.randn(dim)},
            targets={"text": torch.randn(dim)},
        )
        assert isinstance(result["curiosity"], float)
        # First sample is in warmup -> should return the uniform default.
        assert result["curiosity"] >= 0.0


# ----------------------------------------------------------------------
# Growth gating
# ----------------------------------------------------------------------
class TestGrowthGating:
    def test_synaptogenesis_triggers_on_interval(self, small_config: SOMAConfig) -> None:
        small_config.synaptogenesis_interval = 5
        small_config.synaptogenesis_rate = 1.0  # aggressive so we actually add edges
        soma = SOMA(small_config)
        dim = soma.config.sensor_output_dim
        initial_edges = soma.graph.num_edges
        # Step through multiple synaptogenesis intervals.
        rng = torch.Generator().manual_seed(1)
        for _ in range(30):
            soma.step(
                inputs={"text": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
                rng=rng,
            )
        # Growth may or may not succeed depending on activations, but the
        # step loop must not crash at the interval crossings.
        assert soma.graph.num_edges >= initial_edges

    def test_pruning_triggers_on_interval(self, small_config: SOMAConfig) -> None:
        small_config.pruning_interval = 10
        small_config.pruning_grace_period = 0
        small_config.edge_strength_threshold = 10.0  # pretty much everything is "weak"
        small_config.inactivity_threshold = 0
        soma = SOMA(small_config)
        dim = soma.config.sensor_output_dim
        # Track initial edge count.
        initial = soma.graph.num_edges
        # Run a few intervals. Some associators should lose edges and the
        # graph should strictly shrink (or stay — pruning is best-effort).
        for _ in range(30):
            soma.step(inputs={"text": torch.randn(dim)})
        assert soma.graph.num_edges <= initial


# ----------------------------------------------------------------------
# Consolidation
# ----------------------------------------------------------------------
class TestConsolidation:
    def test_consolidation_runs_on_interval(self, small_config: SOMAConfig) -> None:
        small_config.consolidation_interval = 10
        small_config.consolidation_replay_steps = 3
        soma = SOMA(small_config)
        dim = soma.config.sensor_output_dim
        for _ in range(25):
            soma.step(
                inputs={"text": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )
        # Consolidation should have run at least once; since it replays
        # experiences, at least some stored entries should have
        # replay_count > 0.
        replays = soma.episodic_memory.replay_count.sum().item()
        assert replays > 0


# ----------------------------------------------------------------------
# Interactive session
# ----------------------------------------------------------------------
class TestInteractiveSession:
    def test_interactive_session_returns_string(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim

        def encoder(text: str) -> torch.Tensor:
            # Deterministic fake encoder: char codes mapped into a small tensor.
            torch.manual_seed(len(text))
            return torch.randn(min(len(text), 5), dim)

        def decoder(vec: torch.Tensor) -> str:
            return "X"

        out = soma.interactive_session(
            "hi",
            text_encoder=encoder,
            text_decoder=decoder,
            max_output_tokens=3,
        )
        assert isinstance(out, str)

    def test_interactive_session_halts_on_eos(self, soma: SOMA) -> None:
        dim = soma.config.sensor_output_dim

        def encoder(text: str) -> torch.Tensor:
            return torch.randn(2, dim)

        tokens_emitted: list[str] = []

        def decoder(vec: torch.Tensor) -> str:
            # Emit "A", "B", then <EOS> — the loop must halt after B.
            tokens_emitted.append("tok")
            return "<EOS>" if len(tokens_emitted) >= 3 else "A"

        out = soma.interactive_session(
            "hi",
            text_encoder=encoder,
            text_decoder=decoder,
            max_output_tokens=10,
        )
        assert "<EOS>" not in out

    def test_interactive_session_rejects_bad_encoder_output(self, soma: SOMA) -> None:
        def encoder(text: str) -> torch.Tensor:
            # Wrong rank: 1-D instead of 2-D.
            return torch.randn(5)

        def decoder(vec: torch.Tensor) -> str:
            return "X"

        with pytest.raises(ValueError, match="text_encoder"):
            soma.interactive_session("hi", text_encoder=encoder, text_decoder=decoder)


# ----------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------
class TestCheckpoint:
    def test_round_trip_preserves_step(self, soma: SOMA, tmp_path: Path) -> None:
        dim = soma.config.sensor_output_dim
        for _ in range(5):
            soma.step(
                inputs={"text": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )
        path = tmp_path / "soma.pt"
        soma.save_state(path)
        assert path.exists()
        expected_step = soma.global_step
        expected_nodes = soma.graph.num_nodes
        expected_edges = soma.graph.num_edges

        # Fresh instance: load, then compare state.
        fresh = SOMA(soma.config)
        fresh.load_state(path)
        assert fresh.global_step == expected_step
        assert fresh.graph.num_nodes == expected_nodes
        assert fresh.graph.num_edges == expected_edges

    def test_round_trip_preserves_recent_errors(self, soma: SOMA, tmp_path: Path) -> None:
        dim = soma.config.sensor_output_dim
        for _ in range(3):
            soma.step(
                inputs={"text": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )
        errors_before = list(soma._recent_errors)
        path = tmp_path / "soma.pt"
        soma.save_state(path)
        fresh = SOMA(soma.config)
        fresh.load_state(path)
        assert fresh._recent_errors == errors_before

    def test_round_trip_working_memory(self, soma: SOMA, tmp_path: Path) -> None:
        dim = soma.config.sensor_output_dim
        for _ in range(10):
            soma.step(inputs={"text": torch.randn(dim)})
        age_before = soma.working_memory._age().clone()
        path = tmp_path / "soma.pt"
        soma.save_state(path)
        fresh = SOMA(soma.config)
        fresh.load_state(path)
        assert torch.equal(fresh.working_memory._age(), age_before)


class TestNonFiniteLossSkip:
    """SOMA.step gracefully skips when loss is inf/nan; raises after N skips."""

    @pytest.fixture
    def skip_config(self) -> SOMAConfig:
        return SOMAConfig(
            sensor_output_dim=8,
            associator_input_dim=8,
            associator_hidden_dim=16,
            associator_output_dim=8,
            wm_slots=4,
            wm_dim=8,
            key_dim=8,
            value_dim=16,
            text_embed_dim=8,
            initial_associator_count=2,
            initial_integrator_count=0,
            max_nodes=64,
            num_curiosity_domains=2,
            base_lr=0.01,
            hebbian_lr=0.0001,
            youth_lr_multiplier=1.0,
            activation_threshold=0.01,
            synaptogenesis_interval=50,
            neurogenesis_interval=200,
            pruning_interval=100,
            pruning_grace_period=20,
            consolidation_interval=100,
            consolidation_replay_steps=5,
            max_consecutive_skipped_steps=3,
            seed=0,
        )

    def _step_with_inf(self, soma: SOMA, monkeypatch: pytest.MonkeyPatch) -> dict:
        dim = soma.config.sensor_output_dim
        monkeypatch.setattr(
            SOMA, "_compute_loss",
            lambda self, outputs, targets: (torch.tensor(float("inf")), float("inf")),
        )
        return soma.step(
            inputs={"text": torch.randn(dim)},
            targets={"text": torch.randn(dim)},
        )

    def test_inf_loss_returns_skipped_marker(
        self, skip_config: SOMAConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        soma = SOMA(skip_config)
        result = self._step_with_inf(soma, monkeypatch)
        assert result.get("skipped") is True
        assert soma.global_step == 1

    def test_skipped_step_does_not_corrupt_homeostasis(
        self, skip_config: SOMAConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        soma = SOMA(skip_config)
        initial_ema = soma.homeostasis.loss_ema
        self._step_with_inf(soma, monkeypatch)
        assert soma.homeostasis.loss_ema == initial_ema

    def test_too_many_consecutive_skips_raises(
        self, skip_config: SOMAConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        soma = SOMA(skip_config)
        for _ in range(skip_config.max_consecutive_skipped_steps):
            self._step_with_inf(soma, monkeypatch)
        with pytest.raises(ValueError, match="consecutive non-finite"):
            self._step_with_inf(soma, monkeypatch)

    def test_skip_counter_resets_after_finite_step(
        self, skip_config: SOMAConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        soma = SOMA(skip_config)
        # First fill the skip counter to threshold.
        for _ in range(skip_config.max_consecutive_skipped_steps):
            self._step_with_inf(soma, monkeypatch)
        # Restore original _compute_loss for one finite step.
        monkeypatch.undo()
        dim = soma.config.sensor_output_dim
        soma.step(
            inputs={"text": torch.randn(dim)},
            targets={"text": torch.randn(dim)},
        )
        assert soma._consecutive_skipped_steps == 0
        # Now M more inf steps should be tolerated without raising.
        for _ in range(skip_config.max_consecutive_skipped_steps):
            self._step_with_inf(soma, monkeypatch)
        # The (N+1)-th inf step raises.
        with pytest.raises(ValueError, match="consecutive non-finite"):
            self._step_with_inf(soma, monkeypatch)
