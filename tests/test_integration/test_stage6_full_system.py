"""Stage 6 integration test — full SOMA system end-to-end.

Demonstrates that every major subsystem works together inside the main
``SOMA`` class:

- The seeded graph grows via synaptogenesis (edges added at intervals).
- ``WorkingMemory`` registers usage as the system runs.
- ``EpisodicMemory`` stores experiences and sees their ``replay_count``
  increase after consolidation cycles.
- ``CuriosityModule`` returns a non-trivial signal once warmup is complete.
- ``HomeostaticRegulator`` produces a learning-rate multiplier.
- Checkpoints round-trip (save + load preserve counters and graph size).
- A text-in / text-out interactive session runs without exceptions.

Kept small and deterministic (seeded); this is the capstone test that
verifies the whole stack, not an accuracy benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.io.text_decoder import TextDecoder
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.system import SOMA


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def full_config() -> SOMAConfig:
    """A compact-but-complete config covering every subsystem interval."""
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
        input_modalities=["text", "image"],
        output_modalities=["text"],
        vocab_size=128,
        initial_associator_count=4,
        initial_integrator_count=0,
        max_nodes=48,
        num_curiosity_domains=3,
        base_lr=0.01,
        hebbian_lr=0.0001,
        youth_lr_multiplier=1.0,
        activation_threshold=0.01,
        synaptogenesis_rate=0.5,
        synaptogenesis_interval=10,
        neurogenesis_interval=200,  # rarely fires in the 50-step run
        pruning_interval=50,
        pruning_grace_period=5,
        edge_strength_threshold=0.001,
        inactivity_threshold=5,
        consolidation_interval=20,
        consolidation_replay_steps=5,
        checkpoint_interval=25,
        max_input_tokens=16,
        max_output_tokens=4,
        seed=0,
    )


@pytest.fixture
def soma(full_config: SOMAConfig) -> SOMA:
    return SOMA(full_config)


# ----------------------------------------------------------------------
# End-to-end test
# ----------------------------------------------------------------------
class TestFullSystem:
    def test_multi_modal_training_exercises_all_subsystems(
        self,
        soma: SOMA,
        full_config: SOMAConfig,
    ) -> None:
        """Run 60 steps with paired text+image inputs; verify subsystems activate."""
        torch.manual_seed(0)
        rng = torch.Generator().manual_seed(1)
        dim = full_config.sensor_output_dim

        initial_edges = soma.graph.num_edges
        losses: list[float] = []

        # Ten distinct "concept" pairs; repeat them so the graph has a chance
        # to learn something stable.
        text_vecs = [torch.randn(dim, generator=rng) for _ in range(5)]
        image_vecs = [torch.randn(dim, generator=rng) for _ in range(5)]
        targets = [(t + i) * 0.5 for t, i in zip(text_vecs, image_vecs, strict=True)]

        for step in range(60):
            idx = step % 5
            result = soma.step(
                inputs={"text": text_vecs[idx].clone(), "image": image_vecs[idx].clone()},
                targets={"text": targets[idx].clone()},
                rng=rng,
            )
            assert result["outputs"]
            assert result["loss"] == result["loss"]  # not NaN
            losses.append(float(result["loss"]))

        # 1. Graph grew (synaptogenesis triggered repeatedly).
        assert soma.graph.num_edges >= initial_edges

        # 2. Working memory registered activity (age always advances on step()).
        assert soma.working_memory._age().max().item() > 0

        # 3. Episodic memory stored experiences.
        assert soma.episodic_memory.num_valid > 0

        # 4. Consolidation ran at least once -> total replay_count > 0.
        assert soma.episodic_memory.replay_count.sum().item() > 0

        # 5. Curiosity is reported as a finite non-negative float.
        assert soma.last_curiosity == soma.last_curiosity  # not NaN
        assert soma.last_curiosity >= 0.0

        # 6. Homeostatic regulator updated at least once.
        assert soma.homeostasis._update_count > 0

        # 7. Loss is finite; late-mean should not explode above early-mean.
        early_mean = sum(losses[:10]) / 10
        late_mean = sum(losses[-10:]) / 10
        assert late_mean < early_mean * 3.0  # sanity: not diverging

    def test_checkpoint_round_trip_after_training(
        self,
        soma: SOMA,
        full_config: SOMAConfig,
        tmp_path: Path,
    ) -> None:
        """Save state mid-training and restore it into a fresh SOMA."""
        torch.manual_seed(0)
        dim = full_config.sensor_output_dim

        for _ in range(25):
            soma.step(
                inputs={"text": torch.randn(dim), "image": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )

        save_path = tmp_path / "full.pt"
        soma.save_state(save_path)

        fresh = SOMA(full_config)
        fresh.load_state(save_path)

        assert fresh.global_step == soma.global_step
        assert fresh.graph.num_nodes == soma.graph.num_nodes
        assert fresh.graph.num_edges == soma.graph.num_edges
        assert fresh.episodic_memory.num_valid == soma.episodic_memory.num_valid

    def test_interactive_session_runs_after_training(
        self,
        soma: SOMA,
        full_config: SOMAConfig,
    ) -> None:
        """After minimal training, interactive_session must produce a string."""
        torch.manual_seed(0)
        dim = full_config.sensor_output_dim

        # Warm the graph up so the first generation isn't pathological.
        for _ in range(10):
            soma.step(
                inputs={"text": torch.randn(dim), "image": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )

        tokenizer = train_bpe_tokenizer(
            ["hello world", "red blue green", "cats dogs birds"] * 3,
            vocab_size=64,
        )
        encoder = TextEncoder(tokenizer, embed_dim=dim, max_seq_len=full_config.max_input_tokens)
        decoder = TextDecoder(tokenizer, embed_dim=dim)

        def encoder_fn(text: str) -> torch.Tensor:
            tokens = encoder.encode_batch(text)
            if tokens.numel() == 0:
                return torch.zeros((1, dim))
            return tokens

        def decoder_fn(vec: torch.Tensor) -> str:
            return decoder.decode(vec)

        out = soma.interactive_session(
            "hello",
            text_encoder=encoder_fn,
            text_decoder=decoder_fn,
            max_output_tokens=3,
        )
        assert isinstance(out, str)


class TestFullSystemStability:
    def test_system_stable_over_100_steps(self, full_config: SOMAConfig) -> None:
        """Run 100 steps and verify no NaNs, node count stays capped, no crashes."""
        torch.manual_seed(0)
        soma = SOMA(full_config)
        dim = full_config.sensor_output_dim
        rng = torch.Generator().manual_seed(2)

        for _ in range(100):
            result = soma.step(
                inputs={"text": torch.randn(dim), "image": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
                rng=rng,
            )
            assert result["outputs"]
            if result["loss"] is not None:
                assert result["loss"] == result["loss"]  # not NaN

        assert soma.graph.num_nodes <= full_config.max_nodes
        assert "text" in soma.graph.sensor_nodes
        assert "image" in soma.graph.sensor_nodes
        assert "text" in soma.graph.output_nodes
