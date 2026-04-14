"""Tests for ``scripts.train`` — CLI training entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# The scripts directory is not in sys.path by default — add it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train  # noqa: E402

from soma.core.config import SOMAConfig  # noqa: E402
from soma.io.dataset_feeders import Sample, TextDatasetFeeder  # noqa: E402
from soma.system import SOMA  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def small_config() -> SOMAConfig:
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
        max_nodes=32,
        num_curiosity_domains=2,
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
        checkpoint_interval=5,
        max_input_tokens=16,
        max_output_tokens=16,
        seed=0,
    )


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    """A tiny corpus that yields enough tokens for a dataset feeder."""
    corpus = "\n".join(
        [
            "the quick brown fox jumps over the lazy dog again and again",
            "red circle blue square green triangle purple hexagon small big",
            "cat dog bird fish cow horse lion tiger shark whale dolphin",
        ]
        * 4
    )
    path = tmp_path / "corpus.txt"
    path.write_text(corpus, encoding="utf-8")
    return path


@pytest.fixture
def config_file(small_config: SOMAConfig, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    small_config.to_yaml(path)
    return path


# ----------------------------------------------------------------------
# Corpus loading
# ----------------------------------------------------------------------
class TestReadCorpus:
    def test_returns_non_empty_lines(self, corpus_file: Path) -> None:
        lines = train.read_corpus(corpus_file)
        assert len(lines) > 0
        for line in lines:
            assert line.strip() == line
            assert line  # no empty entries

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            train.read_corpus(tmp_path / "does_not_exist.txt")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("\n\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no non-empty lines"):
            train.read_corpus(empty)


# ----------------------------------------------------------------------
# Encoder wiring
# ----------------------------------------------------------------------
class TestBuildEncoders:
    def test_returns_encoder_and_decoder(self, small_config: SOMAConfig, corpus_file: Path) -> None:
        corpus = train.read_corpus(corpus_file)
        encoder, decoder = train.build_encoders(small_config, corpus, vocab_size=128)
        assert encoder.embed_dim == small_config.text_embed_dim
        assert decoder.embed_dim == small_config.text_embed_dim


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
class TestIterTokenPairs:
    def test_yields_one_pair_per_target_token(
        self, small_config: SOMAConfig, corpus_file: Path
    ) -> None:
        corpus = train.read_corpus(corpus_file)
        encoder, _ = train.build_encoders(small_config, corpus, vocab_size=64)
        feeder = TextDatasetFeeder(encoder, corpus, chunk_size=4)
        sample = next(iter(feeder))
        pairs = list(train._iter_token_pairs(sample, "text"))
        assert len(pairs) == sample.target.shape[0]

    def test_input_shorter_than_target_cycles_last_token(self, small_config: SOMAConfig) -> None:
        # Build a Sample by hand with short inputs + long target.
        dim = small_config.sensor_output_dim
        sample = Sample(
            inputs={"text": torch.randn(2, dim)},  # only 2 input tokens
            target=torch.randn(5, dim),  # but 5 target tokens
        )
        pairs = list(train._iter_token_pairs(sample, "text"))
        assert len(pairs) == 5
        # Last three pairs should all have the same input (index 1 — the last).
        last_three_inputs = [p[0]["text"] for p in pairs[2:]]
        for inp in last_three_inputs:
            assert torch.equal(inp, sample.inputs["text"][1])

    def test_empty_target_yields_nothing(self) -> None:
        sample = Sample(inputs={"text": torch.zeros(0, 8)}, target=torch.zeros(0, 8))
        assert list(train._iter_token_pairs(sample, "text")) == []


class TestFullSequenceTraining:
    def test_full_sequence_steps_more_than_samples(
        self, small_config: SOMAConfig, corpus_file: Path, tmp_path: Path
    ) -> None:
        """Full-sequence mode should advance global_step per *token*, not per sample."""
        corpus = train.read_corpus(corpus_file)
        encoder, _ = train.build_encoders(small_config, corpus, vocab_size=64)
        feeder = TextDatasetFeeder(encoder, corpus, chunk_size=4)
        soma = SOMA(small_config)
        # 8 total training steps; feeder produces 4-token targets per sample
        # so we should cover at least 2 samples and global_step == 8.
        train.train(
            soma,
            feeder,
            num_steps=8,
            log_every=4,
            checkpoint_dir=tmp_path / "ckpts",
            full_sequence=True,
        )
        assert soma.global_step == 8

    def test_no_full_sequence_steps_once_per_sample(
        self, small_config: SOMAConfig, corpus_file: Path, tmp_path: Path
    ) -> None:
        corpus = train.read_corpus(corpus_file)
        encoder, _ = train.build_encoders(small_config, corpus, vocab_size=64)
        feeder = TextDatasetFeeder(encoder, corpus, chunk_size=4)
        soma = SOMA(small_config)
        train.train(
            soma,
            feeder,
            num_steps=5,
            log_every=5,
            checkpoint_dir=tmp_path / "ckpts",
            full_sequence=False,
        )
        assert soma.global_step == 5

    def test_wm_reset_between_samples_when_requested(
        self, small_config: SOMAConfig, corpus_file: Path, tmp_path: Path
    ) -> None:
        corpus = train.read_corpus(corpus_file)
        encoder, _ = train.build_encoders(small_config, corpus, vocab_size=64)
        feeder = TextDatasetFeeder(encoder, corpus, chunk_size=2)
        soma = SOMA(small_config)
        train.train(
            soma,
            feeder,
            num_steps=6,
            log_every=6,
            checkpoint_dir=tmp_path / "ckpts",
            full_sequence=True,
            reset_wm_between_samples=True,
        )
        # Not a precise test — we just verify training completes and WM is
        # in a valid state.
        assert soma.global_step == 6
        assert float(soma.working_memory._usage().max().item()) >= 0.0


class TestTrainLoop:
    def test_short_run_writes_checkpoints(
        self,
        small_config: SOMAConfig,
        corpus_file: Path,
        tmp_path: Path,
    ) -> None:
        corpus = train.read_corpus(corpus_file)
        encoder, _decoder = train.build_encoders(small_config, corpus, vocab_size=64)
        feeder = TextDatasetFeeder(encoder, corpus, chunk_size=4)
        soma = SOMA(small_config)
        ckpt_dir = tmp_path / "ckpts"
        metrics_file = tmp_path / "metrics.jsonl"

        summary = train.train(
            soma,
            feeder,
            num_steps=10,
            log_every=5,
            checkpoint_dir=ckpt_dir,
            metrics_file=metrics_file,
        )
        assert summary["final_step"] == 10
        # checkpoint_interval=5, so at step 5 and 10 we write.
        written = sorted(ckpt_dir.glob("soma_step_*.pt"))
        assert len(written) >= 1
        assert (ckpt_dir / "soma_final.pt").exists()
        # Metrics file should have at least one record.
        assert metrics_file.exists()
        records = [
            json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines()
        ]
        assert records
        for rec in records:
            assert {"step", "window_mean_loss", "curiosity", "num_nodes", "num_edges"}.issubset(rec)

    def test_final_summary_fields(
        self,
        small_config: SOMAConfig,
        corpus_file: Path,
        tmp_path: Path,
    ) -> None:
        corpus = train.read_corpus(corpus_file)
        encoder, _decoder = train.build_encoders(small_config, corpus, vocab_size=64)
        feeder = TextDatasetFeeder(encoder, corpus, chunk_size=4)
        soma = SOMA(small_config)
        summary = train.train(
            soma,
            feeder,
            num_steps=5,
            log_every=5,
            checkpoint_dir=tmp_path / "ckpts",
        )
        assert {"final_step", "mean_loss", "num_nodes", "num_edges", "checkpoint_path"}.issubset(
            summary
        )
        assert Path(summary["checkpoint_path"]).exists()


# ----------------------------------------------------------------------
# CLI main entry
# ----------------------------------------------------------------------
class TestMainCLI:
    def test_main_runs_end_to_end(
        self,
        config_file: Path,
        corpus_file: Path,
        tmp_path: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        rc = train.main(
            [
                "--config",
                str(config_file),
                "--corpus",
                str(corpus_file),
                "--steps",
                "5",
                "--log-every",
                "5",
                "--checkpoint-dir",
                str(ckpt_dir),
                "--tokenizer-vocab-size",
                "64",
            ]
        )
        assert rc == 0
        assert (ckpt_dir / "soma_final.pt").exists()

    def test_main_missing_resume_file_errors(
        self,
        config_file: Path,
        corpus_file: Path,
        tmp_path: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        rc = train.main(
            [
                "--config",
                str(config_file),
                "--corpus",
                str(corpus_file),
                "--steps",
                "5",
                "--log-every",
                "5",
                "--checkpoint-dir",
                str(ckpt_dir),
                "--tokenizer-vocab-size",
                "64",
                "--resume",
                str(tmp_path / "no_such_file.pt"),
            ]
        )
        assert rc == 1

    def test_main_resume_loads_from_checkpoint(
        self,
        config_file: Path,
        corpus_file: Path,
        tmp_path: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpt"
        # First run: train a bit, produce a checkpoint.
        train.main(
            [
                "--config",
                str(config_file),
                "--corpus",
                str(corpus_file),
                "--steps",
                "5",
                "--log-every",
                "5",
                "--checkpoint-dir",
                str(ckpt_dir),
                "--tokenizer-vocab-size",
                "64",
            ]
        )
        from soma.core.brain_bundle import peek_payload

        first_final = peek_payload(torch.load(str(ckpt_dir / "soma_final.pt"), weights_only=False))
        first_step = int(first_final["global_step"])

        # Resume: global step must advance past the first run.
        train.main(
            [
                "--config",
                str(config_file),
                "--corpus",
                str(corpus_file),
                "--steps",
                "3",
                "--log-every",
                "5",
                "--checkpoint-dir",
                str(ckpt_dir),
                "--tokenizer-vocab-size",
                "64",
                "--resume",
                str(ckpt_dir / "soma_final.pt"),
            ]
        )
        second_final = peek_payload(torch.load(str(ckpt_dir / "soma_final.pt"), weights_only=False))
        second_step = int(second_final["global_step"])
        assert second_step > first_step
