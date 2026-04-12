"""Tests for ``scripts.interactive`` — REPL for a trained SOMA model."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import interactive  # noqa: E402

from soma.core.config import SOMAConfig  # noqa: E402
from soma.io.text_encoder import train_bpe_tokenizer  # noqa: E402
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
        vocab_size=64,
        initial_associator_count=2,
        initial_integrator_count=0,
        max_nodes=32,
        num_curiosity_domains=2,
        base_lr=0.01,
        hebbian_lr=0.0001,
        youth_lr_multiplier=1.0,
        activation_threshold=0.01,
        synaptogenesis_interval=50,
        neurogenesis_interval=200,
        pruning_interval=100,
        consolidation_interval=100,
        consolidation_replay_steps=5,
        max_input_tokens=16,
        max_output_tokens=4,
        seed=0,
    )


@pytest.fixture
def tokenizer_file(tmp_path: Path) -> Path:
    tokenizer = train_bpe_tokenizer(
        [
            "hello world apples bananas trees fish",
            "cats dogs birds flowers clouds sky",
            "red blue green yellow purple orange",
        ]
        * 3,
        vocab_size=64,
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return path


@pytest.fixture
def corpus_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.txt"
    path.write_text(
        "\n".join(
            [
                "hello world apples bananas trees fish",
                "cats dogs birds flowers clouds sky",
                "red blue green yellow purple orange",
            ]
            * 3
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def checkpoint(small_config: SOMAConfig, tmp_path: Path) -> Path:
    """A tiny trained SOMA checkpoint."""
    soma = SOMA(small_config)
    for _ in range(3):
        soma.step(
            inputs={"text": torch.randn(small_config.sensor_output_dim)},
            targets={"text": torch.randn(small_config.sensor_output_dim)},
        )
    path = tmp_path / "soma.pt"
    soma.save_state(path)
    return path


# ----------------------------------------------------------------------
# Bootstrapping
# ----------------------------------------------------------------------
class TestLoadSoma:
    def test_loads_checkpoint(self, checkpoint: Path) -> None:
        soma = interactive.load_soma(checkpoint)
        assert soma.graph.num_nodes > 0

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            interactive.load_soma(tmp_path / "missing.pt")


class TestPrepareEncoders:
    def test_loads_from_tokenizer_file(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        encoder, decoder = interactive.prepare_encoders(
            small_config, tokenizer_path=tokenizer_file, corpus_path=None
        )
        assert encoder.embed_dim == small_config.text_embed_dim
        assert decoder.embed_dim == small_config.text_embed_dim

    def test_trains_from_corpus_when_tokenizer_missing(
        self, small_config: SOMAConfig, corpus_file: Path
    ) -> None:
        encoder, decoder = interactive.prepare_encoders(
            small_config, tokenizer_path=None, corpus_path=corpus_file
        )
        assert encoder.vocab_size > 0
        assert decoder.vocab_size == encoder.vocab_size

    def test_requires_either_source(self, small_config: SOMAConfig) -> None:
        with pytest.raises(ValueError, match="tokenizer"):
            interactive.prepare_encoders(small_config, tokenizer_path=None, corpus_path=None)

    def test_rejects_empty_corpus(self, small_config: SOMAConfig, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("\n\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            interactive.prepare_encoders(small_config, tokenizer_path=None, corpus_path=empty)

    def test_missing_corpus_raises(self, small_config: SOMAConfig, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            interactive.prepare_encoders(
                small_config, tokenizer_path=None, corpus_path=tmp_path / "absent.txt"
            )


# ----------------------------------------------------------------------
# Command handling
# ----------------------------------------------------------------------
class TestHandleCommand:
    def test_quit_stops(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        msg, cont = interactive.handle_command(":quit", soma)
        assert cont is False
        assert "Goodbye" in msg

    def test_exit_stops(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        _, cont = interactive.handle_command(":exit", soma)
        assert cont is False

    def test_help_lists_commands(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        msg, cont = interactive.handle_command(":help", soma)
        assert cont is True
        assert ":quit" in msg
        assert ":save" in msg
        assert ":stats" in msg

    def test_stats_reports_sizes(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        msg, cont = interactive.handle_command(":stats", soma)
        assert cont is True
        assert f"nodes={soma.graph.num_nodes}" in msg

    def test_reset_memory_clears_wm(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        # Force some usage.
        soma.working_memory.write(
            torch.randn(small_config.wm_dim), torch.randn(small_config.wm_dim)
        )
        interactive.handle_command(":reset_memory", soma)
        assert float(soma.working_memory._usage().max().item()) == 0.0

    def test_save_requires_path(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        msg, cont = interactive.handle_command(":save", soma)
        assert cont is True
        assert "Usage" in msg

    def test_save_writes_file(self, small_config: SOMAConfig, tmp_path: Path) -> None:
        soma = SOMA(small_config)
        dest = tmp_path / "out.pt"
        interactive.handle_command(f":save {dest}", soma)
        assert dest.exists()

    def test_unknown_command(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        msg, cont = interactive.handle_command(":bogus", soma)
        assert cont is True
        assert "Unknown" in msg


# ----------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------
class TestUntrainedWarning:
    def test_zero_step_soma_yields_warning(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        msg = interactive.untrained_warning(soma)
        assert msg is not None
        assert "global_step=0" in msg

    def test_threshold_is_configurable(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        for _ in range(3):
            soma.step(
                inputs={"text": torch.randn(small_config.sensor_output_dim)},
                targets={"text": torch.randn(small_config.sensor_output_dim)},
            )
        # At threshold=2 and 3 real steps, we're past it.
        assert interactive.untrained_warning(soma, threshold=2) is None
        # But at threshold=10 we're still under.
        assert interactive.untrained_warning(soma, threshold=10) is not None

    def test_run_session_emits_warning_for_fresh_soma(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        soma = SOMA(small_config)
        encoder, decoder = interactive.prepare_encoders(
            small_config, tokenizer_path=tokenizer_file, corpus_path=None
        )
        out = io.StringIO()
        interactive.run_session(
            soma,
            encoder,
            decoder,
            input_lines=[":quit"],
            output_stream=out,
            banner=None,
        )
        assert "WARNING" in out.getvalue()

    def test_run_session_skips_warning_for_trained_soma(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        soma = SOMA(small_config)
        # Push past the default threshold.
        dim = small_config.sensor_output_dim
        for _ in range(interactive.UNTRAINED_STEP_THRESHOLD + 5):
            soma.step(inputs={"text": torch.randn(dim)}, targets={"text": torch.randn(dim)})
        encoder, decoder = interactive.prepare_encoders(
            small_config, tokenizer_path=tokenizer_file, corpus_path=None
        )
        out = io.StringIO()
        interactive.run_session(
            soma,
            encoder,
            decoder,
            input_lines=[":quit"],
            output_stream=out,
            banner=None,
        )
        assert "WARNING" not in out.getvalue()


class TestRunSession:
    def _make_encoder_decoder(self, small_config: SOMAConfig, tokenizer_file: Path) -> tuple:
        return interactive.prepare_encoders(
            small_config, tokenizer_path=tokenizer_file, corpus_path=None
        )

    def test_session_handles_quit_command(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        soma = SOMA(small_config)
        encoder, decoder = self._make_encoder_decoder(small_config, tokenizer_file)
        out = io.StringIO()
        interactive.run_session(
            soma,
            encoder,
            decoder,
            input_lines=[":quit"],
            output_stream=out,
            banner=None,
        )
        assert "Goodbye" in out.getvalue()

    def test_session_processes_text_input(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        soma = SOMA(small_config)
        encoder, decoder = self._make_encoder_decoder(small_config, tokenizer_file)
        out = io.StringIO()
        interactive.run_session(
            soma,
            encoder,
            decoder,
            input_lines=["hello", ":quit"],
            output_stream=out,
            max_output_tokens=2,
            banner="BANNER",
        )
        text = out.getvalue()
        assert "BANNER" in text
        assert "Goodbye" in text

    def test_session_stats_command_reports_state(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        soma = SOMA(small_config)
        encoder, decoder = self._make_encoder_decoder(small_config, tokenizer_file)
        out = io.StringIO()
        interactive.run_session(
            soma,
            encoder,
            decoder,
            input_lines=[":stats", ":quit"],
            output_stream=out,
            banner=None,
        )
        text = out.getvalue()
        assert "nodes=" in text
        assert "edges=" in text

    def test_session_skips_empty_lines(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        soma = SOMA(small_config)
        encoder, decoder = self._make_encoder_decoder(small_config, tokenizer_file)
        out = io.StringIO()
        # Empty-line ("") should not trigger command parsing or generation.
        interactive.run_session(
            soma,
            encoder,
            decoder,
            input_lines=["", "   ", ":quit"],
            output_stream=out,
            banner=None,
        )
        assert "Goodbye" in out.getvalue()


# ----------------------------------------------------------------------
# build_callable_pair helper
# ----------------------------------------------------------------------
class TestBuildCallablePair:
    def test_encoder_fallback_for_empty_string(
        self, small_config: SOMAConfig, tokenizer_file: Path
    ) -> None:
        encoder, decoder = interactive.prepare_encoders(
            small_config, tokenizer_path=tokenizer_file, corpus_path=None
        )
        enc_fn, dec_fn = interactive.build_callable_pair(encoder, decoder)
        # Empty string -> encoder returns at least one token's worth of zeros.
        result = enc_fn("")
        assert result.ndim == 2
        assert result.shape[1] == encoder.embed_dim
        # Decoder must accept a single activation vector.
        text = dec_fn(torch.zeros(encoder.embed_dim))
        assert isinstance(text, str)


# ----------------------------------------------------------------------
# CLI main
# ----------------------------------------------------------------------
class TestMainCLI:
    def test_main_missing_checkpoint_returns_1(self, tmp_path: Path) -> None:
        rc = interactive.main(
            [
                "--checkpoint",
                str(tmp_path / "no.pt"),
                "--tokenizer",
                str(tmp_path / "no_tok.json"),
            ]
        )
        assert rc == 1

    def test_main_missing_tokenizer_and_corpus_returns_1(self, checkpoint: Path) -> None:
        rc = interactive.main(["--checkpoint", str(checkpoint)])
        assert rc == 1

    def test_main_runs_with_tokenizer_and_save_on_exit(
        self,
        checkpoint: Path,
        tokenizer_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        saved = tmp_path / "saved.pt"
        monkeypatch.setattr("sys.stdin", io.StringIO(":quit\n"))
        rc = interactive.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer_file),
                "--save-on-exit",
                str(saved),
            ]
        )
        assert rc == 0
        assert saved.exists()
