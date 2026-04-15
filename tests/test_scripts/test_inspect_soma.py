"""Tests for scripts/inspect_soma.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch  # noqa: F401 -- kept so pytest fails early if torch is missing

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "inspect_soma.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _tiny_cfg():  # type: ignore[no-untyped-def]
    from soma.core.config import SOMAConfig

    return SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=4,
        wm_dim=8,
        episodic_capacity=8,
        key_dim=8,
        value_dim=8,
        vocab_size=32,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )


def _build_tiny_soma():  # type: ignore[no-untyped-def]
    from soma.system import SOMA

    return SOMA(_tiny_cfg(), device=torch.device("cpu"))


def test_inspect_directory_bundle(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    soma = _build_tiny_soma()
    out_dir = tmp_path / "bundle"
    soma.save_bundle(str(out_dir))

    result = _run(["--bundle", str(out_dir)])
    assert result.returncode == 0, f"stderr={result.stderr}"

    stdout = result.stdout
    # Top-level envelope.
    assert "SOMA Inspection Report" in stdout
    assert "Schema:" in stdout
    assert "Global step:" in stdout

    # Every section header should appear.
    for section in ("Graph", "Working Memory", "Homeostasis", "Episodic Memory"):
        assert section in stdout, f"missing section {section!r}:\n{stdout}"

    # Node-count sanity: 1 sensor + 1 output + 2 associators + 1 integrator.
    assert "Nodes: 5 total" in stdout
    assert "SENSOR      1" in stdout
    assert "OUTPUT      1" in stdout
    assert "ASSOCIATOR  2" in stdout
    assert "INTEGRATOR  1" in stdout


def test_inspect_single_file(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    soma = _build_tiny_soma()
    pt_path = tmp_path / "brain.pt"
    soma.save_state(str(pt_path))

    result = _run(["--bundle", str(pt_path)])
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "SOMA Inspection Report" in result.stdout
    # Single-file mode should not print a sidecar section.
    assert "Textual I/O sidecars" not in result.stdout


def test_inspect_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    result = _run(["--bundle", str(missing)])
    assert result.returncode == 1
    assert "Bundle not found" in result.stderr


def test_inspect_directory_missing_brain(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    result = _run(["--bundle", str(tmp_path / "empty")])
    assert result.returncode == 1
    assert "brain.pt" in result.stderr


def test_inspect_directory_with_tokenizer(tmp_path: Path) -> None:
    """Directory bundle with a tokenizer -> sidecar section appears."""
    pytest.importorskip("torch")
    from soma.io.text_encoder import train_bpe_tokenizer

    soma = _build_tiny_soma()
    tok = train_bpe_tokenizer(["hello world"], vocab_size=32)
    out_dir = tmp_path / "bundle"
    soma.save_bundle(str(out_dir), tokenizer=tok)

    result = _run(["--bundle", str(out_dir)])
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "Textual I/O sidecars" in result.stdout
    assert "Tokenizer: vocab_size=" in result.stdout
    # Encoder wasn't saved -> explicitly marked missing.
    assert "Encoder: (missing)" in result.stdout
