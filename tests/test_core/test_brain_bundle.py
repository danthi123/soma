import json
from pathlib import Path

import pytest
import torch

from soma.core.brain_bundle import (
    SCHEMA_VERSION,
    MigrationError,
    migrate_payload,
    unwrap_payload,
    wrap_payload,
)
from soma.core.config import SOMAConfig
from soma.system import SOMA


def test_wrap_adds_schema_version_and_metadata():
    payload = {"hello": "world"}
    wrapped = wrap_payload(payload, soma_version="0.1.0", git_sha="abc1234")
    assert wrapped["format"] == "soma-brain"
    assert wrapped["schema_version"] == SCHEMA_VERSION
    assert wrapped["soma_version"] == "0.1.0"
    assert wrapped["git_sha"] == "abc1234"
    assert "torch_version" in wrapped
    assert "tokenizers_version" in wrapped
    assert "created_at" in wrapped
    assert wrapped["payload"] == payload


def test_unwrap_returns_payload_and_metadata():
    wrapped = wrap_payload({"x": 1}, soma_version="0.1.0")
    payload, meta = unwrap_payload(wrapped)
    assert payload == {"x": 1}
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["soma_version"] == "0.1.0"


def test_migrate_identity_for_current_version():
    payload = {"x": 1}
    wrapped = wrap_payload(payload, soma_version="0.1.0")
    migrated, final_version = migrate_payload(wrapped["payload"], from_schema=SCHEMA_VERSION)
    assert migrated == payload
    assert final_version == SCHEMA_VERSION


def test_migrate_rejects_unknown_future_version():
    with pytest.raises(MigrationError):
        migrate_payload({}, from_schema=99)


def test_migrate_legacy_schema_zero_wraps_as_payload():
    """A pre-versioned checkpoint (old single-file format) enters the migrator
    with from_schema=0 — identity-wrap, warn, upgrade to v1."""
    legacy = {"global_step": 100, "graph": {}, "config": {}}
    migrated, final_version = migrate_payload(legacy, from_schema=0)
    assert final_version == SCHEMA_VERSION
    # Assert nothing is lost
    for key in legacy:
        assert key in migrated


def test_to_cpu_normalizes_nested_tensor_state():
    from soma.core.brain_bundle import to_cpu_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = {
        "a": torch.tensor([1.0, 2.0], device=device),
        "nested": {"b": torch.tensor([3.0], device=device)},
        "list": [torch.tensor([4.0], device=device)],
        "scalar": 42,
        "string": "hello",
    }
    normalized = to_cpu_state(state)
    assert normalized["a"].device.type == "cpu"
    assert normalized["nested"]["b"].device.type == "cpu"
    assert normalized["list"][0].device.type == "cpu"
    assert normalized["scalar"] == 42
    assert normalized["string"] == "hello"


def _small_cfg() -> SOMAConfig:
    return SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=2,
        wm_dim=8,
        episodic_capacity=4,
        key_dim=8,
        value_dim=8,
        vocab_size=16,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )


def test_save_then_load_round_trip_cpu(tmp_path: Path):
    cfg = _small_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    save_path = tmp_path / "brain.pt"
    soma.save_state(str(save_path))

    # New instance, same config, fresh weights -> load should overwrite.
    soma2 = SOMA(cfg, device=torch.device("cpu"))
    soma2.load_state(str(save_path))
    # Pick one nn.Parameter and compare.
    orig = next(iter(soma.graph.state_dict().values()))
    loaded = next(iter(soma2.graph.state_dict().values()))
    assert torch.allclose(orig, loaded)


def test_saved_file_is_wrapped_bundle(tmp_path: Path):
    cfg = _small_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    save_path = tmp_path / "brain.pt"
    soma.save_state(str(save_path))
    raw = torch.load(str(save_path), map_location="cpu", weights_only=False)
    assert raw["format"] == "soma-brain"
    assert raw["schema_version"] == 1
    assert "payload" in raw
    assert "global_step" in raw["payload"]


def test_save_bundle_then_load_bundle(tmp_path: Path):
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

    cfg = SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=2,
        wm_dim=8,
        episodic_capacity=4,
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
    soma = SOMA(cfg, device=torch.device("cpu"))
    tok = train_bpe_tokenizer(["hello world"], vocab_size=32)
    enc = TextEncoder(tok, embed_dim=8, max_seq_len=8, device=torch.device("cpu"))

    out_dir = tmp_path / "brain-bundle"
    soma.save_bundle(str(out_dir), tokenizer=tok, encoder=enc)

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "brain.pt").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "encoder.pt").exists()

    soma2 = SOMA(cfg, device=torch.device("cpu"))
    tok2, enc2 = soma2.load_bundle(str(out_dir))
    # vocab size stays the same across round-trip
    assert tok2.get_vocab_size() == tok.get_vocab_size()


def test_interface_spec_in_manifest(tmp_path: Path):
    cfg = SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=2,
        wm_dim=8,
        episodic_capacity=4,
        key_dim=8,
        value_dim=8,
        vocab_size=16,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    out_dir = tmp_path / "bundle"
    soma.save_bundle(str(out_dir))
    manifest = json.loads((out_dir / "manifest.json").read_text())
    spec = manifest["interface_spec"]
    assert spec["sensor_by_modality"]["text"]  # UUID string
    assert spec["output_by_modality"]["text"]
    assert spec["output_dim"] == cfg.integrator_output_dim
    assert spec["sensor_output_dim"] == cfg.sensor_output_dim
    assert spec["text_embed_dim"] == cfg.text_embed_dim


def test_load_state_migrates_legacy_format_end_to_end(tmp_path: Path):
    """Protects the production migration path.

    ``SOMA.save_state`` has wrapped payloads in the envelope since Task 4,
    but any checkpoint written before the envelope landed (or hand-patched
    legacy file) must still be loadable by a current SOMA. This is the
    end-to-end variant of ``test_migrate_legacy_schema_zero_wraps_as_payload``
    in Task 2 — it exercises the full ``SOMA.load_state`` path (not just
    the ``migrate_payload`` helper) so the legacy checkpoint we actually
    ran through ``scripts/migrate_legacy_checkpoint.py`` is covered in CI.
    """
    cfg = _small_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    save_path = tmp_path / "legacy.pt"
    # Write a v1 bundle, then strip the envelope to simulate a pre-envelope
    # legacy single-file checkpoint (as the live ~900K step brain was
    # before the one-shot migrator ran at Phase 1 merge time).
    soma.save_state(str(save_path))
    raw = torch.load(str(save_path), map_location="cpu", weights_only=False)
    assert raw["format"] == "soma-brain"
    legacy_state = raw["payload"]  # bare payload dict, no envelope
    torch.save(legacy_state, str(save_path))

    # Fresh SOMA, load the legacy-format file: migrator should fire, warn,
    # and the loaded state must match.
    soma2 = SOMA(cfg, device=torch.device("cpu"))
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        soma2.load_state(str(save_path))
    assert any("pre-v1" in str(w.message) for w in caught), (
        "expected v0→v1 UserWarning from migrator"
    )
    assert soma2.global_step == soma.global_step
    orig = next(iter(soma.graph.state_dict().values()))
    loaded = next(iter(soma2.graph.state_dict().values()))
    assert torch.allclose(orig, loaded)
