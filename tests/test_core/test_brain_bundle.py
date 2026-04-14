import pytest
import torch

from soma.core.brain_bundle import (
    SCHEMA_VERSION,
    MigrationError,
    migrate_payload,
    unwrap_payload,
    wrap_payload,
)


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
