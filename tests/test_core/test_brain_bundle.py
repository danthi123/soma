from soma.core.brain_bundle import SCHEMA_VERSION, unwrap_payload, wrap_payload


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
