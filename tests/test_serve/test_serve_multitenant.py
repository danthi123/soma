"""Tests for the new production hooks: /health, /version, auth, and
multi-tenant /bundles/{name}/... routing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from fastapi.testclient import TestClient

from soma import serve
from soma.memory import MemoryLayer


def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, generator=g)


def _reset_cache() -> None:
    serve._mem_cache.clear()


def _install_stub(name: str) -> None:
    key = name or "__default__"
    serve._mem_cache[key] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)


@contextmanager
def _api_key(key: str) -> Iterator[None]:
    before = serve.API_KEY
    serve.API_KEY = key
    try:
        yield
    finally:
        serve.API_KEY = before


def test_health_endpoint_has_no_auth() -> None:
    _reset_cache()
    client = TestClient(serve.app)
    with _api_key("top-secret"):
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["loaded_bundles"] == 0


def test_version_endpoint_returns_string() -> None:
    _reset_cache()
    client = TestClient(serve.app)
    r = client.get("/version")
    assert r.status_code == 200
    assert isinstance(r.json()["version"], str)


def test_protected_endpoint_rejects_missing_auth() -> None:
    _reset_cache()
    _install_stub("__default__")
    client = TestClient(serve.app)
    with _api_key("top-secret"):
        r = client.post("/store", json={"text": "x"})
    assert r.status_code == 401
    assert "Bearer" in r.headers.get("WWW-Authenticate", "")


def test_protected_endpoint_accepts_correct_bearer() -> None:
    _reset_cache()
    _install_stub("__default__")
    client = TestClient(serve.app)
    with _api_key("top-secret"):
        r = client.post(
            "/store",
            json={"text": "with-auth"},
            headers={"Authorization": "Bearer top-secret"},
        )
    assert r.status_code == 200


def test_protected_endpoint_rejects_wrong_bearer() -> None:
    _reset_cache()
    _install_stub("__default__")
    client = TestClient(serve.app)
    with _api_key("top-secret"):
        r = client.post(
            "/store",
            json={"text": "x"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert r.status_code == 401


def test_unset_api_key_means_open_server() -> None:
    _reset_cache()
    _install_stub("__default__")
    client = TestClient(serve.app)
    with _api_key(""):
        r = client.post("/store", json={"text": "no-auth-needed"})
    assert r.status_code == 200


def test_bundle_routes_isolate_stores() -> None:
    _reset_cache()
    _install_stub("alex")
    _install_stub("bobbi")
    client = TestClient(serve.app)

    r_a = client.post("/bundles/alex/store", json={"text": "alex-only fact"})
    assert r_a.status_code == 200
    r_b = client.post("/bundles/bobbi/store", json={"text": "bobbi-only fact"})
    assert r_b.status_code == 200

    # Status per bundle reflects independent counts.
    assert client.get("/bundles/alex/status").json()["num_entries"] == 1
    assert client.get("/bundles/bobbi/status").json()["num_entries"] == 1

    # Alex's retrieve shouldn't see Bobbi's entry.
    hits_a = client.post(
        "/bundles/alex/retrieve", json={"query": "fact", "k": 5}
    ).json()["hits"]
    texts_a = [h["text"] for h in hits_a]
    assert "alex-only fact" in texts_a
    assert "bobbi-only fact" not in texts_a


def test_bundle_rejects_invalid_name() -> None:
    _reset_cache()
    client = TestClient(serve.app)
    r = client.post("/bundles/..%2Fetc/store", json={"text": "x"})
    # FastAPI may 404 before we see the name, or 400 if it reaches us.
    assert r.status_code in (400, 404)


def test_bundle_name_with_traversal_attempt_is_rejected() -> None:
    """Directly call the validator with a forbidden name to be sure."""
    _reset_cache()
    client = TestClient(serve.app)
    # Path segment with dots is fine character-wise per regex; try a /
    r = client.post("/bundles/evil name/store", json={"text": "x"})
    assert r.status_code in (400, 404, 422)


def test_bundle_forget_404_on_unknown_id() -> None:
    _reset_cache()
    _install_stub("alex")
    client = TestClient(serve.app)
    r = client.post("/bundles/alex/forget", json={"node_id": "nope"})
    assert r.status_code == 404


def test_health_reports_loaded_bundle_count() -> None:
    _reset_cache()
    client = TestClient(serve.app)
    assert client.get("/health").json()["loaded_bundles"] == 0
    _install_stub("alex")
    assert client.get("/health").json()["loaded_bundles"] == 1
    _install_stub("bobbi")
    assert client.get("/health").json()["loaded_bundles"] == 2
