"""Integration tests for the optional /metrics endpoint.

The endpoint is exposed via ``prometheus-fastapi-instrumentator`` which
is an optional extra. When the extra is installed, /metrics returns
Prometheus text format including the SOMA-specific counters/histograms;
when it is not installed, /metrics 404s (FastAPI default) — the rest
of the server keeps working.
"""

from __future__ import annotations

import importlib

import pytest
import torch
from fastapi.testclient import TestClient

_prom_missing = False
try:
    import prometheus_client  # noqa: F401
    import prometheus_fastapi_instrumentator  # noqa: F401
except ImportError:  # pragma: no cover
    _prom_missing = True

pytestmark = pytest.mark.skipif(
    _prom_missing, reason="prometheus-fastapi-instrumentator not installed"
)


def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, generator=g)


def _client_with_stub_mem(name: str = "__default__") -> TestClient:
    from soma import serve
    from soma.memory import MemoryLayer

    serve._mem_cache.clear()
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    mem._bundle_name = name
    serve._mem_cache[name] = mem
    return TestClient(serve.app)


def test_metrics_endpoint_returns_200_without_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force-require an API key on the rest of the routes.
    monkeypatch.setenv("SOMA_API_KEY", "unit-test-key")
    import soma.serve as serve_mod

    importlib.reload(serve_mod)
    client = TestClient(serve_mod.app)

    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    # Clean up so later tests see an open server again.
    monkeypatch.delenv("SOMA_API_KEY", raising=False)
    importlib.reload(serve_mod)


def test_metrics_advertises_soma_counters() -> None:
    client = _client_with_stub_mem()
    # Exercise some hot paths so the SOMA counters exist in the output.
    client.post("/store", json={"text": "hello world"})
    client.post("/retrieve", json={"query": "hello", "k": 1})

    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "soma_store_total" in body
    assert "soma_retrieve_latency_seconds" in body


def test_metrics_endpoint_tagged_system() -> None:
    client = _client_with_stub_mem()
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    # Find the /metrics path in the openapi spec and check its tags.
    path = spec.get("paths", {}).get("/metrics")
    assert path is not None, "/metrics not in OpenAPI spec"
    # The endpoint has exactly one method (GET).
    method_spec = next(iter(path.values()))
    assert "system" in method_spec.get("tags", []), method_spec.get("tags")


def test_metrics_content_type_is_prometheus_text() -> None:
    client = _client_with_stub_mem()
    r = client.get("/metrics")
    assert r.status_code == 200
    ctype = r.headers.get("content-type", "")
    # Prometheus text exposition format. The instrumentator may return
    # either the traditional text format or OpenMetrics; both start
    # with "text/".
    assert ctype.startswith("text/"), ctype
