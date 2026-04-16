"""End-to-end smoke test for the FastAPI REST surface.

Drives the full /store → /retrieve → /get → /recent → /forget →
/consolidate → /save round trip via TestClient. Uses a stub embed_fn
injected through ``soma.serve._get_mem`` so the test doesn't depend on
network access for sentence-transformers.

This pins the deployment story from ``docs/positioning.md`` —
"the entire brain is a directory; uvicorn or docker compose serves
it" — at unit-test speed. If the REST schema or routing breaks,
this fails fast.
"""

from __future__ import annotations

import torch
from fastapi.testclient import TestClient

from soma import serve
from soma.memory import MemoryLayer


def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, generator=g)


def _client_with_stub_mem(name: str = "__default__") -> TestClient:
    """Replace serve._mem_cache entry with an in-process MemoryLayer so
    the test is hermetic — no model download, no disk bundle."""
    serve._mem_cache.clear()
    serve._mem_cache[name] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    return TestClient(serve.app)


def test_store_then_retrieve_round_trip() -> None:
    client = _client_with_stub_mem()

    r = client.post("/store", json={"text": "alex lives in portland"})
    assert r.status_code == 200, r.text
    nid_alex = r.json()["node_id"]

    r = client.post("/store", json={"text": "jordan works at acme"})
    assert r.status_code == 200, r.text

    r = client.post("/retrieve", json={"query": "alex lives in portland", "k": 2})
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert any(h["text"] == "alex lives in portland" for h in hits)

    r = client.get(f"/get/{nid_alex}")
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "alex lives in portland"


def test_status_endpoint_reports_size() -> None:
    client = _client_with_stub_mem()
    client.post("/store", json={"text": "fact one"})
    client.post("/store", json={"text": "fact two"})
    r = client.get("/status")
    assert r.status_code == 200, r.text
    assert r.json()["num_entries"] == 2


def test_get_unknown_returns_404() -> None:
    client = _client_with_stub_mem()
    r = client.get("/get/not-a-real-id")
    assert r.status_code == 404


def test_forget_removes_entry() -> None:
    client = _client_with_stub_mem()
    nid = client.post("/store", json={"text": "ephemeral"}).json()["node_id"]
    r = client.post("/forget", json={"node_id": nid})
    assert r.status_code == 200, r.text
    assert r.json() == {"removed": True}

    r = client.get(f"/get/{nid}")
    assert r.status_code == 404


def test_recent_returns_in_reverse_order() -> None:
    client = _client_with_stub_mem()
    for i in range(5):
        client.post("/store", json={"text": f"entry {i}"})
    r = client.get("/recent?n=3")
    assert r.status_code == 200, r.text
    texts = [h["text"] for h in r.json()["hits"]]
    assert texts == ["entry 4", "entry 3", "entry 2"]


def test_consolidate_without_soma_is_safe_noop() -> None:
    client = _client_with_stub_mem()
    client.post("/store", json={"text": "anything"})
    r = client.post("/consolidate")
    assert r.status_code == 200, r.text
    # No SOMA attached → consolidate returns 0
    assert r.json()["processed"] == 0


def test_store_batch_bulk_ingest() -> None:
    client = _client_with_stub_mem()
    r = client.post(
        "/store_batch",
        json={
            "texts": ["one", "two", "three"],
            "metadatas": [{"i": 0}, {"i": 1}, {"i": 2}],
        },
    )
    assert r.status_code == 200, r.text
    ids = r.json()["node_ids"]
    assert len(ids) == 3 and len(set(ids)) == 3
    assert client.get("/status").json()["num_entries"] == 3


def test_related_endpoint_returns_neighbors() -> None:
    client = _client_with_stub_mem()
    a = client.post("/store", json={"text": "cat on mat"}).json()["node_id"]
    client.post("/store", json={"text": "dog on log"})
    client.post("/store", json={"text": "fish in dish"})
    r = client.get(f"/related/{a}?k=2")
    assert r.status_code == 200, r.text
    hits = r.json()["hits"]
    assert len(hits) == 2
    assert all(h["node_id"] != a for h in hits)


def test_related_unknown_returns_404() -> None:
    client = _client_with_stub_mem()
    r = client.get("/related/definitely-not-a-real-id")
    assert r.status_code == 404
