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


# ------------------------------------------------------------------
# Phase 19 — POST /snapshot
# ------------------------------------------------------------------
def test_snapshot_writes_bundle_to_path(
    tmp_path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """POST /snapshot with {path} writes a loadable bundle + reports entries."""
    # chdir so the cwd-containment check accepts our tmp_path target.
    monkeypatch.chdir(tmp_path)
    client = _client_with_stub_mem()
    client.post("/store", json={"text": "first"})
    client.post("/store", json={"text": "second"})

    r = client.post("/snapshot", json={"path": "out/bundle"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved"] is True
    assert body["entries"] == 2
    saved_path = tmp_path / "out" / "bundle"
    assert saved_path.exists()
    assert (saved_path / "memory_index.json").exists()
    # Response.path is absolute and points at the real target.
    from pathlib import Path

    assert Path(body["path"]).resolve() == saved_path.resolve()


def test_snapshot_rejects_path_escaping_cwd(
    tmp_path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """A relative path with ``..`` that escapes cwd is a 400, not a write."""
    (tmp_path / "inside").mkdir()
    monkeypatch.chdir(tmp_path / "inside")
    client = _client_with_stub_mem()
    client.post("/store", json={"text": "x"})

    # ../outside escapes the cwd the server was started in.
    r = client.post("/snapshot", json={"path": "../outside"})
    assert r.status_code == 400, r.text
    assert "escapes" in r.json()["detail"].lower() or "cwd" in r.json()["detail"].lower()
    # Nothing was written outside.
    assert not (tmp_path / "outside").exists()


def test_snapshot_requires_write_perm_when_auth_on(
    tmp_path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """With JWT auth configured, a read-only token is 403 on /snapshot.

    Patches the JWT_SECRET module-level global directly (instead of
    reloading the module) so other tests in the file keep seeing the
    open-mode server — no state leak at teardown.
    """
    from datetime import timedelta

    from soma.auth import issue_token

    secret = "snapshot-perm-test-secret"
    monkeypatch.setattr(serve, "JWT_SECRET", secret)
    monkeypatch.setattr(serve, "JWT_ALG", "HS256")
    monkeypatch.setattr(serve, "_JWT_PUBLIC_KEY_PEM", None)
    monkeypatch.setattr(serve, "API_KEY", "")
    monkeypatch.chdir(tmp_path)
    client = _client_with_stub_mem()

    # Read-only token: /snapshot requires write -> 403.
    read_token = issue_token(
        sub="reader",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=secret,
    )
    r = client.post(
        "/snapshot",
        headers={"Authorization": f"Bearer {read_token}"},
        json={"path": "snap1"},
    )
    assert r.status_code == 403, r.text

    # Write token on same bundle: 200.
    write_token = issue_token(
        sub="writer",
        bundles={"__default__": ["read", "write"]},
        expires_in=timedelta(minutes=5),
        secret=secret,
    )
    r = client.post(
        "/snapshot",
        headers={"Authorization": f"Bearer {write_token}"},
        json={"path": "snap2"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["saved"] is True


def test_snapshot_empty_path_is_400(
    tmp_path, monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """Empty-string path is rejected before any filesystem work happens."""
    monkeypatch.chdir(tmp_path)
    client = _client_with_stub_mem()
    r = client.post("/snapshot", json={"path": ""})
    assert r.status_code == 400


def test_rest_store_crash_reload_persists(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end crash-and-reload: REST stores 3 entries, simulated
    crash clears the in-memory cache, next /retrieve reloads from disk
    (snapshot + WAL) and finds the entries.
    """
    # Point the serve module at a fresh bundle dir so the reload path
    # picks up the WAL we just wrote.
    bundle = tmp_path / "crash-bundle"
    original_path = serve.BUNDLE_PATH
    original_embed = serve._embed_fn_cache
    try:
        serve.BUNDLE_PATH = bundle
        serve._embed_fn_cache = _stub_embed
        serve._mem_cache.clear()
        # Pre-seed the cache with a MemoryLayer that has the WAL wired
        # to this bundle — mirrors what a live server would do the first
        # time a request hits this tenant.
        serve._mem_cache["__default__"] = MemoryLayer(
            embed_fn=_stub_embed, embed_dim=32, bundle_path=bundle
        )
        client = TestClient(serve.app)

        ids: list[str] = []
        for text in ["crash fact 1", "crash fact 2", "crash fact 3"]:
            r = client.post("/store", json={"text": text})
            assert r.status_code == 200, r.text
            ids.append(r.json()["node_id"])

        # Simulated crash: drop the in-memory cache. The WAL on disk is
        # still there (durability="sync" is the default).
        cached = serve._mem_cache["__default__"]
        cached.close()
        serve._mem_cache.clear()

        # Next retrieve forces _get_mem() to rehydrate from the bundle
        # on disk. The WAL replay must reinstate all three entries.
        r = client.post("/retrieve", json={"query": "crash fact 2", "k": 3})
        assert r.status_code == 200, r.text
        hits = r.json()["hits"]
        hit_ids = {h["node_id"] for h in hits}
        assert set(ids).issubset(hit_ids), (
            f"expected all 3 ids {ids} after reload, got {hit_ids}"
        )

        # /status should now report 3 entries loaded from disk.
        r = client.get("/status")
        assert r.status_code == 200
        assert r.json()["num_entries"] == 3
    finally:
        for mem in serve._mem_cache.values():
            mem.close()
        serve._mem_cache.clear()
        serve.BUNDLE_PATH = original_path
        serve._embed_fn_cache = original_embed
