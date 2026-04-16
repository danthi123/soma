"""Phase 4 Task 3 + 5 — JWT auth dependency + metric integration.

Covers:
- Open mode (neither SOMA_API_KEY nor SOMA_JWT_SECRET set).
- JWT happy path: read perm hits /retrieve, write perm blocked on /store,
  bundle mismatch rejected, admin reaches everything.
- Expired token -> 401 + WWW-Authenticate.
- Legacy SOMA_API_KEY still works, response carries X-SOMA-Deprecated.
- RS256 mode via file-path env vars.
- ``soma_auth_failures_total`` counter fires with distinct reasons.

Every test reloads ``soma.serve`` under explicit env so module-level
state (API_KEY / JWT_SECRET) picks up the right values. A shared
``MemoryLayer`` is injected through ``serve._mem_cache`` so no model
downloads fire.
"""

from __future__ import annotations

import importlib
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from soma import serve as _initial_serve
from soma.auth import issue_token
from soma.memory import MemoryLayer

_SECRET = "jwt-auth-test-secret"


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------
def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, generator=g)


def _fresh_serve(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    """Reimport soma.serve with a specific env-var snapshot.

    Wipes cached bundles so each test is hermetic.
    """
    for key in (
        "SOMA_API_KEY",
        "SOMA_JWT_SECRET",
        "SOMA_JWT_ALG",
        "SOMA_JWT_PUBLIC_KEY_PATH",
        "SOMA_JWT_LEEWAY",
        "SOMA_JWT_BLOCKLIST_PATH",
        "SOMA_JWT_AUDIENCE",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reloaded = importlib.reload(_initial_serve)
    reloaded._mem_cache.clear()
    reloaded._mem_cache["__default__"] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    reloaded._mem_cache["alex"] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    reloaded._mem_cache["bobbi"] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    return reloaded


def _auth_counter_value(reloaded: object, reason: str) -> float:
    """Read soma_auth_failures_total{reason=...} via registry snapshot."""
    from soma import metrics as m

    counter = m.AUTH_FAILURES_TOTAL
    # Noop fallback path: no observability deps -> skip assertion.
    if not m.prometheus_available:
        return 0.0
    sample = counter.labels(reason=reason)
    return float(sample._value.get())  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# Open mode — neither env var set, legacy behaviour
# ------------------------------------------------------------------
def test_no_token_and_no_api_key_set_still_open(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch)
    client = TestClient(reloaded.app)
    r = client.get("/status")
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------
# JWT happy-path / perm enforcement
# ------------------------------------------------------------------
def test_jwt_valid_read_token_hits_retrieve(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    # /retrieve is not bundle-scoped in the URL; default bundle; read
    # perm anywhere satisfies.
    r = client.post(
        "/retrieve",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "hello", "k": 1},
    )
    assert r.status_code == 200, r.text


def test_jwt_valid_read_token_rejected_on_store(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "nope"},
    )
    assert r.status_code == 403
    # Error body mentions the missing perm.
    body = r.json()
    assert "write" in body.get("detail", "").lower()


def test_jwt_bundle_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"alex": ["read", "write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/bundles/bobbi/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "sneaky"},
    )
    assert r.status_code == 403
    body = r.json()
    assert "bobbi" in body.get("detail", "")


def test_jwt_admin_token_reaches_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="ops",
        bundles={"alex": ["admin"], "bobbi": ["admin"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/status", headers=headers).status_code == 200
    assert (
        client.post("/store", json={"text": "admin store"}, headers=headers).status_code == 200
    )
    assert (
        client.post(
            "/bundles/alex/store", json={"text": "per-tenant"}, headers=headers
        ).status_code
        == 200
    )
    assert client.post("/bundles/bobbi/consolidate", headers=headers).status_code == 200


def test_expired_token_401(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=-10),  # past
        secret=_SECRET,
    )
    r = client.post(
        "/retrieve",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "x", "k": 1},
    )
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate", "").lower().startswith("bearer")


def test_malformed_token_401(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    r = client.post(
        "/retrieve",
        headers={"Authorization": "Bearer not.a.real.jwt"},
        json={"query": "x", "k": 1},
    )
    assert r.status_code == 401


# ------------------------------------------------------------------
# Legacy SOMA_API_KEY escape hatch
# ------------------------------------------------------------------
def test_legacy_api_key_still_works_with_deprecation_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_API_KEY="legacy-key")
    client = TestClient(reloaded.app)
    r = client.post(
        "/store",
        headers={"Authorization": "Bearer legacy-key"},
        json={"text": "grandfathered"},
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("X-SOMA-Deprecated") == "use JWT"


def test_legacy_api_key_wrong_value_401(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_API_KEY="legacy-key")
    client = TestClient(reloaded.app)
    r = client.post(
        "/store",
        headers={"Authorization": "Bearer wrong"},
        json={"text": "nope"},
    )
    assert r.status_code == 401


def test_both_env_set_jwt_path_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # Token valid -> pass through JWT path. Legacy still falls through
    # for a raw match.
    reloaded = _fresh_serve(
        monkeypatch, SOMA_JWT_SECRET=_SECRET, SOMA_API_KEY="legacy-key"
    )
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="a",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "jwt wins"},
    )
    assert r.status_code == 200, r.text
    # JWT success => no deprecation header
    assert "X-SOMA-Deprecated" not in r.headers

    # Legacy raw key also still works.
    r2 = client.post(
        "/store",
        headers={"Authorization": "Bearer legacy-key"},
        json={"text": "legacy"},
    )
    assert r2.status_code == 200, r2.text
    assert r2.headers.get("X-SOMA-Deprecated") == "use JWT"


# ------------------------------------------------------------------
# RS256 mode
# ------------------------------------------------------------------
def test_rs256_mode_verifies_with_public_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub_path = tmp_path / "jwt.pub.pem"
    pub_path.write_bytes(pub_pem)

    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_ALG="RS256",
        SOMA_JWT_PUBLIC_KEY_PATH=str(pub_path),
    )
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="rs256-user",
        bundles={"alex": ["read", "write"]},
        expires_in=timedelta(minutes=5),
        alg="RS256",
        private_key_pem=priv_pem,
    )
    r = client.get(
        "/bundles/alex/status",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------
# Metrics — soma_auth_failures_total{reason=...}
# ------------------------------------------------------------------
def test_auth_failure_increments_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    before = _auth_counter_value(reloaded, "invalid_token")
    client = TestClient(reloaded.app)
    r = client.post(
        "/store",
        headers={"Authorization": "Bearer gibberish"},
        json={"text": "x"},
    )
    assert r.status_code == 401
    after = _auth_counter_value(reloaded, "invalid_token")

    from soma import metrics as m

    if m.prometheus_available:
        assert after >= before + 1


def test_perm_failure_distinct_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    before = _auth_counter_value(reloaded, "insufficient_perm")
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "write denied"},
    )
    assert r.status_code == 403
    after = _auth_counter_value(reloaded, "insufficient_perm")

    from soma import metrics as m

    if m.prometheus_available:
        assert after >= before + 1


# ------------------------------------------------------------------
# Revocation — SOMA_JWT_BLOCKLIST_PATH wires the file blocklist
# ------------------------------------------------------------------
def _revoke_via_bl(bl_path: Path, jti: str, exp_ts: int) -> None:
    """Helper — append a revocation to the file backing the server."""
    import time

    from soma.auth_revocation import FileBlocklist, RevocationRecord

    FileBlocklist(bl_path).add(
        RevocationRecord(
            jti=jti,
            revoked_at=int(time.time()),
            reason="test revocation",
            exp=exp_ts,
        )
    )


def test_revoked_token_returns_401_with_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """/store with a revoked JWT => 401 + detail mentions revoked."""
    import time

    bl_path = tmp_path / "bl.jsonl"
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_JWT_BLOCKLIST_PATH=str(bl_path),
    )
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    principal = __import__("soma.auth", fromlist=["verify_token"]).verify_token(
        token, secret=_SECRET
    )
    assert principal.jti is not None
    _revoke_via_bl(bl_path, principal.jti, exp_ts=int(time.time()) + 600)

    # Clear the module-level blocklist cache so the reader re-reads
    # the file. Writing via a *separate* FileBlocklist instance means
    # the serve module's cached instance hasn't seen the new line yet
    # — force it by bumping mtime via direct touch + calling _reload.
    reloaded._blocklist._reload()

    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "should-be-blocked"},
    )
    assert r.status_code == 401, r.text
    body = r.json()
    assert "revoked" in body.get("detail", "").lower()
    assert r.headers.get("WWW-Authenticate", "").lower().startswith("bearer")


def test_revocation_metric_counter_increments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """soma_auth_failures_total{reason='revoked_token'} advances on 401."""
    import time

    bl_path = tmp_path / "bl.jsonl"
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_JWT_BLOCKLIST_PATH=str(bl_path),
    )
    before = _auth_counter_value(reloaded, "revoked_token")

    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    from soma.auth import verify_token as _vt

    principal = _vt(token, secret=_SECRET)
    assert principal.jti is not None
    _revoke_via_bl(bl_path, principal.jti, exp_ts=int(time.time()) + 600)
    reloaded._blocklist._reload()

    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "nope"},
    )
    assert r.status_code == 401, r.text
    after = _auth_counter_value(reloaded, "revoked_token")

    from soma import metrics as m

    if m.prometheus_available:
        assert after >= before + 1


# ------------------------------------------------------------------
# Phase 18 — SOMA_JWT_AUDIENCE plumbing
# ------------------------------------------------------------------
def test_jwt_audience_match_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server SOMA_JWT_AUDIENCE=svc-A + token aud=svc-A => 200."""
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_JWT_AUDIENCE="svc-A",
    )
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
        audience="svc-A",
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "aud match"},
    )
    assert r.status_code == 200, r.text


def test_jwt_audience_mismatch_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server SOMA_JWT_AUDIENCE=svc-A + token aud=svc-B => 401."""
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_JWT_AUDIENCE="svc-A",
    )
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
        audience="svc-B",
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "wrong aud"},
    )
    assert r.status_code == 401, r.text
    assert r.headers.get("WWW-Authenticate", "").lower().startswith("bearer")


def test_jwt_audience_missing_on_token_401(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server expects aud but token has none => 401 (pyjwt MissingRequiredClaimError)."""
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_JWT_AUDIENCE="svc-A",
    )
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
        # audience deliberately omitted
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "no aud"},
    )
    assert r.status_code == 401, r.text


def test_jwt_audience_unset_on_server_ignores_aud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SOMA_JWT_AUDIENCE unset => token with aud still verifies.

    Backward-compat: a multi-service token should keep working on a
    legacy single-service server that hasn't opted into the check.
    """
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
        audience="svc-A",
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "aud ignored"},
    )
    assert r.status_code == 200, r.text


def test_blocklist_path_unset_behaves_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No SOMA_JWT_BLOCKLIST_PATH => pre-revocation behaviour intact.

    A normal happy-path write succeeds; nothing raises. This pins that
    the blocklist plumbing is transparent when unconfigured.
    """
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="alex",
        bundles={"__default__": ["write"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.post(
        "/store",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "through"},
    )
    assert r.status_code == 200, r.text
