"""POST /auth/revoke — revoke a JWT by jti.

Wraps the existing BlocklistBackend primitive (same one ``soma auth
revoke`` CLI uses) as an HTTP route. Previously the route was
referenced in production prose and docs/auth.md but did not exist —
operators had to shell into the box to run the CLI.
"""
from __future__ import annotations

import importlib
import time
from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_SECRET = "test-secret-at-least-32-bytes-long-aaaa"


@pytest.fixture()
def app_with_blocklist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Boot a clean serve app with a file-backed blocklist."""
    blocklist_path = tmp_path / "blocklist.jsonl"
    # Flush any env inherited from the parent process (matches the
    # pattern used by test_forget_endpoint._fresh_serve).
    for key in (
        "SOMA_API_KEY",
        "SOMA_JWT_ALG",
        "SOMA_JWT_PUBLIC_KEY_PATH",
        "SOMA_JWT_AUDIENCE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(blocklist_path))
    monkeypatch.setenv("SOMA_JWT_SECRET", _SECRET)
    monkeypatch.setenv("SOMA_BUNDLE_PATH", str(tmp_path / "brain"))

    import soma.serve as serve_mod

    reloaded = importlib.reload(serve_mod)
    reloaded._mem_cache.clear()
    return reloaded.app, blocklist_path


def _issue_admin_token(sub: str = "alice") -> tuple[str, str]:
    """Mint a token with admin scope on the default bundle and return
    (token, jti). The ``jti`` is decoded back out of the signed claims
    so tests don't rely on ``issue_token`` accepting a jti parameter —
    it doesn't (auto-generates one per Phase 4).
    """
    import jwt as _jwt

    from soma.auth import issue_token

    token = issue_token(
        sub=sub,
        bundles={"__default__": ["admin"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    claims = _jwt.decode(token, options={"verify_signature": False})
    return token, claims["jti"]


def test_revoke_returns_200_and_blocks_future_use(app_with_blocklist):
    app, blocklist_path = app_with_blocklist
    client = TestClient(app)
    token, jti = _issue_admin_token()

    resp = client.post(
        "/auth/revoke",
        json={"token": token, "reason": "rotated"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["revoked"] == jti
    assert body["reason"] == "rotated"
    assert blocklist_path.exists()
    assert jti in blocklist_path.read_text()


def test_revoke_by_jti(app_with_blocklist):
    app, blocklist_path = app_with_blocklist
    client = TestClient(app)
    admin_token, _ = _issue_admin_token(sub="admin")
    target_jti = "target-jti-0000-0000-0000-000000000001"

    resp = client.post(
        "/auth/revoke",
        json={
            "jti": target_jti,
            "exp": int(time.time()) + 3600,
            "reason": "manual",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["revoked"] == target_jti
    assert target_jti in blocklist_path.read_text()


def test_revoke_requires_auth(app_with_blocklist):
    app, _ = app_with_blocklist
    client = TestClient(app)
    resp = client.post(
        "/auth/revoke",
        json={"jti": "x", "exp": 9999999999},
    )
    assert resp.status_code == 401, resp.text


def test_revoke_rejects_missing_fields(app_with_blocklist):
    app, _ = app_with_blocklist
    client = TestClient(app)
    admin_token, _ = _issue_admin_token(sub="admin")
    # Neither token nor jti+exp — 400
    resp = client.post(
        "/auth/revoke",
        json={"reason": "oops"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 400, resp.text


def test_revoke_returns_503_when_no_blocklist_configured(tmp_path, monkeypatch):
    """If SOMA_JWT_BLOCKLIST_PATH is unset and no Redis is wired,
    ``_blocklist`` is a ``_NullBlocklist`` whose ``add`` is a no-op.
    Returning 200 would mislead callers into thinking the token was
    revoked when it wasn't. Return 503 Service Unavailable instead —
    the feature IS implemented, it's just not configured in this
    deployment. Matches the CLI flow which refuses to run with no
    SOMA_JWT_BLOCKLIST_PATH set."""
    for key in (
        "SOMA_API_KEY",
        "SOMA_JWT_BLOCKLIST_PATH",
        "SOMA_JWT_BLOCKLIST_REDIS_URL",
        "SOMA_JWT_ALG",
        "SOMA_JWT_PUBLIC_KEY_PATH",
        "SOMA_JWT_AUDIENCE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SOMA_JWT_SECRET", _SECRET)
    monkeypatch.setenv("SOMA_BUNDLE_PATH", str(tmp_path / "brain"))

    import soma.serve as serve_mod

    importlib.reload(serve_mod)
    serve_mod._mem_cache.clear()
    client = TestClient(serve_mod.app)

    admin_token, _ = _issue_admin_token(sub="admin")
    resp = client.post(
        "/auth/revoke",
        json={"jti": "some-jti", "exp": int(time.time()) + 3600},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 503, resp.text
    assert "blocklist" in resp.text.lower()
    assert "SOMA_JWT_BLOCKLIST_PATH" in resp.text
