"""Phase 8 Task 2 — ``SOMA_METRICS_PUBLIC=0`` auth gate on ``/metrics``.

Default behaviour (env unset or ``"1"``) keeps ``/metrics`` public so
existing Prometheus scrape configs don't break. Setting ``"0"`` wraps
the endpoint with ``Depends(require_auth(None, "read"))`` so only
bearer-tokened callers can read telemetry in sensitive deploys.

Every test reloads ``soma.serve`` under the env snapshot it needs so
module-level state (the instrumentator's ``.expose()`` binding) reflects
the gate setting.
"""

from __future__ import annotations

import importlib
from datetime import timedelta

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

_SECRET = "metrics-auth-test-secret"


def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, generator=g)


def _fresh_serve(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    """Reimport soma.serve with an explicit env snapshot + empty cache."""
    from soma import serve as _initial_serve
    from soma.memory import MemoryLayer

    for key in (
        "SOMA_API_KEY",
        "SOMA_JWT_SECRET",
        "SOMA_JWT_ALG",
        "SOMA_JWT_PUBLIC_KEY_PATH",
        "SOMA_JWT_LEEWAY",
        "SOMA_JWT_BLOCKLIST_PATH",
        "SOMA_METRICS_PUBLIC",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reloaded = importlib.reload(_initial_serve)
    reloaded._mem_cache.clear()
    reloaded._mem_cache["__default__"] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    return reloaded


# ------------------------------------------------------------------
# Default behaviour — env unset / "1" = public
# ------------------------------------------------------------------
def test_metrics_public_by_default_no_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env set + JWT secret configured => /metrics stays public.

    The key property: adding JWT on other routes must NOT implicitly
    gate /metrics. Operators who scrape Prom without a bearer today
    keep working on upgrade.
    """
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    assert r.text  # some body, not empty


def test_metrics_explicit_public_1_stays_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET, SOMA_METRICS_PUBLIC="1")
    client = TestClient(reloaded.app)
    r = client.get("/metrics")
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------
# Gated — SOMA_METRICS_PUBLIC=0 requires a bearer with read perm
# ------------------------------------------------------------------
def test_metrics_gated_requires_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET, SOMA_METRICS_PUBLIC="0")
    client = TestClient(reloaded.app)
    r = client.get("/metrics")
    assert r.status_code == 401, r.text


def test_metrics_gated_accepts_admin_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soma.auth import issue_token

    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET, SOMA_METRICS_PUBLIC="0")
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="ops",
        bundles={"ops": ["admin"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text
    assert "soma_" in r.text or "python_" in r.text  # Prom text body


def test_metrics_gated_read_perm_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read-scoped token suffices — /metrics is a read-only view.

    The helper route uses ``require_auth(None, "read")`` so any claim
    that meets the read bar anywhere (write or admin implies read)
    passes the check.
    """
    from soma.auth import issue_token

    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET, SOMA_METRICS_PUBLIC="0")
    client = TestClient(reloaded.app)
    token = issue_token(
        sub="prom-scraper",
        bundles={"obs": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )
    r = client.get("/metrics", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200, r.text


def test_metrics_gated_bogus_bearer_401(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET, SOMA_METRICS_PUBLIC="0")
    client = TestClient(reloaded.app)
    r = client.get("/metrics", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


def test_metrics_gated_legacy_api_key_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy SOMA_API_KEY escape hatch still admits /metrics when gated."""
    reloaded = _fresh_serve(monkeypatch, SOMA_API_KEY="legacy-scrape-key", SOMA_METRICS_PUBLIC="0")
    client = TestClient(reloaded.app)
    r = client.get("/metrics", headers={"Authorization": "Bearer legacy-scrape-key"})
    assert r.status_code == 200, r.text


def test_metrics_gated_but_no_auth_configured_stays_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SOMA_METRICS_PUBLIC=0 with no auth envs set is open-mode.

    require_auth's first branch is the _auth_disabled() fast-path — so
    if the operator asked for the gate without wiring auth, we don't
    lock them out of their own server. /metrics still responds 200.
    """
    reloaded = _fresh_serve(monkeypatch, SOMA_METRICS_PUBLIC="0")
    client = TestClient(reloaded.app)
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
