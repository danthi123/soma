"""Phase 26 Task 2 — REST API rate-limit middleware.

Every test reloads ``soma.serve`` under an explicit env snapshot so the
module-level ``_RATE_LIMITER`` picks up the intended config. A shared
stub ``MemoryLayer`` is injected through ``serve._mem_cache`` so no
model downloads fire.
"""

from __future__ import annotations

import importlib
from datetime import timedelta

import pytest
import torch
from fastapi.testclient import TestClient

from soma import serve as _initial_serve
from soma.auth import issue_token
from soma.memory import MemoryLayer

_SECRET = "rate-limit-test-secret"


def _stub_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, generator=g)


def _fresh_serve(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    """Reimport soma.serve with a clean env snapshot + stub bundles."""
    for key in (
        "SOMA_API_KEY",
        "SOMA_JWT_SECRET",
        "SOMA_JWT_ALG",
        "SOMA_JWT_PUBLIC_KEY_PATH",
        "SOMA_JWT_PRIVATE_KEY_PATH",
        "SOMA_JWT_LEEWAY",
        "SOMA_JWT_BLOCKLIST_PATH",
        "SOMA_JWT_AUDIENCE",
        "SOMA_RATE_LIMIT_RPS",
        "SOMA_RATE_LIMIT_BURST",
        "SOMA_RATE_LIMIT_SCOPE",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reloaded = importlib.reload(_initial_serve)
    reloaded._mem_cache.clear()
    reloaded._mem_cache["__default__"] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    reloaded._mem_cache["alex"] = MemoryLayer(embed_fn=_stub_embed, embed_dim=32)
    return reloaded


def _bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _issue(sub: str, perms: list[str] | None = None) -> str:
    perms = perms if perms is not None else ["read", "write"]
    return issue_token(
        sub=sub,
        bundles={"__default__": perms, "alex": perms},
        expires_in=timedelta(minutes=5),
        secret=_SECRET,
    )


# ------------------------------------------------------------------
# Enforcement
# ------------------------------------------------------------------
def test_rate_limit_429_after_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_RATE_LIMIT_RPS="1",
        SOMA_RATE_LIMIT_BURST="2",
    )
    client = TestClient(reloaded.app)
    headers = _bearer_headers(_issue("alex"))
    assert client.get("/status", headers=headers).status_code == 200
    assert client.get("/status", headers=headers).status_code == 200
    r = client.get("/status", headers=headers)
    assert r.status_code == 429, r.text
    assert "Retry-After" in r.headers
    # Retry-After is seconds-as-string; parses as a float > 0.
    assert float(r.headers["Retry-After"]) > 0.0


def test_rate_limit_exempts_health(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_RATE_LIMIT_RPS="1",
        SOMA_RATE_LIMIT_BURST="1",
    )
    client = TestClient(reloaded.app)
    headers = _bearer_headers(_issue("alex"))
    # Burn the budget on /status.
    assert client.get("/status", headers=headers).status_code == 200
    assert client.get("/status", headers=headers).status_code == 429
    # /health is unauthenticated AND exempt — always 200.
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_rate_limit_exempts_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_RATE_LIMIT_RPS="1",
        SOMA_RATE_LIMIT_BURST="1",
    )
    client = TestClient(reloaded.app)
    headers = _bearer_headers(_issue("alex"))
    assert client.get("/status", headers=headers).status_code == 200
    assert client.get("/status", headers=headers).status_code == 429
    # /metrics stays reachable even when the caller's bucket is drained
    # (it's exempt from the limiter). Prometheus scrapes shouldn't be
    # throttled by a noisy tenant.
    for _ in range(5):
        assert client.get("/metrics").status_code in (200, 404)
        # 404 only if the metrics extra isn't installed — also acceptable.


def test_rate_limit_per_token_scope_isolates_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_RATE_LIMIT_RPS="1",
        SOMA_RATE_LIMIT_BURST="1",
        SOMA_RATE_LIMIT_SCOPE="per-token",
    )
    client = TestClient(reloaded.app)
    # Two distinct tokens -> two distinct jtis -> two distinct buckets.
    token_a = _issue("alex")
    token_b = _issue("alex")  # same sub but fresh jti
    assert client.get("/status", headers=_bearer_headers(token_a)).status_code == 200
    assert client.get("/status", headers=_bearer_headers(token_a)).status_code == 429
    # Token B still has full budget.
    assert client.get("/status", headers=_bearer_headers(token_b)).status_code == 200


def test_rate_limit_per_subject_scope_shares_across_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_RATE_LIMIT_RPS="1",
        SOMA_RATE_LIMIT_BURST="1",
        SOMA_RATE_LIMIT_SCOPE="per-subject",
    )
    client = TestClient(reloaded.app)
    token_a = _issue("alex")
    token_b = _issue("alex")  # same sub, fresh jti
    assert client.get("/status", headers=_bearer_headers(token_a)).status_code == 200
    # Token B reuses the same sub-keyed bucket -> blocked.
    r = client.get("/status", headers=_bearer_headers(token_b))
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_rate_limit_disabled_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    reloaded = _fresh_serve(monkeypatch, SOMA_JWT_SECRET=_SECRET)
    client = TestClient(reloaded.app)
    headers = _bearer_headers(_issue("alex"))
    # Many rapid requests -> all succeed, no 429 at all.
    for _ in range(20):
        assert client.get("/status", headers=headers).status_code == 200


def test_rate_limited_total_metric_increments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reloaded = _fresh_serve(
        monkeypatch,
        SOMA_JWT_SECRET=_SECRET,
        SOMA_RATE_LIMIT_RPS="1",
        SOMA_RATE_LIMIT_BURST="1",
    )
    from soma import rate_limit as rl

    if not getattr(rl, "_PROM_AVAILABLE", False):
        pytest.skip("prometheus-client not installed — metric is a noop")

    client = TestClient(reloaded.app)
    headers = _bearer_headers(_issue("alex"))
    before = float(
        rl.RATE_LIMITED_TOTAL.labels(scope="per-token")._value.get()  # type: ignore[attr-defined]
    )
    assert client.get("/status", headers=headers).status_code == 200
    assert client.get("/status", headers=headers).status_code == 429
    after = float(
        rl.RATE_LIMITED_TOTAL.labels(scope="per-token")._value.get()  # type: ignore[attr-defined]
    )
    assert after - before >= 1.0


def test_rate_limit_open_mode_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Open mode: no SOMA_JWT_SECRET, no SOMA_API_KEY. The rate-limit env
    # is also unset so we're testing "no auth + no limiter" — still
    # frictionless local dev.
    reloaded = _fresh_serve(monkeypatch)
    client = TestClient(reloaded.app)
    for _ in range(10):
        assert client.get("/status").status_code == 200
