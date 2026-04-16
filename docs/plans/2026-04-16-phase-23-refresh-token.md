# Phase 23: Refresh-Token Endpoint

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Add `POST /auth/refresh` so clients can exchange a valid (non-expired, non-revoked) bearer token for a fresh one. Avoids forcing users to re-authenticate mid-session when tokens carry a conservative `exp`. Deliberately scoped down from a full OAuth refresh-token flow — no separate refresh-token class, no rotation tracking, just "mint a new access token with the same claims but a fresh exp + jti."

**Architecture:**
- New endpoint: `POST /auth/refresh` — reads the current bearer token from `Authorization:` header, verifies it via existing `require_auth`, mints a new token via `issue_token()` reusing `sub` / `bundles` / `audience` (if present) with a fresh `jti` and `exp`.
- `exp` extension configurable via env `SOMA_JWT_REFRESH_TTL` (default: same as the original token's `iat → exp` window, capped by `SOMA_JWT_MAX_TTL` if set).
- Old token NOT auto-revoked — deliberate. Revocation is a separate `POST /auth/revoke` (already shipped). Operators who want rotation can call revoke + refresh together from client code.
- Server needs signing materials on the request path — i.e. `SOMA_JWT_SECRET` (HS256) or `SOMA_JWT_PRIVATE_KEY_PATH` (RS256). Today the server only needs the public material for verification; refresh requires private-key access. Document this clearly.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/auth.md`.

---

### Task 1: Helper function in `soma.auth`

**Files:**
- Modify: `src/soma/auth.py` — add `refresh_token(current_token, ...)` helper
- Extend: `tests/test_auth/test_auth_core.py`

**API:**
```python
def refresh_token(
    current_token: str,
    *,
    secret: str | None = None,
    private_key_pem: bytes | None = None,
    alg: str = "HS256",
    new_expires_in: timedelta | None = None,
    blocklist: BlocklistBackend | None = None,
) -> str:
    """Verify the current token; mint a fresh one with the same sub /
    bundles / audience but a new jti and exp.

    Raises the same exceptions as ``verify_token`` (expired, revoked,
    invalid signature). ``new_expires_in`` defaults to the original
    token's ``exp - iat`` window.
    """
```

**Step 1: Tests.**
```python
def test_refresh_round_trip_preserves_claims():
    t1 = issue_token(sub="a", bundles={"b": ["read"]}, expires_in=timedelta(hours=1), secret="x")
    t2 = refresh_token(t1, secret="x")
    p1 = verify_token(t1, secret="x")
    p2 = verify_token(t2, secret="x")
    assert p1.sub == p2.sub
    assert p1.bundles == p2.bundles
    assert p1.jti != p2.jti   # fresh jti

def test_refresh_extends_exp():
    # new exp > old exp by roughly the new_expires_in delta.

def test_refresh_rejects_expired_token():
    t = issue_token(sub="a", expires_in=timedelta(seconds=-1), secret="x")
    with pytest.raises(...):
        refresh_token(t, secret="x")

def test_refresh_rejects_revoked_token():
    # Token revoked in blocklist; refresh raises.

def test_refresh_preserves_audience_when_present():
    ...

def test_refresh_default_ttl_matches_original_window():
    # Original expires_in=timedelta(days=30); refresh without new_expires_in
    # mints a token with exp - iat ~= 30 days.
```

**Step 5:** `git commit -m "feat(auth): refresh_token helper"`

---

### Task 2: REST endpoint

**Files:**
- Modify: `src/soma/serve.py`
- Extend: `tests/test_serve/test_jwt_auth.py`

**Endpoint:**
```python
@app.post("/auth/refresh")
def auth_refresh(
    principal: Principal = Depends(require_auth(None, "read")),
    request: Request = ...,
) -> dict:
    """Exchange the current bearer for a fresh token with the same claims."""
    # Extract the raw token string from the Authorization header.
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer")
    raw = auth[len("Bearer "):]

    # Re-mint via helper.
    try:
        new_token = refresh_token(
            raw,
            secret=os.environ.get("SOMA_JWT_SECRET") or None,
            private_key_pem=_jwt_private_key_pem(),
            alg=os.environ.get("SOMA_JWT_ALG", "HS256"),
            new_expires_in=_refresh_ttl_from_env(),
            blocklist=_blocklist,
        )
    except Exception as exc:
        raise HTTPException(401, f"refresh failed: {exc}")
    return {"token": new_token, "exp": _extract_exp(new_token)}
```

`_refresh_ttl_from_env()` reads `SOMA_JWT_REFRESH_TTL` (format: `30d | 24h | 60m`, same parser as `soma auth issue --expires`). Defaults to `None` (match original window).

**Tests:**
```python
def test_refresh_endpoint_happy_path(monkeypatch):
    # Mint a token; call POST /auth/refresh with it; get back a new valid token.

def test_refresh_endpoint_without_bearer_returns_401():
    ...

def test_refresh_endpoint_with_expired_token_returns_401():
    ...

def test_refresh_endpoint_with_revoked_token_returns_401():
    # Token in the blocklist; refresh fails.

def test_refresh_endpoint_respects_SOMA_JWT_REFRESH_TTL(monkeypatch):
    monkeypatch.setenv("SOMA_JWT_REFRESH_TTL", "60m")
    # Refreshed exp should be ~60 min from now, not the original window.
```

**Step 5:** `git commit -m "feat(serve): POST /auth/refresh endpoint"`

---

### Task 3: CLI support (optional convenience)

**Files:**
- Modify: `src/soma/cli.py` — `soma auth refresh --token <jwt>` verb
- Extend: `tests/test_cli.py`

Mirrors `soma auth verify` structure. Lower priority than Tasks 1-2; if the work budget is tight, ship Tasks 1-2 and cut this.

**Step 5:** `git commit -m "feat(cli): soma auth refresh"`

---

### Final sanity

```bash
ruff check src/soma tests
pytest tests/test_auth tests/test_serve tests/test_cli.py -q
mypy src/soma/auth.py
```

Baseline: ~158 tests in auth+serve+cli post-Phase-18 + Phase-20. Target +10 new tests, 0 regressions.

**Security considerations worth noting in the commit:**
1. Refresh doesn't require re-auth — the bearer is proof enough. Intentional; matches how short-lived access tokens are typically rotated. Operators wanting stronger guarantees should use a separate `/auth/rotate` flow with a different credential class.
2. The refreshed token carries the same `jti` → no, fresh jti is mandatory so a future revoke of the old jti doesn't affect the new one. Pin this in a test.
3. If a revoked token is presented for refresh, refresh must fail (enforced via the blocklist kwarg to `verify_token` inside `refresh_token`).
