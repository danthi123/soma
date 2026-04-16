# Phase 26: Per-Token Rate Limiting

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Add an in-process rate limiter keyed on `jti` (token id) that
caps requests-per-second per authenticated token. Originally punted to
"use a reverse proxy" — shipping an in-proc fallback so deployments
without an upstream proxy get basic abuse protection. Small, opt-in,
cheap — not a replacement for a real WAF.

**Architecture:**
- New module `src/soma/rate_limit.py` with a `TokenBucket` primitive
  and a `RateLimiter` front-end.
- Per-`jti` buckets stored in a dict keyed by jti (fallback to `sub`
  if jti missing). Bucket evicted after `exp - now` seconds of
  inactivity so dead tokens don't leak memory.
- Config via env (all unset → disabled, behaviour unchanged):
  * `SOMA_RATE_LIMIT_RPS` — steady-state requests per second (float).
  * `SOMA_RATE_LIMIT_BURST` — max tokens in bucket (int, defaults to
    `max(1, ceil(RPS))`).
  * `SOMA_RATE_LIMIT_SCOPE` — `per-token` (default) or `per-subject`
    (shares a bucket across refreshes of the same `sub`).
- Middleware in `src/soma/serve.py` runs after `require_auth`. On
  exhaustion: HTTP 429 + `Retry-After: <seconds>` header.
- Metric: `soma_rate_limited_total` counter (label: `scope`). No
  per-jti/sub label — that's a cardinality explosion risk. The
  operator's grafana query will be "total 429s / minute" which is the
  actionable number.

**Out-of-scope:**
- Distributed rate limiting (Redis-backed). Trivially layerable later
  — the `RateLimiter` interface stays the same, just swap the store.
- Rate limit on *unauthenticated* endpoints. Those should live behind
  a real proxy.
- `/metrics` or `/health` limits — always exempt.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/auth.md`, `deferred-items.md` strikethrough.

---

### Task 1: `TokenBucket` + `RateLimiter` primitives

**Files:**
- Create: `src/soma/rate_limit.py`
- Create: `tests/test_rate_limit.py`

**API:**
```python
@dataclass
class TokenBucket:
    rps: float           # refill rate in tokens per second
    burst: int           # capacity
    tokens: float        # current token count (seeded to burst)
    last_refill: float   # monotonic timestamp

    def try_acquire(self, now: float) -> bool:
        # Refill, then try to consume one token.
        # Returns True if acquired, False if bucket empty.

    def retry_after(self, now: float) -> float:
        # Seconds until one token is available. Assumes try_acquire
        # just returned False.


class RateLimiter:
    def __init__(
        self,
        *,
        rps: float,
        burst: int,
        idle_evict_seconds: float = 300.0,
    ) -> None: ...

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). retry_after is 0 when allowed."""

    @classmethod
    def from_env(cls) -> "RateLimiter | None":
        """Returns None if SOMA_RATE_LIMIT_RPS unset."""
```

**Step 1: Failing tests.**
```python
def test_token_bucket_allows_up_to_burst():
    b = TokenBucket(rps=1.0, burst=3, tokens=3.0, last_refill=0.0)
    for _ in range(3):
        assert b.try_acquire(now=0.0)
    assert not b.try_acquire(now=0.0)

def test_token_bucket_refills_over_time():
    b = TokenBucket(rps=1.0, burst=1, tokens=0.0, last_refill=0.0)
    assert not b.try_acquire(now=0.5)
    assert b.try_acquire(now=1.0)

def test_token_bucket_retry_after_reports_seconds():
    b = TokenBucket(rps=2.0, burst=1, tokens=0.0, last_refill=0.0)
    b.try_acquire(now=0.0)   # consumes whatever's there
    assert b.retry_after(now=0.0) == pytest.approx(0.5, rel=0.01)

def test_rate_limiter_per_key_isolation():
    r = RateLimiter(rps=1.0, burst=1)
    assert r.check("a")[0]         # "a" consumed
    assert not r.check("a")[0]     # "a" exhausted
    assert r.check("b")[0]         # "b" untouched

def test_rate_limiter_idle_eviction():
    r = RateLimiter(rps=1.0, burst=1, idle_evict_seconds=60.0)
    r.check("a", now=0.0)
    r.check("b", now=0.0)
    # After 120s of silence, "a" and "b" buckets are gone
    r.check("c", now=120.0)   # triggers prune
    assert "a" not in r._buckets
    assert "b" not in r._buckets

def test_rate_limiter_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("SOMA_RATE_LIMIT_RPS", raising=False)
    assert RateLimiter.from_env() is None

def test_rate_limiter_from_env_reads_rps_and_burst(monkeypatch):
    monkeypatch.setenv("SOMA_RATE_LIMIT_RPS", "5")
    monkeypatch.setenv("SOMA_RATE_LIMIT_BURST", "10")
    r = RateLimiter.from_env()
    assert r is not None
    assert r._rps == 5.0
    assert r._burst == 10

def test_rate_limiter_from_env_default_burst_ceils_rps(monkeypatch):
    monkeypatch.setenv("SOMA_RATE_LIMIT_RPS", "2.5")
    r = RateLimiter.from_env()
    assert r._burst == 3
```

**Step 5:** `git commit -m "feat(rate_limit): TokenBucket + RateLimiter primitives"`

---

### Task 2: Server middleware

**Files:**
- Modify: `src/soma/serve.py`
- Extend: `tests/test_serve/test_jwt_auth.py` (or new `test_rate_limit_serve.py`)

**Middleware sketch:**
```python
_RATE_LIMITER: RateLimiter | None = RateLimiter.from_env()

async def _enforce_rate_limit(request: Request, principal: Principal) -> None:
    if _RATE_LIMITER is None:
        return
    if request.url.path in {"/metrics", "/health"}:
        return
    key = principal.jti if _RATE_LIMIT_SCOPE == "per-token" else principal.sub
    allowed, retry_after = _RATE_LIMITER.check(key)
    if not allowed:
        _RATE_LIMITED_TOTAL.labels(scope=_RATE_LIMIT_SCOPE).inc()
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": f"{retry_after:.2f}"},
        )
```

Wire into `require_auth` — after verification, run the limiter check
with the principal. Keeps the decorator surface unchanged for
callers; no per-endpoint wiring needed.

**Metric declaration:**
```python
_RATE_LIMITED_TOTAL = Counter(
    "soma_rate_limited_total",
    "Total requests rejected by the rate limiter.",
    labelnames=("scope",),
    registry=_REGISTRY,
)
```

**Tests:**
```python
def test_rate_limit_429_after_burst(monkeypatch, tmp_path):
    monkeypatch.setenv("SOMA_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("SOMA_RATE_LIMIT_BURST", "2")
    # Reload serve module to pick up env

    client = _make_client(...)
    token = _issue(...)
    headers = {"Authorization": f"Bearer {token}"}
    # 2 requests fast → 200; 3rd → 429 + Retry-After
    assert client.get("/bundles", headers=headers).status_code == 200
    assert client.get("/bundles", headers=headers).status_code == 200
    r = client.get("/bundles", headers=headers)
    assert r.status_code == 429
    assert "Retry-After" in r.headers

def test_rate_limit_exempts_metrics_and_health(monkeypatch):
    # /metrics and /health bypass the limiter even after burst
    # exhaustion on another endpoint.

def test_rate_limit_per_token_scope_isolates_tokens(monkeypatch):
    # Two tokens; one exhausts, the other still has budget.

def test_rate_limit_per_subject_scope_shares_across_refresh(monkeypatch):
    # Issue token-A for sub=X, exhaust; issue fresh token-B for sub=X;
    # token-B is also rate-limited.

def test_rate_limit_disabled_when_env_unset():
    # No env set → no 429 even after many requests.

def test_rate_limited_total_metric_increments(monkeypatch):
    # After a 429, /metrics shows soma_rate_limited_total{scope="..."} == 1
```

**Step 5:** `git commit -m "feat(serve): in-proc per-token rate limiter + 429 responses"`

---

### Task 3: Docs

**Files:**
- Modify: `docs/auth.md` — short section on when to enable the in-proc
  limiter vs. relying on a reverse proxy. Env var table.

**Step 5:** `git commit -m "docs(auth): in-proc rate limiter env vars"`

---

### Final sanity

```bash
ruff check src/soma tests
pytest tests/test_serve tests/test_rate_limit.py -q
```

Baseline post-Phase-23 serve/auth suite: ~225 tests. Target +~10 new,
0 regressions.

**Security notes worth a commit-message line:**
1. In-proc limiter is NOT a WAF. A single worker can still be
   saturated by unauth'd traffic; put a real proxy (nginx, Caddy,
   Cloudflare) in front for production.
2. `per-subject` scope shares a bucket across refreshes — operators
   who want per-refresh isolation should stick with `per-token`.
3. Token-bucket math is monotonic-clock based, so jumps / DST /
   NTP-slew don't break accounting. Tests use injected `now=`.
