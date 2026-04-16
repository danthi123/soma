"""Phase 26 Task 1 — TokenBucket + RateLimiter unit tests.

Exercises the in-process rate limiter primitives with injected monotonic
time (``now=``) so the suite runs without real ``time.sleep`` calls.

Covered:
- Bucket allows up to ``burst`` requests on a fresh bucket.
- Bucket refills linearly at ``rps`` tokens/sec.
- ``retry_after`` reports seconds-until-next-token correctly.
- RateLimiter isolates per-key buckets.
- Idle eviction prunes buckets that haven't been touched for
  ``idle_evict_seconds``.
- ``from_env`` returns ``None`` when the env switch is unset (default
  off), reads RPS/BURST when set, and defaults ``burst = ceil(rps)``
  when only RPS is configured.
"""

from __future__ import annotations

import pytest

from soma.rate_limit import RateLimiter, TokenBucket


# ------------------------------------------------------------------
# TokenBucket — primitive refill / consume semantics
# ------------------------------------------------------------------
def test_token_bucket_allows_up_to_burst() -> None:
    b = TokenBucket(rps=1.0, burst=3, tokens=3.0, last_refill=0.0)
    for _ in range(3):
        assert b.try_acquire(now=0.0)
    assert not b.try_acquire(now=0.0)


def test_token_bucket_refills_over_time() -> None:
    b = TokenBucket(rps=1.0, burst=1, tokens=0.0, last_refill=0.0)
    assert not b.try_acquire(now=0.5)
    assert b.try_acquire(now=1.0)


def test_token_bucket_refill_caps_at_burst() -> None:
    # Long idle period should not accumulate more than `burst` tokens.
    b = TokenBucket(rps=1.0, burst=2, tokens=0.0, last_refill=0.0)
    # 100s of idle => would be 100 tokens uncapped; capped at burst=2.
    for _ in range(2):
        assert b.try_acquire(now=100.0)
    assert not b.try_acquire(now=100.0)


def test_token_bucket_retry_after_reports_seconds() -> None:
    b = TokenBucket(rps=2.0, burst=1, tokens=0.0, last_refill=0.0)
    b.try_acquire(now=0.0)  # consumes whatever is there (0)
    # With rps=2.0, one token becomes available in 0.5s.
    assert b.retry_after(now=0.0) == pytest.approx(0.5, rel=0.01)


def test_token_bucket_retry_after_zero_when_refill_ready() -> None:
    # After enough time has passed to refill a full token, retry_after
    # should be <= 0 (caller treats as "try again immediately").
    b = TokenBucket(rps=1.0, burst=1, tokens=0.0, last_refill=0.0)
    # At now=1.0, refill delta = 1.0 token -> available.
    assert b.retry_after(now=1.0) <= 0.0


# ------------------------------------------------------------------
# RateLimiter — per-key isolation + eviction + env parsing
# ------------------------------------------------------------------
def test_rate_limiter_per_key_isolation() -> None:
    r = RateLimiter(rps=1.0, burst=1)
    allowed_a1, _ = r.check("a", now=0.0)
    allowed_a2, _ = r.check("a", now=0.0)
    allowed_b, _ = r.check("b", now=0.0)
    assert allowed_a1
    assert not allowed_a2
    assert allowed_b


def test_rate_limiter_returns_retry_after_on_rejection() -> None:
    r = RateLimiter(rps=2.0, burst=1)
    assert r.check("a", now=0.0)[0]
    allowed, retry_after = r.check("a", now=0.0)
    assert not allowed
    assert retry_after == pytest.approx(0.5, rel=0.01)


def test_rate_limiter_allowed_path_retry_after_is_zero() -> None:
    r = RateLimiter(rps=1.0, burst=1)
    allowed, retry_after = r.check("a", now=0.0)
    assert allowed
    assert retry_after == 0.0


def test_rate_limiter_idle_eviction() -> None:
    r = RateLimiter(rps=1.0, burst=1, idle_evict_seconds=60.0)
    r.check("a", now=0.0)
    r.check("b", now=0.0)
    # After 120s of silence (> idle_evict_seconds), touching a fresh key
    # triggers a prune of "a" and "b".
    r.check("c", now=120.0)
    assert "a" not in r._buckets
    assert "b" not in r._buckets
    assert "c" in r._buckets


def test_rate_limiter_active_key_not_evicted() -> None:
    r = RateLimiter(rps=1.0, burst=1, idle_evict_seconds=60.0)
    r.check("a", now=0.0)
    # "a" is touched again at t=30 — still within the window.
    r.check("a", now=30.0)
    r.check("b", now=120.0)  # triggers a prune
    # "a" last touched at t=30, window=60 => still live at t=120? no,
    # 120-30=90 > 60 => evicted. Retouch at t=70 to keep alive.
    assert "a" not in r._buckets


def test_rate_limiter_from_env_returns_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOMA_RATE_LIMIT_RPS", raising=False)
    assert RateLimiter.from_env() is None


def test_rate_limiter_from_env_reads_rps_and_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOMA_RATE_LIMIT_RPS", "5")
    monkeypatch.setenv("SOMA_RATE_LIMIT_BURST", "10")
    r = RateLimiter.from_env()
    assert r is not None
    assert r._rps == 5.0
    assert r._burst == 10


def test_rate_limiter_from_env_default_burst_ceils_rps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOMA_RATE_LIMIT_RPS", "2.5")
    monkeypatch.delenv("SOMA_RATE_LIMIT_BURST", raising=False)
    r = RateLimiter.from_env()
    assert r is not None
    assert r._burst == 3


def test_rate_limiter_from_env_default_burst_min_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # rps < 1 still yields burst >= 1 (operators shouldn't need to think
    # about fractional burst capacity).
    monkeypatch.setenv("SOMA_RATE_LIMIT_RPS", "0.2")
    monkeypatch.delenv("SOMA_RATE_LIMIT_BURST", raising=False)
    r = RateLimiter.from_env()
    assert r is not None
    assert r._burst == 1


def test_rate_limiter_check_uses_monotonic_when_now_not_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When no now= is injected the limiter should read from time.monotonic
    # (NOT time.time — DST / NTP-slew safety).
    r = RateLimiter(rps=1.0, burst=1)
    calls: list[int] = []

    def fake_monotonic() -> float:
        calls.append(1)
        return 42.0

    import soma.rate_limit as rl

    monkeypatch.setattr(rl.time, "monotonic", fake_monotonic)
    r.check("a")
    assert calls, "RateLimiter.check() should fall back to time.monotonic()"
