"""In-process per-token rate limiter for the SOMA REST API (Phase 26).

A tiny token-bucket primitive plus a front-end that stores one bucket
per key (typically a JWT ``jti`` — falls back to ``sub`` when the token
predates Phase 18). Buckets are pruned after a configurable idle window
so long-lived servers don't leak memory.

This is **not** a replacement for a real WAF or reverse-proxy rate
limiter. A single worker can still be saturated by unauthenticated
traffic, and there's no coordination across workers. The knob is here
to give single-binary deploys a cheap in-proc abuse ceiling.

Config via env (all unset -> disabled, behaviour unchanged):

- ``SOMA_RATE_LIMIT_RPS`` — steady-state requests per second (float).
- ``SOMA_RATE_LIMIT_BURST`` — max tokens in bucket (int, defaults to
  ``max(1, ceil(RPS))``).
- ``SOMA_RATE_LIMIT_SCOPE`` — ``per-token`` (default) or ``per-subject``
  (shares a bucket across refreshes of the same ``sub``). Read by the
  serve-layer middleware, not this module.

Token-bucket math is monotonic-clock based so NTP slew, DST, and manual
clock rewinds don't corrupt the accounting. Tests inject ``now=`` to
step time deterministically without ``sleep``.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

__all__ = ["RateLimiter", "TokenBucket"]


# ---------------------------------------------------------------------------
# Primitive: token bucket
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Classic token-bucket rate-limiter primitive.

    ``tokens`` floats between 0 and ``burst``. Each call to
    :meth:`try_acquire` refills proportionally to the time elapsed
    since ``last_refill`` at ``rps`` tokens/sec, caps at ``burst``,
    and consumes one token if available.
    """

    rps: float
    burst: int
    tokens: float
    last_refill: float

    def _refill(self, now: float) -> None:
        """Credit any tokens earned since ``last_refill``.

        No-op when ``now <= last_refill`` (monotonic-clock invariant —
        just in case the caller passes a stale timestamp).
        """
        delta = now - self.last_refill
        if delta <= 0:
            return
        self.tokens = min(float(self.burst), self.tokens + delta * self.rps)
        self.last_refill = now

    def try_acquire(self, now: float) -> bool:
        """Refill, then consume one token if available.

        Returns ``True`` on success (request allowed), ``False`` when
        the bucket is empty (request should be rejected).
        """
        self._refill(now)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def retry_after(self, now: float) -> float:
        """Seconds until one token becomes available.

        Assumes :meth:`try_acquire` just returned ``False``. For an
        already-full-ish bucket this returns ``<= 0``; callers may
        treat that as "retry immediately".
        """
        self._refill(now)
        if self.tokens >= 1.0:
            return 0.0
        if self.rps <= 0.0:
            # Pathological config — caller shouldn't construct a bucket
            # with rps=0, but guard against div-by-zero anyway.
            return float("inf")
        needed = 1.0 - self.tokens
        return needed / self.rps


# ---------------------------------------------------------------------------
# Front-end: per-key bucket store with idle eviction
# ---------------------------------------------------------------------------


@dataclass
class _BucketEntry:
    """Bucket plus the last-seen timestamp used for idle eviction."""

    bucket: TokenBucket
    last_seen: float = 0.0


class RateLimiter:
    """Per-key token-bucket store with idle-bucket eviction.

    Typical usage (HTTP middleware):

        limiter = RateLimiter.from_env()
        if limiter is not None:
            allowed, retry_after = limiter.check(principal.jti)
            if not allowed:
                raise HTTPException(429, headers={"Retry-After": ...})

    ``from_env()`` returns ``None`` when ``SOMA_RATE_LIMIT_RPS`` is
    unset — that's the switch-off path.
    """

    def __init__(
        self,
        *,
        rps: float,
        burst: int,
        idle_evict_seconds: float = 300.0,
    ) -> None:
        if rps <= 0.0:
            raise ValueError(f"rps must be > 0, got {rps!r}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst!r}")
        self._rps = float(rps)
        self._burst = int(burst)
        self._idle_evict_seconds = float(idle_evict_seconds)
        self._buckets: dict[str, _BucketEntry] = {}

    def check(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Attempt one acquisition against ``key``'s bucket.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is
        ``0.0`` on the allowed path. ``now`` is injectable for tests —
        production callers leave it ``None`` and we read from
        ``time.monotonic()`` so clock jumps don't corrupt the window.
        """
        t = time.monotonic() if now is None else now
        self._evict_idle(t)
        entry = self._buckets.get(key)
        if entry is None:
            entry = _BucketEntry(
                bucket=TokenBucket(
                    rps=self._rps,
                    burst=self._burst,
                    tokens=float(self._burst),
                    last_refill=t,
                ),
                last_seen=t,
            )
            self._buckets[key] = entry
        entry.last_seen = t
        allowed = entry.bucket.try_acquire(now=t)
        if allowed:
            return True, 0.0
        return False, entry.bucket.retry_after(now=t)

    def _evict_idle(self, now: float) -> None:
        """Drop entries whose ``last_seen`` is older than the window.

        Called opportunistically on every ``check()``. O(n) scan, which
        is fine for the 10^3–10^4 key counts this limiter is sized for;
        anything larger wants a distributed store anyway.
        """
        cutoff = now - self._idle_evict_seconds
        stale = [k for k, e in self._buckets.items() if e.last_seen < cutoff]
        for k in stale:
            del self._buckets[k]

    # ------------------------------------------------------------------
    # Env-driven construction
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> RateLimiter | None:
        """Build a RateLimiter from env vars, or ``None`` if disabled.

        Reads ``SOMA_RATE_LIMIT_RPS`` (required; unset -> returns None)
        and ``SOMA_RATE_LIMIT_BURST`` (optional; defaults to
        ``max(1, ceil(rps))``). Malformed values raise ``ValueError``
        at import time so a misconfigured server crashes loud rather
        than silently serving with no limit.
        """
        raw_rps = os.environ.get("SOMA_RATE_LIMIT_RPS", "").strip()
        if not raw_rps:
            return None
        try:
            rps = float(raw_rps)
        except ValueError as exc:
            raise ValueError(
                f"SOMA_RATE_LIMIT_RPS={raw_rps!r} is not a valid float"
            ) from exc
        raw_burst = os.environ.get("SOMA_RATE_LIMIT_BURST", "").strip()
        if raw_burst:
            try:
                burst = int(raw_burst)
            except ValueError as exc:
                raise ValueError(
                    f"SOMA_RATE_LIMIT_BURST={raw_burst!r} is not a valid int"
                ) from exc
        else:
            burst = max(1, math.ceil(rps))
        return cls(rps=rps, burst=burst)
