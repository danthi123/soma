# Phase 20: Redis-Backed Revocation Blocklist

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Add an optional `RedisBlocklist` alongside the shipped `FileBlocklist` for multi-host / k8s deploys where the file-backed store's 30-second poll lag is too slow. Instant propagation across workers + automatic TTL equal to the token's `exp` claim.

**Architecture:** Pure addition. Existing `FileBlocklist` stays the default. New class `RedisBlocklist` implements the same `BlocklistBackend` Protocol. New optional extra `soma[redis-revocation]`. `blocklist_from_env()` picks the Redis backend when `SOMA_JWT_BLOCKLIST_REDIS_URL` is set, falling through to file-backed otherwise.

**Tech Stack:** `redis-py` (sync client). Already widely deployed; minimal dep surface.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/auth.md`.

---

### Task 1: `RedisBlocklist` class

**Files:**
- Modify: `src/soma/auth_revocation.py` — add `RedisBlocklist` class alongside `FileBlocklist`
- Modify: `pyproject.toml` — add `redis-revocation = ["redis>=5.0"]` extra
- Create: `tests/test_auth/test_redis_revocation.py`

**API (mirrors `FileBlocklist`):**
```python
@dataclass
class RedisBlocklist:
    url: str                # e.g. "redis://localhost:6379/0"
    key_prefix: str = "soma:jwt:revoked:"
    hashed: bool = False

    def add(self, jti: str, exp: int, reason: str = "") -> None:
        key = self._key(jti)
        ttl = max(1, exp - int(time.time()))   # seconds until natural expiry
        self._client.setex(key, ttl, reason[:256] or "revoked")

    def contains(self, jti: str) -> bool:
        return self._client.exists(self._key(jti)) == 1

    def gc_expired(self) -> int:
        # No-op — Redis TTL auto-expires entries. Return 0.

    def _key(self, jti: str) -> str:
        if self.hashed:
            jti = hashlib.sha256(jti.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}{jti}"
```

**Test strategy:** use `fakeredis` (pytest extra) for unit tests so CI doesn't need a real Redis. Guard the import — if fakeredis not installed, `pytest.skip` the tests. Mirrors the existing pattern for `lancedb` and `qdrant` tests.

**Step 1: Write failing tests.**
```python
def test_redis_add_sets_key_with_ttl():
    bl = RedisBlocklist(url="redis://fake", key_prefix="t:")
    bl.add("abc", exp=int(time.time()) + 60)
    assert bl.contains("abc")
    # Verify TTL <= 60s via the fake client.

def test_redis_natural_expiry_drops_entry(freeze_time):
    bl = RedisBlocklist(url="redis://fake")
    bl.add("abc", exp=int(time.time()) + 10)
    assert bl.contains("abc")
    freeze_time.tick(11)
    assert not bl.contains("abc")

def test_redis_hashed_mode_matches_file_hashed_mode():
    # Same jti hashes identically across backends — interop test.

def test_redis_gc_expired_is_noop_returning_zero():
    ...

def test_redis_import_error_on_missing_dep():
    # If redis SDK not installed, constructing raises ImportError with
    # a helpful message pointing at the extra.
```

**Step 2-4: TDD.**

**Step 5:** `git commit -m "feat(auth): RedisBlocklist for multi-host deploys"`

---

### Task 2: `blocklist_from_env` dispatches to Redis when URL set

**Files:**
- Modify: `src/soma/auth_revocation.py` (the `blocklist_from_env` helper)
- Extend: `tests/test_auth/test_revocation.py`

**Contract:**
- `SOMA_JWT_BLOCKLIST_REDIS_URL=redis://...` → instantiate `RedisBlocklist` with that URL. `SOMA_JWT_BLOCKLIST_HASHED=1` still honored.
- Unset → fall through to `FileBlocklist` with the existing `SOMA_JWT_BLOCKLIST_PATH` default.
- Both set → Redis wins; log a deprecation-flavour warning that `SOMA_JWT_BLOCKLIST_PATH` is ignored.

**Step 5:** `git commit -m "feat(auth): SOMA_JWT_BLOCKLIST_REDIS_URL env dispatch"`

---

### Task 3: Docs touch

**Files:**
- Modify: `docs/auth.md` (small section on Redis blocklist — when to use it, env vars, minimal docker-compose entry)

**Step 5:** `git commit -m "docs(auth): Redis blocklist section"`

---

### Final sanity

```bash
pip install fakeredis  # one-time local install for tests
ruff check src/soma/auth_revocation.py tests/test_auth
SOMA_EMBED_MODEL=stub pytest tests/test_auth -q
mypy src/soma/auth_revocation.py
```

Baseline ~95 tests in test_auth; target +~8 new Redis tests, 0 regressions.

**If fakeredis is unavailable:** skip the Redis tests cleanly (pytest.skip at module level) rather than failing. This keeps the CI runnable without the dev-only dep.
