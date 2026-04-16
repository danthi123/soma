# Phase 18: Hashed-Token Blocklist + Audience Claim

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Two auth polish items combined:
1. Optional `sha256(jti)` hashing in `FileBlocklist` so the revocation file is safe to exfiltrate (it leaks *whether* a jti is revoked, not *which* jti).
2. Optional `aud` (audience) JWT claim for multi-server fleets — lets a single JWT signing secret serve multiple services without cross-service token replay.

Both are optional, opt-in via env var or kwarg, backward-compatible.

**Architecture:**
- `FileBlocklist(path, hashed=True)` — new kwarg. When True, `add(jti, ...)` stores `sha256(jti).hexdigest()`; `contains(jti)` hashes before lookup. Plain-text mode is the default so existing blocklist files keep working.
- New env var: `SOMA_JWT_BLOCKLIST_HASHED=1` → propagated through `blocklist_from_env`.
- `issue_token(..., audience="my-service")` — new kwarg, populates the `aud` claim.
- `verify_token(..., expected_audience="my-service")` — new kwarg. When set, rejects tokens whose `aud` doesn't match. When unset, behavior unchanged.
- New env var: `SOMA_JWT_AUDIENCE` → propagated through the server-side `require_auth` integration.

**Tech Stack:** `hashlib.sha256` from stdlib. `pyjwt` already supports `aud` natively — we just expose it in our wrapper.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/auth.md`.

---

### Task 1: Hashed-token blocklist

**Files:**
- Modify: `src/soma/auth_revocation.py`
- Modify: `tests/test_auth/test_revocation.py`

**API:**
```python
@dataclass
class FileBlocklist:
    path: Path
    hashed: bool = False   # NEW

    def add(self, jti: str, exp: int, reason: str = "") -> None:
        key = self._key(jti)
        # ... write {"jti_key": key, "exp": exp, "reason": reason[:256], ...}

    def contains(self, jti: str) -> bool:
        return self._key(jti) in self._load_cache()

    def _key(self, jti: str) -> str:
        return hashlib.sha256(jti.encode("utf-8")).hexdigest() if self.hashed else jti
```

**Schema migration:** the JSONL record changes `{"jti": ...}` → `{"jti_key": ...}` when hashed. Keep both keys recognized on load so a half-migrated file doesn't break:
```python
def _load_cache(self) -> set[str]:
    keys = set()
    for record in ...:
        # Accept either the new jti_key or the legacy jti.
        key = record.get("jti_key") or record.get("jti")
        if key:
            keys.add(key)
    return keys
```

When `hashed=True` we write only `jti_key`. When `hashed=False` we write only `jti` (legacy mode, unchanged bytes).

**Step 1: Write failing tests.**
```python
def test_hashed_blocklist_stores_sha256_only(tmp_path):
    bl = FileBlocklist(tmp_path / "bl.jsonl", hashed=True)
    bl.add("my-secret-jti-12345", exp=9999999999, reason="leaked")
    raw = (tmp_path / "bl.jsonl").read_text()
    assert "my-secret-jti-12345" not in raw
    assert hashlib.sha256(b"my-secret-jti-12345").hexdigest() in raw

def test_hashed_blocklist_contains_works(tmp_path):
    bl = FileBlocklist(tmp_path / "bl.jsonl", hashed=True)
    bl.add("abc", exp=9999999999)
    assert bl.contains("abc")
    assert not bl.contains("xyz")

def test_plain_blocklist_unchanged_by_default(tmp_path):
    # hashed=False is the default; existing bytes match the pre-Phase-18 schema.

def test_blocklist_load_accepts_legacy_jti_records(tmp_path):
    # Handwrite a legacy {"jti": "abc", ...} record; verify contains("abc") works
    # even when instantiated with hashed=True.

def test_gc_expired_preserves_hashed_mode(tmp_path):
    # gc_expired() on a hashed blocklist rewrites only hashed records.
```

**Step 2:** Fail.

**Step 3:** Implement per design above. Update `gc_expired` to preserve whatever key scheme each record uses.

**Step 4:** Tests pass.

**Step 5:** `git commit -m "feat(auth): FileBlocklist hashed=True (sha256(jti) at rest)"`

---

### Task 2: `blocklist_from_env` wires the hashed flag

**Files:**
- Modify: `src/soma/auth_revocation.py` (the `blocklist_from_env` helper)
- Extend: `tests/test_auth/test_revocation.py`

**Env contract:**
- `SOMA_JWT_BLOCKLIST_HASHED=1` → instantiate with `hashed=True`
- unset or `0` → `hashed=False` (default)

**Step 5:** `git commit -m "feat(auth): SOMA_JWT_BLOCKLIST_HASHED env gate"`

---

### Task 3: `aud` claim support

**Files:**
- Modify: `src/soma/auth.py`
- Modify: `tests/test_auth/test_auth_core.py`

**API:**
```python
def issue_token(
    *,
    sub: str,
    bundles: dict[str, list[str]] | None = None,
    expires_in: timedelta,
    secret: str | None = None,
    private_key_pem: bytes | None = None,
    alg: str = "HS256",
    audience: str | None = None,   # NEW
) -> str:
    claims = _build_claims(sub=sub, bundles=bundles, expires_in=expires_in)
    if audience is not None:
        claims["aud"] = audience
    ...


def verify_token(
    token: str,
    *,
    secret: str | None = None,
    public_key_pem: bytes | None = None,
    alg: str = "HS256",
    blocklist: BlocklistBackend | None = None,
    expected_audience: str | None = None,   # NEW
) -> Principal:
    ...
    # Pyjwt accepts audience= in jwt.decode(); pass through.
    decoded = jwt.decode(
        token, key=key, algorithms=[alg],
        audience=expected_audience,
    )
    ...
```

**Tests:**
```python
def test_aud_round_trip():
    t = issue_token(sub="a", expires_in=timedelta(minutes=5), secret="x", audience="svc-A")
    p = verify_token(t, secret="x", expected_audience="svc-A")
    assert p.sub == "a"

def test_aud_mismatch_raises():
    t = issue_token(sub="a", expires_in=timedelta(minutes=5), secret="x", audience="svc-A")
    with pytest.raises(Exception):  # pyjwt raises InvalidAudienceError
        verify_token(t, secret="x", expected_audience="svc-B")

def test_aud_unset_on_token_is_fine_when_not_required():
    # Old tokens without aud still verify when expected_audience=None.

def test_aud_required_but_missing_raises():
    # Token has no aud; verify with expected_audience set → raises.
```

**Step 5:** `git commit -m "feat(auth): audience claim + verify_token expected_audience"`

---

### Task 4: CLI + serve wiring

**Files:**
- Modify: `src/soma/cli.py` — `soma auth issue --audience svc-A`
- Modify: `src/soma/serve.py` — read `SOMA_JWT_AUDIENCE` env; pass to `verify_token`
- Extend: `tests/test_cli.py`, `tests/test_serve/test_jwt_auth.py`

**Env contract:**
- `SOMA_JWT_AUDIENCE=svc-A` on the server → `verify_token(..., expected_audience="svc-A")`.
- Unset → no audience check (today's behaviour).

**Step 5:** `git commit -m "feat(serve): SOMA_JWT_AUDIENCE env + --audience CLI flag"`

---

### Final sanity

```bash
ruff check src/soma/auth.py src/soma/auth_revocation.py src/soma/cli.py src/soma/serve.py tests/test_auth tests/test_serve tests/test_cli.py
SOMA_EMBED_MODEL=stub pytest tests/test_auth tests/test_serve tests/test_cli.py -q
```

Baseline per earlier runs: ~125 tests in test_auth + test_serve combined, 21-24 in test_cli. Target: +~10 new tests, 0 regressions.

**Compatibility note:** both features strictly additive. Old tokens / old blocklist files / old callers keep working unchanged because the new kwargs default to None / False.
