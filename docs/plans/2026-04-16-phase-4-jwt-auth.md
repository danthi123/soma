# Phase 4 — Per-bundle JWT auth

> **For Claude:** Execute via TDD after Phase 3 (observability) lands. Each task has failing tests first, then minimal impl, then commit.

**Goal:** Replace the single shared `SOMA_API_KEY` bearer token with signed JWTs carrying per-bundle permissions. A token can say "bearer can read+write bundle `alex` but nothing else." Keep `SOMA_API_KEY` as a backward-compat admin escape hatch with a deprecation header.

**Architecture:**
- **PyJWT 2.x** (added to `serve` extra as `pyjwt[crypto]>=2.8`). Reject `alg=none`. `exp` required + `leeway=60s` for NTP drift.
- **HS256 default** (shared secret via `SOMA_JWT_SECRET`), **RS256 opt-in** for issuer/verifier separation.
- **Claim shape:** standard `iss/sub/iat/exp/jti` + `soma: {v:1, bundles: {name: [perms]}}`. Three perm tiers: `read` / `write` / `admin`.
- **New module `src/soma/auth.py`** shared between `serve.py` and `cli.py` (claim builder, decoder with leeway, scope check). Keeps FastAPI deps thin and CLI free of HTTP imports.
- **CLI:** `soma auth issue / verify / rotate-secret`. No `list` (needs persistent state).
- **Edge cases to build:** clock skew leeway, `exp` required, `alg=none` reject, `jti` for future revocation. Punt: refresh tokens, revocation blocklist, rate limiting.

**Tech stack:** `pyjwt[crypto]>=2.8`. Everything else stdlib.

---

### Task 1: `soma.auth` module

**Files:**
- Create: `src/soma/auth.py`
- Modify: `pyproject.toml` — add `pyjwt[crypto]>=2.8` to `serve` extra
- Test: `tests/test_auth/test_auth_core.py`

**Step 1:** failing tests:
- `test_issue_and_verify_hs256_roundtrip` — encode then decode, claims identical.
- `test_verify_rejects_missing_exp` — token without `exp` → raises.
- `test_verify_rejects_alg_none` — `alg=none` → raises.
- `test_verify_accepts_clock_skew_within_60s` — exp 30s past → still valid.
- `test_verify_rejects_beyond_skew` — exp 5min past → raises.
- `test_has_perm_implies_read_from_write` — write-level token allowed for read-only route.
- `test_has_perm_admin_implies_all` — admin-level token allowed everywhere.
- `test_has_perm_bundle_isolation` — token for bundle alex can't access bobbi.
- `test_rs256_roundtrip_with_key_pair` — encode with private key, verify with public.
- `test_generate_secret_is_high_entropy` — ≥32 bytes urlsafe-b64.

**Step 2:** implement:
```python
from dataclasses import dataclass
from typing import Literal

Perm = Literal["read", "write", "admin"]
PERM_HIERARCHY = {"read": 0, "write": 1, "admin": 2}

@dataclass(frozen=True)
class Principal:
    sub: str
    bundles: dict[str, list[Perm]]
    jti: str | None = None

    def has_perm(self, bundle: str | None, required: Perm) -> bool: ...

def issue_token(
    *, sub: str, bundles: dict[str, list[Perm]],
    expires_in: timedelta, alg: str = "HS256",
    secret: str | None = None, private_key_pem: bytes | None = None,
) -> str: ...

def verify_token(
    token: str, *, alg: str = "HS256",
    secret: str | None = None, public_key_pem: bytes | None = None,
    leeway: int = 60,
) -> Principal: ...

def generate_secret() -> str: ...
```
`has_perm` logic: `None` bundle = route doesn't need bundle scoping (e.g., `/status`); token must have the required perm on ANY bundle OR admin. Else token must have required perm on the specific bundle.

**Step 3:** commit `feat(auth): PyJWT core — issue/verify/generate_secret + Principal`.

### Task 2: CLI — `soma auth issue/verify/rotate-secret`

**Files:**
- Modify: `src/soma/cli.py`
- Test: `tests/test_cli.py` — extend

**Step 1:** failing tests:
- `test_cli_auth_issue_prints_token` — `soma auth issue --sub X --bundle alex:read,write --expires 30d`; stdout is a valid JWT.
- `test_cli_auth_issue_refuses_without_secret` — `SOMA_JWT_SECRET` unset → error with clear message.
- `test_cli_auth_verify_prints_claims` — given a valid token, prints JSON claims and exits 0.
- `test_cli_auth_verify_bad_token_exits_nonzero` — tampered token → exit 2.
- `test_cli_auth_rotate_secret_prints_high_entropy_secret` — ≥32 bytes urlsafe-b64.
- `test_cli_auth_issue_supports_multiple_bundles` — repeatable `--bundle`.

**Step 2:** add `p_auth = sub.add_parser("auth", ...)` with nested subparsers. Parse `--bundle NAME:PERMS` CSV into `{name: [perm, ...]}`. Parse `--expires` as `30d | 7d | 24h | 60m` → timedelta.

**Step 3:** commit `feat(cli): soma auth issue/verify/rotate-secret`.

### Task 3: Replace `require_api_key` with `require_auth` in serve.py

**Files:**
- Modify: `src/soma/serve.py` — new dependency factory, route decorators
- Test: `tests/test_serve/test_jwt_auth.py`

**Step 1:** failing tests:
- `test_no_token_and_no_api_key_set_still_open` — backward compat when neither env set.
- `test_jwt_valid_read_token_hits_retrieve` — 200.
- `test_jwt_valid_read_token_rejected_on_store` — 403 with clear detail.
- `test_jwt_bundle_mismatch_rejected` — token for alex can't POST to /bundles/bobbi/store.
- `test_jwt_admin_token_reaches_everything` — 200 on all routes.
- `test_expired_token_401` — 401 with WWW-Authenticate header.
- `test_legacy_api_key_still_works_with_deprecation_header` — `SOMA_API_KEY=...` + client bearer matching → 200 + `X-SOMA-Deprecated: use JWT` header in response.
- `test_rs256_mode_verifies_with_public_key` — `SOMA_JWT_ALG=RS256` + `SOMA_JWT_PUBLIC_KEY_PATH` set; signed with private key; 200.

**Step 2:** new dependency factory:
```python
def require_auth(bundle_path_param: str | None, perm: Perm):
    async def dep(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    ) -> Principal | None:
        # Path 1: both env vars unset → open
        if not API_KEY and not JWT_SECRET and not JWT_PUBLIC_KEY:
            return None
        # Path 2: JWT
        if credentials and credentials.scheme.lower() == "bearer":
            try:
                principal = verify_token(credentials.credentials, ...)
                bundle = request.path_params.get(bundle_path_param) if bundle_path_param else None
                if not principal.has_perm(bundle, perm):
                    raise HTTPException(403, detail=f"perm {perm} required for bundle {bundle}")
                return principal
            except jwt.InvalidTokenError as exc:
                # Fall through to legacy API key check
                ...
        # Path 3: legacy SOMA_API_KEY (admin escape hatch)
        if API_KEY and credentials and credentials.credentials == API_KEY:
            response.headers["X-SOMA-Deprecated"] = "use JWT"
            return Principal(sub="legacy", bundles={}, jti=None)  # implicit admin
        raise HTTPException(401, detail="invalid or missing Authorization header",
                           headers={"WWW-Authenticate": "Bearer"})
    return dep
```
Wire it on every protected route with correct `bundle_path_param` and `perm`:
- read perm: `/status`, `/get/{id}`, `/recent`, `/related/{id}`, `/retrieve` (and `/bundles/*` twins)
- write perm: `/store`, `/store_batch`, `/forget`, `/consolidate`, `/save` (and `/bundles/*` twins)

**Step 3:** commit `feat(serve): JWT auth with per-bundle perm claims; SOMA_API_KEY stays as deprecated admin`.

### Task 4: OpenAPI spec carries security requirements

**Files:**
- Modify: `src/soma/serve.py` — wire security scheme metadata
- Test: `tests/test_serve/test_schema_polish.py` — extend

**Step 1:** failing test:
- `test_openapi_security_scheme_is_bearer_jwt` — spec has `securitySchemes.bearerAuth.type=http`, `scheme=bearer`, `bearerFormat=JWT`.
- `test_protected_routes_reference_security_scheme` — `/store` operation has `security: [{bearerAuth: []}]`.

**Step 2:** in the FastAPI app setup, ensure `HTTPBearer(bearerFormat="JWT", scheme_name="bearerAuth")`. Apply `security` to every protected route (FastAPI does this automatically when `Depends(require_auth(...))` is on the route).

**Step 3:** commit `feat(serve): OpenAPI advertises Bearer JWT scheme per route`.

### Task 5: Soma auth failure metric (Phase 3 integration)

**Files:**
- Modify: `src/soma/metrics.py` — add `soma_auth_failures_total{reason}` counter
- Modify: `src/soma/serve.py` — increment on each 401/403

**Step 1:** failing test:
- `test_auth_failure_increments_counter` — bad token hits /store; `soma_auth_failures_total{reason="invalid_token"}` ≥ 1.
- `test_perm_failure_distinct_reason` — token lacks write perm; counter with `reason="insufficient_perm"` increments.

**Step 2:** wire via a small helper in `serve.py`: `record_auth_failure(reason)` called before `raise HTTPException(...)`.

**Step 3:** commit `feat(metrics): soma_auth_failures_total with reason label`.

### Task 6: Docs + docker-compose.yml

**Files:**
- Create: `docs/auth.md`
- Modify: `docs/cookbook.md` — update curl examples to show JWT issuance
- Modify: `docker-compose.yml` — add `SOMA_JWT_SECRET`, `SOMA_JWT_ALG` env block
- Modify: `CHANGELOG.md`

**Step 1:** docs cover: quickstart (generate secret → issue token → use), HS256 vs RS256 tradeoffs, claim shape reference, deprecation timeline for `SOMA_API_KEY`, `soma auth` CLI reference, operational notes (rotate secret = invalidate all tokens).

**Step 2:** commit `docs: Phase 4 — JWT auth reference + CLI examples`.

---

## Risks

1. **JWT library supply chain.** PyJWT has a good track record. Pin to `>=2.8` (2024+ fixes for algorithm-confusion); rely on the bundled `cryptography` for RS256.
2. **Clock skew on Windows/containers.** 60s leeway covers typical NTP drift. Document the trade-off; operators can tune via `SOMA_JWT_LEEWAY`.
3. **`alg=none` regression risk.** PyJWT rejects by default but assert in code — belt and suspenders.
4. **Key leakage in logs.** `require_auth` must never log the raw Authorization header; the JSON retrieve log line (Phase 3) already omits it. Audit in review.
5. **Backward-compat test coverage.** Three env-var combos: neither set (open), only `SOMA_API_KEY` (legacy), only `SOMA_JWT_SECRET` (modern). Test all three. Both set → JWT path wins; legacy falls through as admin escape.
6. **OpenAPI exposure of auth scheme.** Schema now advertises auth → CI schema-drift check (Phase 5) picks up the change cleanly.

## Punted

- **Revocation blocklist** (needs persistent state).
- **Refresh-token endpoint** (OAuth-flow complexity; rotate by re-issuing instead).
- **Per-token rate limiting** (Redis or in-proc counters; use reverse proxy for now).
- **Hashed-token store** (same — operator problem, not library's).

## Related plans

- Phase 3 — `docs/plans/2026-04-16-phase-3-observability.md` (Task 5 here adds `soma_auth_failures_total`)
- Phase 5 — `docs/plans/2026-04-16-phase-5-typescript-client.md` (TS codegen consumes the updated OpenAPI spec with bearer security)
