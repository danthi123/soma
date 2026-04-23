# SOMA Auth — Per-bundle JWTs

The SOMA REST API ships with three auth modes:

1. **Open** (default for local dev) — no env vars set, every route is
   reachable without a token.
2. **JWT** — `SOMA_JWT_SECRET` (HS256) or `SOMA_JWT_PUBLIC_KEY_PATH`
   (RS256) is set; bearer tokens are decoded and their per-bundle
   permission claims enforced.
3. **Legacy** — `SOMA_API_KEY` is set; a raw match against the bearer
   token unlocks the admin-everywhere escape hatch. Responses carry
   `X-SOMA-Deprecated: use JWT`.

JWT + legacy can coexist. Valid JWTs win; an invalid JWT falls through
to the legacy check before 401-ing.

---

## Quickstart (HS256)

```bash
# 1. Generate and export a shared secret (store it like any other
#    sensitive credential — a git-ignored .env or a secret manager).
export SOMA_JWT_SECRET=$(soma auth rotate-secret)

# 2. Issue a token for the 'alex' caller scoped to bundle 'alex' with
#    read+write for the next 30 days.
export TOKEN=$(
  soma auth issue \
    --sub alex \
    --bundle alex:read,write \
    --expires 30d
)

# 3. Start the server and call it with the token.
soma serve --port 8420 &

curl -X POST http://localhost:8420/bundles/alex/store \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "alex likes pho"}'
```

## Quickstart (RS256, issuer/verifier split)

Keep the private key on a single signing host; distribute only the
public key to each REST server.

```bash
# One-time keygen on the signing host:
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out jwt.key
openssl rsa -in jwt.key -pubout -out jwt.pub

# On the signer:
export SOMA_JWT_ALG=RS256
export SOMA_JWT_PRIVATE_KEY_PATH=/path/to/jwt.key
soma auth issue --sub alex --bundle alex:read,write --expires 7d

# On the REST server:
export SOMA_JWT_ALG=RS256
export SOMA_JWT_PUBLIC_KEY_PATH=/path/to/jwt.pub
soma serve --port 8420
```

---

## Claim shape

```json
{
  "iss": "soma",
  "sub": "<caller id>",
  "iat": 1713225600,
  "exp": 1715817600,
  "jti": "b8f1...-uuid4",
  "soma": {
    "v": 1,
    "bundles": {
      "alex": ["read", "write"],
      "ops": ["admin"]
    }
  }
}
```

Three permission tiers in strict hierarchy:

| Perm     | Implies                     |
|----------|-----------------------------|
| `read`   | -                           |
| `write`  | `read`                      |
| `admin`  | `write`, `read` (everything)|

A token with `{"alex": ["write"]}` can hit both `GET /bundles/alex/get/<id>`
and `POST /bundles/alex/store`, but NOT `POST /bundles/bobbi/store`.

Routes that are not tenant-scoped (e.g., `/status`, `/retrieve` on the
default bundle) require the claim to grant the perm on ANY bundle —
this keeps the token model uniform without special-casing "default".

---

## Route perm map

| Route                             | Perm    | Bundle source       |
|-----------------------------------|---------|---------------------|
| `GET /health`, `/version`, `/metrics` | none (public) | —              |
| `GET /status`                     | `read`  | any                 |
| `GET /get/{id}`, `/recent`, `/related/{id}` | `read` | any         |
| `POST /retrieve`                  | `read`  | any                 |
| `POST /store`, `/store_batch`, `/forget`, `/consolidate`, `/save` | `write` | any |
| `GET /bundles/{name}/<ro>`        | `read`  | `name`              |
| `POST /bundles/{name}/<rw>`       | `write` | `name`              |

`consolidate` mutates graph weights and is treated as `write`.

---

## CLI reference

### `soma auth issue`

```
soma auth issue \
  --sub    <caller id>                       \
  --bundle NAME:PERMS  [--bundle NAME:PERMS] \
  --expires 30d | 7d | 24h | 60m
```

`--bundle` is repeatable; PERMS is a comma-separated subset of
`read,write,admin`. Omitting `--bundle` mints a token with an empty
bundle map — useful as an identity proof without any SOMA grants
(rejected on all protected routes unless the operator still has
`SOMA_API_KEY` set as an escape hatch).

Env: `SOMA_JWT_SECRET` (HS256) or `SOMA_JWT_PRIVATE_KEY_PATH` (RS256)
plus `SOMA_JWT_ALG` if RS256.

### `soma auth verify`

```
soma auth verify --token <JWT>
```

Decodes the token, prints claims as indented JSON, exits 0 on
success. On failure exits 2 with the underlying PyJWT error message.
Uses `SOMA_JWT_SECRET` (HS256) or `SOMA_JWT_PUBLIC_KEY_PATH` (RS256).

### `soma auth rotate-secret`

```
soma auth rotate-secret > .soma.secret
```

Emits a fresh 32-byte urlsafe-b64 shared secret to stdout. Use it as
the new `SOMA_JWT_SECRET`. **Rotating invalidates every previously
issued token.** For single-token revocation without nuking the secret,
use the revocation blocklist (below).

### `soma auth revoke`

```
soma auth revoke --token <jwt> --reason "leaked on slack"
# or, if you only have the jti from an access log:
soma auth revoke --jti <id> --exp <epoch> --reason "suspected compromise"
```

Appends a record to `SOMA_JWT_BLOCKLIST_PATH`. The running server
picks it up on its next mtime poll (~30 s). See "Revocation" below.

### `soma auth list-revoked`

```
soma auth list-revoked
```

Prints one JSON record per live revocation (past-exp entries are
skipped — the JWT verifier rejects them on `exp` alone). Use for
audit.

### `soma auth gc`

```
soma auth gc
```

Rewrites the blocklist file dropping past-exp entries. Safe to cron
daily; writes via temp-file-replace so concurrent `soma auth revoke`
calls don't race.

---

## HS256 vs RS256

| Property              | HS256                         | RS256                                 |
|-----------------------|-------------------------------|---------------------------------------|
| Key shape             | shared secret                 | private/public key pair               |
| Issuer == verifier?   | must be                       | split OK                              |
| Rotation cost         | re-issue every token          | re-issue every token (no blocklist)   |
| Dep weight            | PyJWT only                    | PyJWT + `cryptography`                |
| Recommended for       | single-process / docker-compose | multi-service / team deployments    |

Both modes use the same claim shape. Switching is an env-var change,
not a wire-format change.

---

## Operational notes

- **Clock skew.** `verify_token` applies a 60 s leeway by default.
  Override with `SOMA_JWT_LEEWAY=<seconds>` if your NTP drift exceeds
  that.
- **`alg=none` rejection.** PyJWT 2.x already rejects it, and we
  belt-and-suspenders the header check in `soma.auth.verify_token`.
- **No logging of raw tokens.** `require_auth` never logs the
  `Authorization` header. Auth failures increment
  `soma_auth_failures_total{reason}` for observability — grep the
  `reason` label, not the token.
- **`/metrics`, `/health`, `/version` stay public.** Liveness and
  monitoring must not depend on the auth envelope.
- **Secret-rotation workflow.** `soma auth rotate-secret > .new` →
  restart the server with `SOMA_JWT_SECRET=$(cat .new)` →
  re-issue active tokens with the new secret. Operators can script
  this via any secret manager (Vault, 1Password, AWS SM).

---

## Revocation

Phase 4 auto-populated a `jti` (UUID4) on every issued token so a
future revocation blocklist could key off it. That blocklist shipped
as a file-backed JSONL store.

### Enable it

```bash
export SOMA_JWT_BLOCKLIST_PATH=./data/jwt-blocklist.jsonl
soma serve --port 8420            # server reads the file on startup + polls mtime
```

Without the env var, nothing changes — the revocation check is a
no-op and the server behaves exactly like pre-revocation Phase 4.

### Revoke a token

```bash
# Full token (server pulls jti + exp from signed claims):
soma auth revoke --token "$LEAKED" --reason "leaked on slack 2026-04-16"

# Just the jti (from an access log), with explicit exp:
soma auth revoke --jti 3a8b-... --exp 1715817600 --reason "suspected compromise"
```

### POST /auth/revoke

Revoke a token over HTTP. Same semantics as ``soma auth revoke`` —
writes a ``RevocationRecord`` to the blocklist keyed off
``SOMA_JWT_BLOCKLIST_PATH`` (or ``SOMA_JWT_BLOCKLIST_REDIS_URL``).

**Request (by token):**

```json
{"token": "<full JWT>", "reason": "rotated"}
```

**Request (by jti):**

```json
{"jti": "<uuid>", "exp": 9999999999, "reason": "rotated"}
```

**Response (200):**

```json
{"revoked": "<jti>", "exp": 9999999999, "reason": "<echo>"}
```

Requires ``admin`` scope on any bundle in the authenticated token's
claim. 401 if unauthenticated, 403 if the caller lacks admin scope,
400 if neither ``token`` nor ``jti``+``exp`` is supplied, **503 if
the server was booted without** ``SOMA_JWT_BLOCKLIST_PATH`` (or
``SOMA_JWT_BLOCKLIST_REDIS_URL``) — the feature is implemented but
revocation can't persist; set the env and restart.

Propagation:

- Same process issuing the revoke: instant (in-memory cache updated
  at write time).
- Other processes on the same host (uvicorn worker, sibling `soma
  serve` instance): up to 30 s (mtime poll cadence).
- Token verification: the next request made by the revoked token
  returns HTTP 401 `{"detail": "token revoked"}` with the
  `soma_auth_failures_total{reason="revoked_token"}` counter
  advancing.

### Inspect / audit

```bash
soma auth list-revoked     # one JSON per live entry
```

Past-exp entries are hidden because the JWT verifier rejects them on
`exp` alone — they'd clutter the audit view without adding
information.

### GC cadence

The file grows append-only. Run `soma auth gc` on a cron or systemd
timer to drop past-exp entries — once a day is ample for most
workloads. The GC rewrites the file atomically via temp + replace
under the same sibling-file lock that revocations take, so it's safe
alongside active writers.

```bash
# /etc/cron.daily/soma-gc
#!/bin/sh
env SOMA_JWT_BLOCKLIST_PATH=/var/lib/soma/jwt-blocklist.jsonl soma auth gc
```

### Redis backend (multi-host / k8s)

The file backend coordinates peers via mtime polling (~30 s cadence).
That's fine on a single host but wrong for k8s replicas or geographically
split workers — you need a shared store with instant propagation.

Install the optional extra and point the env var at your Redis:

```bash
pip install 'soma-memory[redis-revocation]'
export SOMA_JWT_BLOCKLIST_REDIS_URL=redis://redis.internal:6379/0
# SOMA_JWT_BLOCKLIST_PATH is ignored when REDIS_URL is set (warning logged).
# SOMA_JWT_BLOCKLIST_HASHED=1 still works — same sha256(jti) key scheme
# as the file backend, so migrations are a straight re-key.
soma serve --port 8420
```

Each revocation becomes one Redis key
(`soma:jwt:revoked:<jti_or_hash>`) with a TTL set to `exp - now` in
seconds. Redis auto-expires entries, so `soma auth gc` is a no-op
against the Redis backend — there's nothing to clean up. Propagation
across workers is instant: the next `is_revoked` check on any replica
hits Redis directly, no poll.

Minimal docker-compose for local dev:

```yaml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  soma:
    image: soma:latest
    environment:
      SOMA_JWT_SECRET: "${SOMA_JWT_SECRET}"
      SOMA_JWT_BLOCKLIST_REDIS_URL: "redis://redis:6379/0"
      SOMA_JWT_BLOCKLIST_HASHED: "1"
    depends_on: [redis]
    ports: ["8420:8420"]
```

**When to pick which:**

| Deployment shape                    | Backend |
|-------------------------------------|---------|
| Single host, one or more uvicorn workers | File |
| Multi-host, sticky-session load balancer | File (shared NFS path) |
| k8s deployment, 2+ pods, no sticky sessions | Redis |
| Horizontal auto-scaling workers     | Redis |
| File share with sketchy locking semantics | Redis |

### Operational notes

- **Single source of truth.** One file per deployment. Point every
  `soma serve` worker at the same path; they coordinate via
  portalocker + mtime poll.
- **No in-memory-only mode.** Revocations always persist. An
  in-memory list that disappears on restart would be a footgun.
- **Redis backend ships as `soma[redis-revocation]`.** For multi-host
  / k8s deploys where the 30 s file-poll lag is unacceptable, flip to
  the Redis backend (see below). Instant cross-worker propagation and
  automatic TTL — no GC cron needed.
- **Field limits.** `reason` is capped at 256 chars at the store
  boundary; longer strings are truncated. Keep reasons concise —
  they're an operator note, not an incident report.
- **Reading the file by hand.** JSONL, one record per line. Safe to
  `jq` / `grep` for forensics:

    ```bash
    jq -r 'select(.reason | test("slack"))' /path/to/jwt-blocklist.jsonl
    ```

---

## Deprecation of `SOMA_API_KEY`

`SOMA_API_KEY` still works so existing compose files keep running, but
every response it unlocks carries:

```
X-SOMA-Deprecated: use JWT
```

Plan to remove support after the next major release. Migrate by:

1. `export SOMA_JWT_SECRET=$(soma auth rotate-secret)`
2. Issue tokens for each caller with the right bundle scope.
3. Remove `SOMA_API_KEY` from the server env.

---

## Metrics

Auth failures are counted at `soma_auth_failures_total{reason=...}`:

| `reason`              | When                                                     |
|-----------------------|----------------------------------------------------------|
| `missing_credentials` | no `Authorization` header + not in open mode             |
| `invalid_token`       | bearer token fails signature / format / wrong secret     |
| `expired_token`       | token past `exp + leeway`                                |
| `insufficient_perm`   | valid token but lacks the perm for the requested route   |
| `revoked_token`       | token's `jti` is in `SOMA_JWT_BLOCKLIST_PATH`            |

See `docs/observability.md` for the full metric catalogue.

---

## Rate limiting

`soma serve` ships an optional in-process rate limiter (Phase 26) for
single-binary deploys that don't front the server with a real reverse
proxy. It's keyed on the JWT's `jti` (falling back to `sub` for old
tokens that predate Phase 18), evicts idle buckets after 5 minutes,
and rejects with `HTTP 429` + a `Retry-After` header when the bucket
is drained.

**This is not a WAF.** A single worker can still be saturated by
unauthenticated traffic, and there's no coordination across workers
or hosts. For production, terminate at nginx / Caddy / Cloudflare and
let them do the rate limiting. The in-proc limiter is a cheap abuse
ceiling for:

- Local dev where you want to catch a runaway client loop.
- Small-scale homelab deploys behind a Tailscale / VPN.
- Single-tenant personal memory servers.

### Env vars

| Variable                 | Default      | Meaning                                                                                                  |
|--------------------------|--------------|----------------------------------------------------------------------------------------------------------|
| `SOMA_RATE_LIMIT_RPS`    | *unset*      | Steady-state requests per second per key (float). Unset = limiter disabled, every request flows through. |
| `SOMA_RATE_LIMIT_BURST`  | `ceil(rps)`  | Max tokens in each bucket (int). Defaults to `max(1, ceil(rps))` when only RPS is set.                   |
| `SOMA_RATE_LIMIT_SCOPE`  | `per-token`  | `per-token` (default, keys on `jti`) or `per-subject` (shares a bucket across refreshes of the same `sub`). |

`/metrics` and `/health` are always exempt so Prometheus scrapes and
k8s liveness probes never get throttled.

### Scope trade-offs

- **`per-token`** — each refresh gets a fresh budget. Cheap revocation
  story: burn the noisy `jti` and the caller resets to a clean bucket
  on their next refresh. Recommended default.
- **`per-subject`** — the same `sub` shares a bucket across all its
  live tokens. Useful when you want refresh loops to be rate-limited
  rather than be a budget-reset mechanism. Trade-off: you can't throttle
  one token independently of its refresh chain.

### Metric

`soma_rate_limited_total{scope}` counts every 429 the limiter issues.
Labelled by `scope` only — there's no per-`jti` / per-`sub` label
because that's a cardinality explosion waiting to happen. The
actionable query is "total 429s / minute"; use access logs if you need
to find the offending caller.

### Clock safety

Accounting uses `time.monotonic()` so NTP slew, DST, and manual clock
rewinds can't corrupt the bucket state. Tests inject `now=` to step
time deterministically without `sleep`.
