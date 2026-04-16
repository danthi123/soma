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
issued token** — no blocklist, no revocation list; re-issue what you
need.

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

See `docs/observability.md` for the full metric catalogue.
