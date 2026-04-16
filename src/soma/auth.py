"""JWT-backed auth for SOMA's REST API.

Shared between ``soma.serve`` (request-time verification) and
``soma.cli`` (``soma auth issue``). The module has no FastAPI imports,
so CLI usage stays light.

Claim shape (v1)::

    {
        "iss": "soma",
        "sub": "<caller id>",
        "iat": <epoch>,
        "exp": <epoch>,
        "jti": "<uuid4>",
        "soma": {
            "v": 1,
            "bundles": {
                "alex": ["read", "write"],
                "bobbi": ["read"]
            }
        }
    }

Three permission tiers in strict hierarchy::

    read  < write < admin

``write`` implies ``read``; ``admin`` implies everything.
``verify_token`` enforces the hierarchy so callers don't have to.

Operational notes:
- Rejects ``alg=none`` via explicit check on the unverified header in
  addition to PyJWT's default allow-list.
- ``exp`` is required; a token without one is rejected.
- ``leeway=60s`` default covers typical NTP drift. Operators can tune
  at call sites via ``SOMA_JWT_LEEWAY``.
- ``generate_secret`` returns 32 raw bytes urlsafe-b64 encoded — good
  enough entropy for HS256.
"""

from __future__ import annotations

import re
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import jwt

if TYPE_CHECKING:
    from soma.auth_revocation import BlocklistBackend

__all__ = [
    "PERM_HIERARCHY",
    "Perm",
    "Principal",
    "generate_secret",
    "issue_token",
    "parse_ttl_spec",
    "refresh_token",
    "verify_token",
]

Perm = Literal["read", "write", "admin"]
PERM_HIERARCHY: dict[str, int] = {"read": 0, "write": 1, "admin": 2}


@dataclass(frozen=True)
class Principal:
    """Verified caller identity + per-bundle permissions.

    ``sub`` is the opaque caller id the token was issued to.
    ``bundles`` maps bundle name to a list of permission strings.
    ``jti`` is the JWT id (UUID4) if present — kept so a future
    revocation blocklist can key off it without re-parsing the token.
    """

    sub: str
    bundles: dict[str, list[Perm]] = field(default_factory=dict)
    jti: str | None = None

    def has_perm(self, bundle: str | None, required: Perm) -> bool:
        """Check whether this principal holds ``required`` on ``bundle``.

        - ``bundle=None``: route is not bundle-scoped (e.g., ``/status``).
          ANY bundle entry with a perm >= required, OR an ``admin``
          anywhere, satisfies.
        - ``bundle="name"``: the claim must list ``name`` and grant a
          perm >= required. An ``admin`` on ``name`` always satisfies.
        """
        req_level = PERM_HIERARCHY[required]
        if bundle is None:
            # No bundle scoping — any claim that meets the bar counts.
            for perms in self.bundles.values():
                for p in perms:
                    if PERM_HIERARCHY.get(p, -1) >= req_level:
                        return True
            return False
        perms = self.bundles.get(bundle, [])
        return any(PERM_HIERARCHY.get(p, -1) >= req_level for p in perms)


def issue_token(
    *,
    sub: str,
    bundles: dict[str, list[Perm]],
    expires_in: timedelta,
    alg: str = "HS256",
    secret: str | None = None,
    private_key_pem: bytes | None = None,
    issuer: str = "soma",
    audience: str | None = None,
) -> str:
    """Mint a signed JWT with the SOMA claim envelope.

    HS256 uses ``secret``; RS256 uses ``private_key_pem`` (a PKCS8 PEM
    blob). Any other algorithm raises ``ValueError`` — we explicitly
    do not support ``none``.

    ``audience`` (optional) populates the standard ``aud`` claim. When
    set, verifiers that pass ``expected_audience=`` reject tokens whose
    ``aud`` doesn't match. Leaving it unset preserves Phase 4 tokens
    exactly (no ``aud`` claim emitted).
    """
    if alg == "HS256":
        if not secret:
            raise ValueError("HS256 requires secret= (non-empty shared key)")
        signing_key: str | bytes = secret
    elif alg == "RS256":
        if not private_key_pem:
            raise ValueError("RS256 requires private_key_pem= (PEM-encoded private key)")
        signing_key = private_key_pem
    else:
        raise ValueError(f"unsupported alg {alg!r}; use HS256 or RS256")

    now = datetime.now(tz=UTC)
    exp = now + expires_in
    claims: dict[str, object] = {
        "iss": issuer,
        "sub": sub,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": str(uuid.uuid4()),
        "soma": {"v": 1, "bundles": bundles},
    }
    if audience is not None:
        claims["aud"] = audience
    return jwt.encode(claims, signing_key, algorithm=alg)


def verify_token(
    token: str,
    *,
    alg: str = "HS256",
    secret: str | None = None,
    public_key_pem: bytes | None = None,
    leeway: int = 60,
    issuer: str | None = "soma",
    blocklist: BlocklistBackend | None = None,
    expected_audience: str | None = None,
) -> Principal:
    """Decode + validate a JWT; return a :class:`Principal`.

    Raises :class:`jwt.InvalidTokenError` (or a subclass) on any
    failure: bad signature, expired beyond leeway, missing ``exp``,
    ``alg=none``, wrong issuer, unknown algorithm. Callers in
    ``serve.py`` catch this single base class.

    ``leeway`` (seconds) covers NTP drift — a token expired a moment
    ago on the verifier's clock still decodes. Default 60 s.

    ``blocklist`` (optional) consults a :class:`BlocklistBackend` for
    the token's ``jti`` claim *after* the signature + exp checks pass.
    A revoked jti raises :class:`jwt.InvalidTokenError("token revoked")`.
    When ``blocklist=None`` (default), the check is skipped — existing
    callers in CLI / tests behave unchanged. Tokens without a ``jti``
    claim (legacy / externally-issued) always pass the blocklist gate
    since there's nothing to key on.

    ``expected_audience`` (optional) pins the verified ``aud`` claim to
    a specific service string. When set, pyjwt enforces the match and
    raises :class:`jwt.InvalidAudienceError` (a subclass of
    :class:`jwt.InvalidTokenError`) on mismatch or missing ``aud``.
    When unset (default), ``aud`` is not checked — tokens issued
    without it (pre-Phase-18 or single-server deployments) keep
    verifying unchanged, and tokens issued *with* ``aud`` still pass
    (pyjwt's documented default).
    """
    # Belt-and-suspenders: reject alg=none before PyJWT gets a shot.
    # PyJWT 2.x already rejects 'none' unless explicitly allowed, but
    # an unverified peek is cheap insurance.
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError:
        raise
    if str(header.get("alg", "")).lower() == "none":
        raise jwt.InvalidAlgorithmError("alg=none is not allowed")

    if alg == "HS256":
        if not secret:
            raise ValueError("HS256 verify requires secret=")
        verify_key: str | bytes = secret
    elif alg == "RS256":
        if not public_key_pem:
            raise ValueError("RS256 verify requires public_key_pem=")
        verify_key = public_key_pem
    else:
        raise ValueError(f"unsupported alg {alg!r}; use HS256 or RS256")

    # ``require`` forces exp to be present; ``verify_exp=True`` is default.
    # ``verify_aud`` is explicitly tied to whether the caller supplied
    # ``expected_audience``: pyjwt 2.x otherwise *rejects* tokens that
    # carry an ``aud`` claim when the verifier passes ``audience=None``
    # (InvalidAudienceError, "Invalid audience"). We want the opposite
    # default — tokens with ``aud`` should still verify against legacy
    # single-service callers who haven't opted into the check.
    options: dict[str, object] = {
        "require": ["exp"],
        "verify_exp": True,
        "verify_aud": expected_audience is not None,
    }
    claims = jwt.decode(
        token,
        verify_key,
        algorithms=[alg],
        options=options,
        leeway=leeway,
        issuer=issuer,
        audience=expected_audience,
    )
    soma_claims = claims.get("soma") or {}
    bundles_raw = soma_claims.get("bundles") if isinstance(soma_claims, dict) else None
    bundles: dict[str, list[Perm]] = {}
    if isinstance(bundles_raw, dict):
        for name, perms in bundles_raw.items():
            if isinstance(name, str) and isinstance(perms, list):
                bundles[name] = [p for p in perms if p in PERM_HIERARCHY]

    jti = claims.get("jti")

    # Revocation check runs last — after signature + exp + claim shape
    # pass. Legacy tokens without a jti claim aren't keyable in the
    # blocklist so we let them through (the broader fix there is to
    # re-issue with jti, not to fail-closed and break existing callers).
    if blocklist is not None and isinstance(jti, str) and blocklist.is_revoked(jti):
        raise jwt.InvalidTokenError("token revoked")

    return Principal(
        sub=str(claims.get("sub") or ""),
        bundles=bundles,
        jti=jti,
    )


def generate_secret() -> str:
    """Return a fresh 32-byte urlsafe-b64 shared secret for HS256."""
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------
# Phase 23 — refresh-token helper + shared TTL-spec parser.
# ---------------------------------------------------------------------
_TTL_SPEC_RE = re.compile(r"^(\d+)([dhm])$")


def parse_ttl_spec(spec: str) -> timedelta:
    """Parse ``30d | 24h | 60m`` shorthand into :class:`timedelta`.

    Lives on ``soma.auth`` so both the CLI (``soma auth issue --expires``)
    and the server (``SOMA_JWT_REFRESH_TTL`` / ``SOMA_JWT_MAX_TTL`` env
    vars) share one grammar. Raises :class:`ValueError` on any malformed
    spec — CLI callers surface it as exit=2, server callers trap it and
    fall back to the original token's window.
    """
    m = _TTL_SPEC_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"invalid TTL spec {spec!r}; expected NUMBER + unit (d|h|m), e.g. 30d"
        )
    n, unit = int(m.group(1)), m.group(2)
    if n <= 0:
        raise ValueError(f"TTL must be positive; got {spec!r}")
    if unit == "d":
        return timedelta(days=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(minutes=n)


def refresh_token(
    current_token: str,
    *,
    alg: str = "HS256",
    secret: str | None = None,
    private_key_pem: bytes | None = None,
    public_key_pem: bytes | None = None,
    leeway: int = 60,
    issuer: str | None = "soma",
    blocklist: BlocklistBackend | None = None,
    expected_audience: str | None = None,
    new_expires_in: timedelta | None = None,
    max_expires_in: timedelta | None = None,
) -> str:
    """Verify ``current_token`` and mint a fresh one with the same claims.

    Preserves ``sub``, per-bundle ``bundles``, and ``aud`` (when present
    on the original). Always allocates a **fresh** ``jti`` so a future
    revoke of the old id doesn't invalidate the new one — pinned in
    tests, deliberate security property.

    Raises the same exceptions as :func:`verify_token` on an expired,
    revoked, or otherwise invalid current token (subclasses of
    :class:`jwt.InvalidTokenError`). Callers catch the base class.

    Signing material lookup mirrors :func:`issue_token`:

    - HS256: ``secret`` is used for both verification and signing.
    - RS256: ``public_key_pem`` verifies the current token, and
      ``private_key_pem`` signs the new one. Servers that only hold the
      public key (read-only verification fleet) cannot call this — they
      must route refresh requests to the signing node instead.

    ``new_expires_in`` (optional) sets the new token's TTL. When unset
    (default), the new token reuses the original's ``exp - iat`` window
    so callers get the conservative "same lifetime, fresh exp" behaviour.

    ``max_expires_in`` (optional) caps the TTL — protects against runaway
    token lifetimes when an operator accidentally sets a huge
    ``SOMA_JWT_REFRESH_TTL``. Applied AFTER ``new_expires_in`` resolution.
    """
    # Step 1: verify the current token. Any InvalidTokenError subclass
    # (expired, revoked, bad signature, missing exp, wrong aud) bubbles
    # straight out — caller decides whether to translate to a 401.
    principal = verify_token(
        current_token,
        alg=alg,
        secret=secret,
        public_key_pem=public_key_pem if alg == "RS256" else None,
        leeway=leeway,
        issuer=issuer,
        blocklist=blocklist,
        expected_audience=expected_audience,
    )

    # Step 2: decode the signed claims again — this time to recover the
    # exp/iat window and any optional ``aud`` claim, neither of which is
    # surfaced on the Principal. Signature has already been checked by
    # verify_token above; a fresh decode with verify_signature=False is
    # the standard pyjwt idiom for "read extra claims from a trusted
    # token".
    raw_claims = jwt.decode(current_token, options={"verify_signature": False})
    original_exp = int(raw_claims.get("exp", 0))
    original_iat = int(raw_claims.get("iat", 0))
    original_window = original_exp - original_iat
    audience = raw_claims.get("aud")

    # Step 3: resolve new TTL. Default to the original window; override
    # if the caller passed new_expires_in; cap at max_expires_in.
    if new_expires_in is not None:
        new_ttl = new_expires_in
    elif original_window > 0:
        new_ttl = timedelta(seconds=original_window)
    else:
        # Degenerate token with exp <= iat (shouldn't happen — verify
        # already rejected exp-less tokens — but belt-and-suspenders).
        new_ttl = timedelta(minutes=5)
    if max_expires_in is not None and new_ttl > max_expires_in:
        new_ttl = max_expires_in

    # Step 4: mint. issue_token allocates a fresh jti automatically.
    return issue_token(
        sub=principal.sub,
        bundles=principal.bundles,
        expires_in=new_ttl,
        alg=alg,
        secret=secret,
        private_key_pem=private_key_pem,
        issuer=issuer if issuer is not None else "soma",
        audience=audience if isinstance(audience, str) else None,
    )
