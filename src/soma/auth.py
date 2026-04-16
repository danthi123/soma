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
