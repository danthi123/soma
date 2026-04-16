"""Core PyJWT-backed auth module tests.

Phase 4 Task 1 — covers issue_token / verify_token / Principal /
has_perm, plus edge cases: missing exp, alg=none, clock skew,
RS256 round-trip, secret entropy.

All tests use fresh, ephemeral secrets so they are order-independent
and do not pollute the environment.
"""

from __future__ import annotations

import base64
from datetime import timedelta

import jwt
import pytest

from soma.auth import (
    PERM_HIERARCHY,
    Principal,
    generate_secret,
    issue_token,
    parse_ttl_spec,
    refresh_token,
    verify_token,
)

SECRET = "test-secret-not-used-outside-this-file"


def test_issue_and_verify_hs256_roundtrip() -> None:
    token = issue_token(
        sub="alex",
        bundles={"alex": ["read", "write"]},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
    )
    principal = verify_token(token, secret=SECRET)
    assert principal.sub == "alex"
    assert principal.bundles == {"alex": ["read", "write"]}
    # jti should be auto-populated for future revocation
    assert principal.jti is not None


def test_verify_rejects_missing_exp() -> None:
    # Craft a token with no exp claim by calling PyJWT directly.
    import jwt

    token = jwt.encode({"sub": "x"}, SECRET, algorithm="HS256")
    with pytest.raises(jwt.InvalidTokenError):
        verify_token(token, secret=SECRET)


def test_verify_rejects_alg_none() -> None:
    # alg=none must be rejected even if we try to force it.
    import jwt

    token = jwt.encode(
        {"sub": "x", "exp": 9999999999},
        key="",
        algorithm="none",
    )
    with pytest.raises(jwt.InvalidTokenError):
        verify_token(token, secret=SECRET)


def test_verify_accepts_clock_skew_within_60s() -> None:
    # Token expired 30 s ago — within 60 s leeway, still valid. A bare
    # `pytest.raises(Exception)` would trip B017; this test asserts the
    # positive path so no raises wrapper is needed.
    token = issue_token(
        sub="x",
        bundles={},
        expires_in=timedelta(seconds=-30),
        secret=SECRET,
    )
    principal = verify_token(token, secret=SECRET, leeway=60)
    assert principal.sub == "x"


def test_verify_rejects_beyond_skew() -> None:
    # Expired 5 min ago — beyond 60 s leeway.
    token = issue_token(
        sub="x",
        bundles={},
        expires_in=timedelta(minutes=-5),
        secret=SECRET,
    )
    with pytest.raises(jwt.InvalidTokenError):
        verify_token(token, secret=SECRET, leeway=60)


def test_has_perm_implies_read_from_write() -> None:
    p = Principal(sub="alex", bundles={"alex": ["write"]})
    assert p.has_perm("alex", "read") is True
    assert p.has_perm("alex", "write") is True
    assert p.has_perm("alex", "admin") is False


def test_has_perm_admin_implies_all() -> None:
    p = Principal(sub="op", bundles={"any": ["admin"]})
    assert p.has_perm("any", "read") is True
    assert p.has_perm("any", "write") is True
    assert p.has_perm("any", "admin") is True


def test_has_perm_bundle_isolation() -> None:
    p = Principal(sub="alex", bundles={"alex": ["read", "write"]})
    assert p.has_perm("alex", "read") is True
    assert p.has_perm("bobbi", "read") is False
    assert p.has_perm("bobbi", "write") is False


def test_has_perm_none_bundle_uses_any_claim() -> None:
    # None = route doesn't scope by bundle (e.g., /status). Any perm on
    # any bundle satisfies if it is >= required.
    p = Principal(sub="x", bundles={"alex": ["read"]})
    assert p.has_perm(None, "read") is True
    assert p.has_perm(None, "write") is False

    p2 = Principal(sub="x", bundles={"alex": ["admin"]})
    assert p2.has_perm(None, "write") is True
    assert p2.has_perm(None, "admin") is True


def test_perm_hierarchy_constants() -> None:
    assert PERM_HIERARCHY == {"read": 0, "write": 1, "admin": 2}


def test_rs256_roundtrip_with_key_pair() -> None:
    # Generate a short-lived RSA key pair in-process; encode with the
    # private key, verify with the public one.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    token = issue_token(
        sub="rs256-user",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=1),
        alg="RS256",
        private_key_pem=private_pem,
    )
    principal = verify_token(token, alg="RS256", public_key_pem=public_pem)
    assert principal.sub == "rs256-user"
    assert principal.bundles == {"alex": ["read"]}


def test_generate_secret_is_high_entropy() -> None:
    s = generate_secret()
    # urlsafe-b64 of at least 32 bytes -> at least 43 chars
    assert isinstance(s, str)
    assert len(s) >= 43
    # Must decode as urlsafe-b64 (with potential padding)
    padded = s + "=" * (-len(s) % 4)
    decoded = base64.urlsafe_b64decode(padded)
    assert len(decoded) >= 32
    # Two calls must differ (entropy check, not just length)
    assert generate_secret() != s


def test_verify_rejects_wrong_secret() -> None:
    token = issue_token(
        sub="x",
        bundles={},
        expires_in=timedelta(minutes=1),
        secret=SECRET,
    )
    with pytest.raises(jwt.InvalidTokenError):
        verify_token(token, secret="wrong-secret")


def test_issue_requires_either_secret_or_private_key() -> None:
    # HS256 needs secret; RS256 needs private_key_pem.
    with pytest.raises(ValueError):
        issue_token(
            sub="x",
            bundles={},
            expires_in=timedelta(minutes=1),
            alg="HS256",
            secret=None,
        )
    with pytest.raises(ValueError):
        issue_token(
            sub="x",
            bundles={},
            expires_in=timedelta(minutes=1),
            alg="RS256",
            private_key_pem=None,
        )


# ------------------------------------------------------------------
# Revocation — verify_token consults an optional BlocklistBackend
# ------------------------------------------------------------------
def test_verify_rejects_revoked_jti(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A revoked jti + non-null blocklist => InvalidTokenError('token revoked')."""
    import time

    from soma.auth_revocation import FileBlocklist, RevocationRecord

    token = issue_token(
        sub="alex",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
    )
    # Decode to pull the jti off the fresh token.
    principal = verify_token(token, secret=SECRET)
    assert principal.jti is not None

    bl = FileBlocklist(tmp_path / "bl.jsonl")
    now = int(time.time())
    bl.add(
        RevocationRecord(
            jti=principal.jti,
            revoked_at=now,
            reason="leaked in test",
            exp=now + 600,
        )
    )

    with pytest.raises(jwt.InvalidTokenError) as exc:
        verify_token(token, secret=SECRET, blocklist=bl)
    assert "revoked" in str(exc.value).lower()


def test_verify_accepts_unrevoked_jti(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Non-empty blocklist that doesn't list this jti => verify passes."""
    import time

    from soma.auth_revocation import FileBlocklist, RevocationRecord

    token = issue_token(
        sub="alex",
        bundles={"alex": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
    )
    bl = FileBlocklist(tmp_path / "bl.jsonl")
    now = int(time.time())
    # Populate with an *unrelated* jti to force a cache load.
    bl.add(
        RevocationRecord(
            jti="unrelated-jti-xyz",
            revoked_at=now,
            reason="smoke",
            exp=now + 600,
        )
    )

    principal = verify_token(token, secret=SECRET, blocklist=bl)
    assert principal.sub == "alex"


# ------------------------------------------------------------------
# Phase 18 — optional `aud` claim for multi-service fleets
# ------------------------------------------------------------------
def test_aud_round_trip() -> None:
    """issue with audience=X, verify with expected_audience=X => pass."""
    token = issue_token(
        sub="a",
        bundles={"a": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
        audience="svc-A",
    )
    principal = verify_token(token, secret=SECRET, expected_audience="svc-A")
    assert principal.sub == "a"


def test_aud_mismatch_raises() -> None:
    """Token aud=svc-A verified against expected_audience=svc-B => reject."""
    token = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
        audience="svc-A",
    )
    with pytest.raises(jwt.InvalidTokenError):
        verify_token(token, secret=SECRET, expected_audience="svc-B")


def test_aud_unset_on_token_fine_when_not_required() -> None:
    """A pre-Phase-18 token (no aud) still verifies when expected_audience=None.

    Backward-compat: operators who don't care about multi-service fleets
    should see zero behaviour change.
    """
    token = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
    )
    principal = verify_token(token, secret=SECRET)
    assert principal.sub == "a"


def test_aud_required_but_missing_raises() -> None:
    """Token with no aud + expected_audience set => InvalidTokenError.

    Pyjwt's default: MissingRequiredClaimError (subclass of
    InvalidTokenError) when the caller expects an audience the token
    didn't declare.
    """
    token = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
    )
    with pytest.raises(jwt.InvalidTokenError):
        verify_token(token, secret=SECRET, expected_audience="svc-A")


def test_aud_set_on_token_but_verifier_unset_still_passes() -> None:
    """Token with aud=svc-A + expected_audience=None => pass.

    Pyjwt documented behaviour: aud is only checked when the caller
    explicitly supplies audience=. A token carrying aud can therefore
    still verify against legacy single-service verifiers that never
    opt into the check.
    """
    token = issue_token(
        sub="a",
        bundles={"a": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
        audience="svc-A",
    )
    principal = verify_token(token, secret=SECRET)
    assert principal.sub == "a"


# ------------------------------------------------------------------
# Phase 23 — refresh_token helper + parse_ttl_spec grammar
# ------------------------------------------------------------------
def test_parse_ttl_spec_shapes() -> None:
    assert parse_ttl_spec("30d") == timedelta(days=30)
    assert parse_ttl_spec("24h") == timedelta(hours=24)
    assert parse_ttl_spec("60m") == timedelta(minutes=60)
    # Whitespace trimmed.
    assert parse_ttl_spec("  5m ") == timedelta(minutes=5)


def test_parse_ttl_spec_rejects_garbage() -> None:
    for bad in ("", "0d", "-1h", "7", "7x", "abc", "1.5h"):
        with pytest.raises(ValueError):
            parse_ttl_spec(bad)


def test_refresh_round_trip_preserves_claims() -> None:
    """sub, bundles, and (if present) aud carry over; jti must be fresh."""
    t1 = issue_token(
        sub="alex",
        bundles={"alex": ["read", "write"], "bobbi": ["read"]},
        expires_in=timedelta(hours=1),
        secret=SECRET,
    )
    t2 = refresh_token(t1, secret=SECRET)
    p1 = verify_token(t1, secret=SECRET)
    p2 = verify_token(t2, secret=SECRET)
    assert p1.sub == p2.sub == "alex"
    assert p1.bundles == p2.bundles == {"alex": ["read", "write"], "bobbi": ["read"]}
    # Fresh jti — pinned security property.
    assert p1.jti is not None
    assert p2.jti is not None
    assert p1.jti != p2.jti


def test_refresh_extends_exp() -> None:
    """The refreshed token's exp sits in the future relative to the old one.

    With ``new_expires_in=None`` the new window matches the original
    exp-iat delta, but since iat advances when we mint, new exp > old exp.
    """
    t1 = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(minutes=30),
        secret=SECRET,
    )
    old_claims = jwt.decode(t1, SECRET, algorithms=["HS256"])
    t2 = refresh_token(t1, secret=SECRET)
    new_claims = jwt.decode(t2, SECRET, algorithms=["HS256"])
    assert new_claims["exp"] >= old_claims["exp"]
    # The new iat is at or after the old iat — monotonic clock.
    assert new_claims["iat"] >= old_claims["iat"]


def test_refresh_rejects_expired_token() -> None:
    t = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(minutes=-10),
        secret=SECRET,
    )
    with pytest.raises(jwt.InvalidTokenError):
        refresh_token(t, secret=SECRET)


def test_refresh_rejects_revoked_token(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import time

    from soma.auth_revocation import FileBlocklist, RevocationRecord

    t = issue_token(
        sub="a",
        bundles={"a": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
    )
    principal = verify_token(t, secret=SECRET)
    assert principal.jti is not None
    bl = FileBlocklist(tmp_path / "bl.jsonl")
    now = int(time.time())
    bl.add(
        RevocationRecord(
            jti=principal.jti,
            revoked_at=now,
            reason="leaked",
            exp=now + 600,
        )
    )
    with pytest.raises(jwt.InvalidTokenError) as exc:
        refresh_token(t, secret=SECRET, blocklist=bl)
    assert "revoked" in str(exc.value).lower()


def test_refresh_preserves_audience_when_present() -> None:
    t1 = issue_token(
        sub="a",
        bundles={"a": ["read"]},
        expires_in=timedelta(minutes=5),
        secret=SECRET,
        audience="svc-A",
    )
    t2 = refresh_token(t1, secret=SECRET, expected_audience="svc-A")
    # The new token must verify against the same audience.
    principal = verify_token(t2, secret=SECRET, expected_audience="svc-A")
    assert principal.sub == "a"


def test_refresh_default_ttl_matches_original_window() -> None:
    """Without new_expires_in, refreshed exp-iat ~= original exp-iat."""
    orig_window = 3600  # 1 hour in seconds
    t1 = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(seconds=orig_window),
        secret=SECRET,
    )
    t2 = refresh_token(t1, secret=SECRET)
    c2 = jwt.decode(t2, SECRET, algorithms=["HS256"])
    new_window = int(c2["exp"]) - int(c2["iat"])
    # Allow 2s slack for clock granularity between mint steps.
    assert abs(new_window - orig_window) <= 2


def test_refresh_honors_new_expires_in_override() -> None:
    t1 = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(days=30),
        secret=SECRET,
    )
    t2 = refresh_token(t1, secret=SECRET, new_expires_in=timedelta(minutes=15))
    c2 = jwt.decode(t2, SECRET, algorithms=["HS256"])
    new_window = int(c2["exp"]) - int(c2["iat"])
    # 15 minutes ± 2s.
    assert abs(new_window - 900) <= 2


def test_refresh_honors_max_expires_in_cap() -> None:
    """max_expires_in truncates even when new_expires_in is larger."""
    t1 = issue_token(
        sub="a",
        bundles={},
        expires_in=timedelta(days=30),
        secret=SECRET,
    )
    t2 = refresh_token(
        t1,
        secret=SECRET,
        new_expires_in=timedelta(days=365),
        max_expires_in=timedelta(hours=1),
    )
    c2 = jwt.decode(t2, SECRET, algorithms=["HS256"])
    new_window = int(c2["exp"]) - int(c2["iat"])
    assert abs(new_window - 3600) <= 2


def test_verify_ignores_blocklist_for_tokens_without_jti() -> None:
    """Legacy tokens without a jti claim pass unaffected.

    Belt-and-suspenders — the blocklist is keyed on jti, so missing
    jti means nothing to look up. We deliberately don't fail-closed on
    missing jti because legacy callers (pre-Phase-4 + external issuers)
    may emit tokens without one.
    """
    # Craft a token with no jti claim via PyJWT directly (bypassing
    # issue_token, which always populates jti).
    import tempfile
    import time as _t
    from pathlib import Path as _P

    from soma.auth_revocation import FileBlocklist

    now = int(_t.time())
    raw = jwt.encode(
        {"iss": "soma", "sub": "no-jti", "iat": now, "exp": now + 300},
        SECRET,
        algorithm="HS256",
    )
    # Blocklist exists but doesn't contain anything matching — the
    # no-jti branch must still accept.
    with tempfile.TemporaryDirectory() as td:
        bl = FileBlocklist(_P(td) / "bl.jsonl")
        principal = verify_token(raw, secret=SECRET, blocklist=bl)
        assert principal.sub == "no-jti"
        assert principal.jti is None
