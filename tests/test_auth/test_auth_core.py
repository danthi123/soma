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
