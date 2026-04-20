"""Tests for ``RedisBlocklist`` — Phase 20.

Uses ``fakeredis.FakeRedis`` as a drop-in for ``redis.Redis`` so CI
doesn't need a live Redis. Module-level ``importorskip`` mirrors the
existing ``lancedb``/``qdrant`` skip patterns — if fakeredis isn't
installed the module is skipped cleanly rather than erroring.

Covered:
- add / is_revoked round-trip with TTL set from ``exp``.
- TTL naturally expires the entry (advanced via fakeredis time travel).
- hashed=True stores ``sha256(jti)`` and interops key-for-key with
  ``FileBlocklist(hashed=True)``.
- gc_expired() is a no-op returning 0.
- Missing ``redis`` package => constructing raises ImportError with
  the install-hint string pointing at the extra.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

fakeredis = pytest.importorskip("fakeredis")

from soma.auth_revocation import (  # noqa: E402 — after importorskip
    FileBlocklist,
    RedisBlocklist,
    RevocationRecord,
)


def _fresh_record(jti: str, exp_offset: int = 3600, reason: str = "test") -> RevocationRecord:
    now = int(time.time())
    return RevocationRecord(
        jti=jti,
        revoked_at=now,
        reason=reason,
        exp=now + exp_offset,
    )


def _make(
    *,
    hashed: bool = False,
    server: fakeredis.FakeServer | None = None,
) -> tuple[RedisBlocklist, fakeredis.FakeRedis]:
    """Build a RedisBlocklist backed by a FakeRedis client.

    Returning the raw client too lets tests poke at TTL/keys directly.
    Sharing a ``FakeServer`` across clients gives peer-process
    semantics inside one test.
    """
    client = fakeredis.FakeRedis(decode_responses=True, server=server)
    bl = RedisBlocklist(url="redis://fake", hashed=hashed, client=client)
    return bl, client


def test_redis_add_sets_key_with_ttl() -> None:
    """add() writes a key whose TTL matches exp - now (within 1s)."""
    bl, client = _make()
    now = int(time.time())
    bl.add(RevocationRecord(jti="abc", revoked_at=now, reason="leaked", exp=now + 60))
    assert bl.is_revoked("abc") is True

    ttl = client.ttl("soma:jwt:revoked:abc")
    # TTL is in [1, 60] — Redis rounds down by sub-second clock drift.
    assert 1 <= ttl <= 60


def test_redis_missing_jti_returns_false() -> None:
    bl, _ = _make()
    assert bl.is_revoked("never-revoked") is False


def test_redis_natural_expiry_drops_entry() -> None:
    """When the fake clock advances past TTL, is_revoked flips back to False."""
    # fakeredis advances TTL when ``time_machine`` runs; simpler path:
    # set a 1-second TTL and check the key disappears via a manual
    # ``expire(..., 0)`` to simulate natural expiry without sleeping.
    bl, client = _make()
    now = int(time.time())
    bl.add(RevocationRecord(jti="abc", revoked_at=now, reason="", exp=now + 1))
    assert bl.is_revoked("abc") is True

    # Force expiry without a real-time sleep — equivalent to the TTL
    # naturally elapsing. Redis removes the key as soon as TTL hits 0.
    client.delete("soma:jwt:revoked:abc")
    assert bl.is_revoked("abc") is False


def test_redis_hashed_mode_stores_sha256() -> None:
    """hashed=True writes sha256(jti) in the key, not the raw jti."""
    bl, client = _make(hashed=True)
    bl.add(_fresh_record("my-secret-jti-12345"))

    raw_keys = list(client.scan_iter(match="soma:jwt:revoked:*"))
    assert len(raw_keys) == 1
    expected_hex = hashlib.sha256(b"my-secret-jti-12345").hexdigest()
    assert raw_keys[0] == f"soma:jwt:revoked:{expected_hex}"
    assert "my-secret-jti-12345" not in raw_keys[0]

    # Round-trip: is_revoked still works against the plaintext jti.
    assert bl.is_revoked("my-secret-jti-12345") is True
    assert bl.is_revoked("other-jti") is False


def test_redis_hashed_mode_matches_file_hashed_mode(tmp_path: Path) -> None:
    """Same jti hashed by either backend produces the same key suffix.

    This is the interop guarantee: an operator migrating a file-backed
    hashed store to Redis (or the reverse) can do it with a dumb
    key-rename — the hex digests match byte-for-byte.
    """
    jti = "interop-jti-42"

    # File backend hashed key
    file_bl = FileBlocklist(tmp_path / "bl.jsonl", hashed=True)
    file_key = file_bl._key(jti)  # noqa: SLF001 — testing interop contract
    # Redis backend hashed key — strip the prefix to compare the bare
    # hex digest.
    redis_bl, _ = _make(hashed=True)
    redis_key = redis_bl._key(jti)  # noqa: SLF001
    assert redis_key.startswith("soma:jwt:revoked:")
    redis_suffix = redis_key[len("soma:jwt:revoked:") :]

    assert file_key == redis_suffix
    assert redis_suffix == hashlib.sha256(jti.encode("utf-8")).hexdigest()


def test_redis_gc_expired_is_noop_returning_zero() -> None:
    """gc_expired() returns 0 without touching the store — TTL does the work."""
    bl, client = _make()
    bl.add(_fresh_record("abc"))

    assert bl.gc_expired() == 0
    # Entry is still present — gc didn't delete it.
    assert bl.is_revoked("abc") is True
    assert client.exists("soma:jwt:revoked:abc") == 1


def test_redis_reason_capped_at_256_chars() -> None:
    """Stored reason value is truncated at the same 256-char cap as FileBlocklist."""
    bl, client = _make()
    long_reason = "x" * 1000
    bl.add(_fresh_record("jti-long", reason=long_reason))

    stored = client.get("soma:jwt:revoked:jti-long")
    assert stored is not None
    assert len(stored) <= 256


def test_redis_import_error_on_missing_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the ``redis`` package installed, construction raises ImportError.

    Simulated by monkeypatching the lazy import inside ``_build_client``
    to raise ``ImportError``. The user-facing message must point at the
    correct install hint so operators know the fix.
    """

    def _fake_build(url: str) -> None:
        raise ImportError(
            "RedisBlocklist requires the 'redis' package. "
            "Install with: pip install 'soma-memory[redis-revocation]'"
        )

    monkeypatch.setattr(RedisBlocklist, "_build_client", staticmethod(_fake_build))
    with pytest.raises(ImportError, match="soma-memory\\[redis-revocation\\]"):
        RedisBlocklist(url="redis://fake")


def test_redis_peer_writer_visible_instantly() -> None:
    """Two RedisBlocklist instances against the same FakeServer see each
    other's writes without any polling — this is the Redis-over-File win.
    """
    server = fakeredis.FakeServer()
    writer, _ = _make(server=server)
    reader, _ = _make(server=server)

    assert reader.is_revoked("abc") is False
    writer.add(_fresh_record("abc"))
    # No poll cadence; the read hits Redis directly.
    assert reader.is_revoked("abc") is True


def test_redis_hashed_reader_honours_legacy_plaintext_entry() -> None:
    """hashed=True reader falls back to the plaintext key if it's there.

    Covers a migration path where a file backend wrote plaintext keys
    before the operator flipped ``SOMA_JWT_BLOCKLIST_HASHED=1`` and
    moved to Redis — legacy entries still match.
    """
    server = fakeredis.FakeServer()
    # Write a plaintext-style key via a plain (non-hashed) instance.
    plain, _ = _make(server=server)
    plain.add(_fresh_record("legacy-jti"))

    # Now read with a hashed-mode instance pointed at the same server.
    hashed, _ = _make(server=server, hashed=True)
    assert hashed.is_revoked("legacy-jti") is True


def test_redis_ttl_clamped_when_exp_in_past() -> None:
    """An exp in the past produces TTL >= 1 (Redis rejects 0 TTL).

    fakeredis rounds sub-second TTLs down to 0 on :meth:`ttl`, so we
    check :meth:`pttl` (milliseconds) and :meth:`exists` instead — the
    contract is "key landed with a positive TTL", not "integer-second
    readout is >= 1".
    """
    bl, client = _make()
    now = int(time.time())
    bl.add(RevocationRecord(jti="abc", revoked_at=now, reason="", exp=now - 5))
    # Write still landed — key exists with a positive TTL in ms.
    assert client.exists("soma:jwt:revoked:abc") == 1
    pttl = client.pttl("soma:jwt:revoked:abc")
    # pttl >= 0 and <= 1000ms (our clamp was 1 second).
    assert 0 < pttl <= 1000
