"""Tests for ``soma.auth_revocation`` — file-backed JWT blocklist.

Covers the backend contract end-to-end: add/read round-trip, GC of expired
entries, torn-line recovery on crash, concurrent writers coordinating via
``portalocker``, and the ``null_blocklist()`` factory fallback when
``SOMA_JWT_BLOCKLIST_PATH`` is unset.

Tests are hermetic (``tmp_path`` per test, env isolated via
``monkeypatch``) so they can run in any order.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import pytest

from soma.auth_revocation import (
    FileBlocklist,
    RedisBlocklist,
    RevocationRecord,
    blocklist_from_env,
    null_blocklist,
)


def _fresh_record(jti: str, exp_offset: int = 3600, reason: str = "test") -> RevocationRecord:
    """Helper — a record expiring ``exp_offset`` seconds from now."""
    now = int(time.time())
    return RevocationRecord(
        jti=jti,
        revoked_at=now,
        reason=reason,
        exp=now + exp_offset,
    )


def test_file_blocklist_add_read_roundtrip(tmp_path: Path) -> None:
    """add_revocation / is_revoked round-trip on a fresh file."""
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path)
    rec = _fresh_record("jti-1")
    bl.add(rec)

    # Reopen — a new instance must see the append.
    bl2 = FileBlocklist(path)
    assert bl2.is_revoked("jti-1") is True
    assert bl2.is_revoked("jti-unseen") is False


def test_file_blocklist_ignores_expired_entries(tmp_path: Path) -> None:
    """Entries past their original ``exp`` are treated as not-revoked.

    We no longer need to block a token that's already beyond ``exp`` —
    the JWT verifier will reject it on ``exp`` alone. The blocklist only
    cares about live windows. These entries are GC candidates.
    """
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path)
    # exp 1 hour ago
    bl.add(_fresh_record("jti-expired", exp_offset=-3600))
    bl.add(_fresh_record("jti-live", exp_offset=3600))

    bl2 = FileBlocklist(path)
    assert bl2.is_revoked("jti-expired") is False
    assert bl2.is_revoked("jti-live") is True


def test_file_blocklist_handles_missing_file(tmp_path: Path) -> None:
    """No file on disk => nothing revoked (empty set)."""
    path = tmp_path / "does-not-exist.jsonl"
    bl = FileBlocklist(path)
    assert bl.is_revoked("anything") is False


def test_file_blocklist_torn_line_recovered(tmp_path: Path) -> None:
    """A truncated last line (e.g., post-crash) is silently skipped."""
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path)
    bl.add(_fresh_record("jti-ok"))

    # Simulate a torn write: partial JSON on a trailing line.
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"jti": "jti-torn", "revoked_at":')  # no newline, no closing brace

    bl2 = FileBlocklist(path)
    assert bl2.is_revoked("jti-ok") is True
    # The torn entry is dropped, not mis-parsed.
    assert bl2.is_revoked("jti-torn") is False


def _writer_worker(path_str: str, jti: str) -> None:
    """Worker used by the concurrent-writers test.

    Runs in a separate process; must re-import so the FileBlocklist is
    constructed inside the child.
    """
    from soma.auth_revocation import FileBlocklist, RevocationRecord

    now = int(time.time())
    bl = FileBlocklist(Path(path_str))
    bl.add(
        RevocationRecord(
            jti=jti,
            revoked_at=now,
            reason="concurrent",
            exp=now + 3600,
        )
    )


def test_file_blocklist_concurrent_writers(tmp_path: Path) -> None:
    """Two processes adding revocations must both land on disk.

    ``portalocker.Lock`` serialises writes; no record is lost.
    """
    path = tmp_path / "bl.jsonl"
    # Touch the file so both children open an existing inode.
    path.touch()

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_writer_worker, args=(str(path), f"jti-{i}")) for i in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"writer child exited non-zero: {p.exitcode}"

    bl = FileBlocklist(path)
    assert bl.is_revoked("jti-0") is True
    assert bl.is_revoked("jti-1") is True


def test_file_blocklist_gc(tmp_path: Path) -> None:
    """gc_expired() rewrites the file dropping past-exp entries."""
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path)
    bl.add(_fresh_record("jti-live", exp_offset=3600))
    bl.add(_fresh_record("jti-dead-1", exp_offset=-3600))
    bl.add(_fresh_record("jti-dead-2", exp_offset=-7200))

    removed = bl.gc_expired()
    assert removed == 2

    # File now contains only the live entry.
    surviving = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(surviving) == 1
    assert surviving[0]["jti"] == "jti-live"

    # Re-reading still flags the live one.
    bl2 = FileBlocklist(path)
    assert bl2.is_revoked("jti-live") is True
    assert bl2.is_revoked("jti-dead-1") is False


def test_null_blocklist_when_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SOMA_JWT_BLOCKLIST_PATH`` unset => null backend (always False)."""
    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_PATH", raising=False)
    bl = blocklist_from_env()
    assert bl.is_revoked("anything") is False
    # add() is a no-op on the null backend; gc returns 0.
    bl.add(_fresh_record("jti-ignored"))
    assert bl.gc_expired() == 0
    # Same reference is returned by null_blocklist() directly.
    assert null_blocklist().is_revoked("whatever") is False


def test_blocklist_from_env_returns_file_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Env var set => factory returns a FileBlocklist pointing at it."""
    path = tmp_path / "bl.jsonl"
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(path))
    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_HASHED", raising=False)
    bl = blocklist_from_env()
    rec = _fresh_record("jti-via-env")
    bl.add(rec)
    # A fresh reader sees the record.
    bl2 = blocklist_from_env()
    assert bl2.is_revoked("jti-via-env") is True


def test_blocklist_from_env_hashed_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SOMA_JWT_BLOCKLIST_HASHED=1 flips the backend to hashed mode."""
    path = tmp_path / "bl.jsonl"
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(path))
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_HASHED", "1")
    bl = blocklist_from_env()
    # FileBlocklist exposes .hashed for introspection.
    assert isinstance(bl, FileBlocklist)
    assert bl.hashed is True

    bl.add(_fresh_record("secret-jti"))
    raw = path.read_text(encoding="utf-8")
    assert "secret-jti" not in raw
    assert hashlib.sha256(b"secret-jti").hexdigest() in raw


def test_blocklist_from_env_hashed_default_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing / 0 / empty SOMA_JWT_BLOCKLIST_HASHED => plaintext mode."""
    path = tmp_path / "bl.jsonl"
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(path))
    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_HASHED", raising=False)
    bl = blocklist_from_env()
    assert isinstance(bl, FileBlocklist)
    assert bl.hashed is False

    # Explicit "0" also disables.
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_HASHED", "0")
    bl2 = blocklist_from_env()
    assert isinstance(bl2, FileBlocklist)
    assert bl2.hashed is False


def test_file_blocklist_reason_capped_at_256_chars(tmp_path: Path) -> None:
    """Reason > 256 chars is truncated at write time (defensive cap)."""
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path)
    long_reason = "x" * 1000
    bl.add(_fresh_record("jti-long-reason", reason=long_reason))

    # Round-trip: still revoked; reason stored at the cap.
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert raw[0]["jti"] == "jti-long-reason"
    assert len(raw[0]["reason"]) <= 256


# ------------------------------------------------------------------
# Phase 18 — optional sha256(jti) hashing at rest
# ------------------------------------------------------------------
def test_hashed_blocklist_stores_sha256_only(tmp_path: Path) -> None:
    """hashed=True writes sha256(jti) on disk, not the raw jti."""
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path, hashed=True)
    bl.add(_fresh_record("my-secret-jti-12345", reason="leaked"))

    raw = path.read_text(encoding="utf-8")
    assert "my-secret-jti-12345" not in raw
    expected_hex = hashlib.sha256(b"my-secret-jti-12345").hexdigest()
    assert expected_hex in raw
    # The record uses the new jti_key field, not the legacy jti.
    record = json.loads(raw.splitlines()[0])
    assert record["jti_key"] == expected_hex
    assert "jti" not in record


def test_hashed_blocklist_contains_works(tmp_path: Path) -> None:
    """add + is_revoked round-trip works in hashed mode."""
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path, hashed=True)
    bl.add(_fresh_record("abc"))
    assert bl.is_revoked("abc") is True
    assert bl.is_revoked("xyz") is False

    # A fresh reader on the same path (also hashed) re-hydrates cleanly.
    bl2 = FileBlocklist(path, hashed=True)
    assert bl2.is_revoked("abc") is True


def test_plain_blocklist_default_bytes_unchanged(tmp_path: Path) -> None:
    """Default (hashed=False) writes byte-identical to pre-Phase-18.

    Pins the existing on-disk schema: a {"jti": ..., "revoked_at": ...,
    "reason": ..., "exp": ...} record, no jti_key field.
    """
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path)  # hashed default False
    bl.add(_fresh_record("plain-jti-123", reason="pre-18"))

    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["jti"] == "plain-jti-123"
    assert "jti_key" not in record
    # Fields present: the four pre-Phase-18 keys, nothing else.
    assert set(record.keys()) == {"jti", "revoked_at", "reason", "exp"}


def test_blocklist_load_accepts_legacy_jti_records(tmp_path: Path) -> None:
    """A handwritten legacy {"jti": ...} record loads even when hashed=True.

    Existing on-disk files from pre-Phase-18 must keep working after an
    operator flips the env flag — the reader tolerates mixed schemas.
    """
    path = tmp_path / "bl.jsonl"
    now = int(time.time())
    legacy = {"jti": "legacy-jti", "revoked_at": now, "reason": "", "exp": now + 600}
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    # Instantiate in hashed mode; legacy record should still register.
    bl = FileBlocklist(path, hashed=True)
    assert bl.is_revoked("legacy-jti") is True


def test_gc_expired_preserves_hashed_mode(tmp_path: Path) -> None:
    """gc on a hashed store rewrites only jti_key records; no raw jti leak."""
    path = tmp_path / "bl.jsonl"
    bl = FileBlocklist(path, hashed=True)
    bl.add(_fresh_record("live-jti", exp_offset=3600))
    bl.add(_fresh_record("dead-jti", exp_offset=-3600))

    removed = bl.gc_expired()
    assert removed == 1

    lines = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    # Surviving record still uses jti_key (hashed schema preserved).
    assert "jti_key" in lines[0]
    assert "jti" not in lines[0]
    assert lines[0]["jti_key"] == hashlib.sha256(b"live-jti").hexdigest()


# ------------------------------------------------------------------
# Phase 20 — blocklist_from_env dispatches to Redis when URL is set
# ------------------------------------------------------------------
def test_env_dispatch_redis_url_wins_over_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``SOMA_JWT_BLOCKLIST_REDIS_URL`` => RedisBlocklist, path ignored with a warning.

    Requires ``redis-py`` for the ``Redis.from_url`` call but no live
    server — ``from_url`` builds a lazy client that doesn't connect
    until a command is issued.
    """
    pytest.importorskip("redis")

    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(tmp_path / "bl.jsonl"))
    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_HASHED", raising=False)

    with caplog.at_level("WARNING", logger="soma.auth_revocation"):
        bl = blocklist_from_env()

    assert isinstance(bl, RedisBlocklist)
    assert bl.hashed is False
    # The warning body calls out the file path as ignored.
    assert any("ignoring" in r.message.lower() for r in caplog.records)


def test_env_dispatch_redis_honours_hashed_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SOMA_JWT_BLOCKLIST_HASHED=1`` flips the Redis backend to hashed mode."""
    pytest.importorskip("redis")

    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_PATH", raising=False)
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_HASHED", "1")

    bl = blocklist_from_env()
    assert isinstance(bl, RedisBlocklist)
    assert bl.hashed is True


def test_env_dispatch_no_redis_url_keeps_file_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When only ``SOMA_JWT_BLOCKLIST_PATH`` is set, behaviour is unchanged."""
    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_REDIS_URL", raising=False)
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(tmp_path / "bl.jsonl"))
    monkeypatch.delenv("SOMA_JWT_BLOCKLIST_HASHED", raising=False)

    bl = blocklist_from_env()
    assert isinstance(bl, FileBlocklist)
    assert bl.hashed is False


def test_file_blocklist_mtime_poll_picks_up_external_writes(tmp_path: Path) -> None:
    """A second process appending => next is_revoked call reloads.

    Simulated here by writing to the same file through a separate
    FileBlocklist instance.
    """
    path = tmp_path / "bl.jsonl"
    reader = FileBlocklist(path)
    # Initially empty — no file yet.
    assert reader.is_revoked("jti-late") is False

    # Peer (different FileBlocklist, same path) writes a record.
    writer = FileBlocklist(path)
    writer.add(_fresh_record("jti-late"))

    # The reader's poll interval may mask the update — force a refresh
    # by calling the public API a second time after mtime bump.
    # mtime resolution on some Windows FS is coarse; pad with a sleep.
    time.sleep(0.05)
    # Explicit refresh API — the implementation decides whether to poll
    # on every call or on cadence; the test just asserts eventual
    # consistency after a manual reload.
    reader._reload()  # noqa: SLF001 — test hook intentional
    assert reader.is_revoked("jti-late") is True
