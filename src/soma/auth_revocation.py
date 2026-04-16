"""JWT revocation blocklist — file-backed default, optional Redis.

Phase 4 shipped JWTs with auto-populated ``jti`` claims but no way to
revoke a leaked token short of rotating ``SOMA_JWT_SECRET`` (kills every
active token) or waiting for ``exp`` (default 30 days). This module
plugs that gap with a persistent ``jti`` blocklist.

Design decision lives in ``docs/plans/2026-04-16-jwt-revocation.md``.
Summary: file-backed JSONL is the default; Redis is opt-in via the
``soma[redis-revocation]`` extra for multi-host / k8s deploys where
the 30 s file-poll lag is unacceptable (Phase 20).

Layout::

    {"jti": "<uuid>", "revoked_at": <epoch>, "reason": "...", "exp": <epoch>}
    {"jti": "<uuid>", "revoked_at": <epoch>, "reason": "...", "exp": <epoch>}
    ...

One line per revocation, append-only, ``portalocker.Lock`` serialising
writes. Reads load the whole file into a ``set[str]`` on first call and
refresh when ``mtime`` advances (30 s poll cadence by default). Entries
past their original ``exp`` are treated as not-revoked and dropped on
the next ``gc_expired()`` — the verifier already rejects the underlying
token on ``exp`` alone, so keeping them around just wastes disk.

Callers opt in via ``SOMA_JWT_BLOCKLIST_PATH``. When the env var is
unset, :func:`blocklist_from_env` returns a null backend that treats
every ``is_revoked`` query as False. The JWT verifier accepts an
optional ``blocklist`` kwarg, so repos with the env var unset behave
exactly like pre-revocation Phase 4.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import portalocker

__all__ = [
    "BlocklistBackend",
    "FileBlocklist",
    "RedisBlocklist",
    "RevocationRecord",
    "blocklist_from_env",
    "null_blocklist",
]

# Reason strings are free-form but capped to keep pathological inputs
# from bloating the on-disk file. 256 chars is ample for a human
# explanation ("leaked on slack 2026-04-16, rotated laptop") without
# being a denial-of-service vector.
_REASON_MAX_LEN = 256

# mtime poll cadence — cheap stat() every 30 s catches peer-process
# appends without thrashing the filesystem. Operators running a single
# uvicorn process and writing revocations via the CLI on the same host
# should see propagation within one poll window.
_POLL_INTERVAL_SEC = 30.0


@dataclass(frozen=True)
class RevocationRecord:
    """One blocklist entry.

    ``jti`` keys the record — the same ``jti`` populated in
    :func:`soma.auth.issue_token`'s claims map. ``revoked_at`` is the
    epoch second the revocation happened; ``reason`` is a free-form
    operator note (capped at 256 chars at the store layer); ``exp`` is
    the original token's expiry so GC can drop entries whose tokens
    would already fail JWT validation.
    """

    jti: str
    revoked_at: int
    reason: str
    exp: int


@runtime_checkable
class BlocklistBackend(Protocol):
    """Minimal contract every blocklist backend satisfies.

    The ``verify_token`` hook only needs ``is_revoked``; the CLI layer
    uses ``add`` and ``gc_expired`` too. Keeping the surface small
    leaves room for a future :class:`RedisBlocklist` without touching
    call sites.
    """

    def is_revoked(self, jti: str) -> bool: ...

    def add(self, record: RevocationRecord) -> None: ...

    def gc_expired(self) -> int: ...


class _NullBlocklist:
    """No-op backend used when ``SOMA_JWT_BLOCKLIST_PATH`` is unset.

    Every ``is_revoked`` returns ``False`` so the REST server behaves
    exactly like pre-revocation Phase 4. ``add`` silently swallows —
    operators who try to revoke without configuring a path will hit the
    CLI-layer error instead (``_cmd_auth_revoke`` refuses to run when
    the env var is missing).
    """

    def is_revoked(self, jti: str) -> bool:  # noqa: ARG002 — Protocol shape
        return False

    def add(self, record: RevocationRecord) -> None:  # noqa: ARG002
        return None

    def gc_expired(self) -> int:
        return 0


_NULL_SINGLETON = _NullBlocklist()


def null_blocklist() -> BlocklistBackend:
    """Return the shared null backend — always safe to call.

    Exposed as a factory (not a public singleton) so future Redis /
    in-memory variants can be swapped in via the same import site.
    """
    return _NULL_SINGLETON


class FileBlocklist:
    """Append-only JSONL blocklist with mtime-polled refresh.

    Instances hold a cached ``set[str]`` of live ``jti`` values plus the
    last observed ``mtime``. Any call to :meth:`is_revoked` reloads the
    set when the file has advanced; writes go through ``portalocker``
    so peer processes on the same host can share the store without
    stepping on each other.

    ``hashed`` (default ``False``) switches the on-disk representation
    from plaintext ``jti`` to ``sha256(jti).hexdigest()``. Hashed stores
    are safe to exfiltrate — they leak *that* a jti is revoked, not
    *which* jti — at the cost of being unable to reverse-map the file
    back to the original tokens. Both schemas are always accepted on
    read (``jti_key`` from hashed writes, ``jti`` from legacy writes),
    so a half-migrated file keeps working; only the write path is
    mode-specific.
    """

    def __init__(self, path: Path, hashed: bool = False) -> None:
        self._path = Path(path)
        self._hashed = bool(hashed)
        self._cache: set[str] = set()
        # Sentinel: negative mtime forces the first is_revoked call to
        # populate the cache. time.time()-based cadence starts at 0 so
        # the first call always touches the fs.
        self._mtime: float = -1.0
        self._last_poll: float = 0.0
        self._reload()

    @property
    def hashed(self) -> bool:
        """Whether this instance writes sha256(jti) instead of plaintext."""
        return self._hashed

    def _key(self, jti: str) -> str:
        """Lookup/store key for ``jti`` under the configured mode."""
        if self._hashed:
            return hashlib.sha256(jti.encode("utf-8")).hexdigest()
        return jti

    # ------------------------------------------------------------------
    # BlocklistBackend surface
    # ------------------------------------------------------------------
    def is_revoked(self, jti: str) -> bool:
        """True iff ``jti`` is in the store AND its token hasn't expired.

        Refreshes from disk when mtime has advanced OR the poll
        interval has elapsed since the last check. The double gate
        means a rapid burst of calls doesn't hammer ``stat()`` while
        still catching peer appends within one poll window.
        """
        self._maybe_refresh()
        # Hashed-mode readers also honour legacy plaintext records that
        # may have survived from a pre-Phase-18 file: check both the
        # hashed key *and* the raw jti. Plain mode only stores raw jti,
        # so the second check is redundant but cheap.
        if self._key(jti) in self._cache:
            return True
        return bool(self._hashed and jti in self._cache)

    def add(self, record: RevocationRecord) -> None:
        """Append a revocation. Serialises with other writers on host.

        Takes a ``portalocker.Lock`` on the parent directory's
        ``.blocklist.lock`` file so concurrent ``soma auth revoke``
        invocations across processes don't interleave JSON lines. The
        cache is updated in-memory after a successful flush so the
        caller doesn't need to wait for the next poll.
        """
        # Cap the reason length at the store boundary — keeps the file
        # compact and stops a malicious caller from ballooning disk
        # with a 1 MB "reason".
        reason = record.reason
        if len(reason) > _REASON_MAX_LEN:
            reason = reason[:_REASON_MAX_LEN]
        # Schema: hashed mode writes {"jti_key": sha256(...)}; legacy /
        # plaintext mode writes {"jti": ...} with byte-identical layout
        # to pre-Phase-18 so existing files stay stable on upgrade.
        record_payload: dict[str, object] = {
            "revoked_at": int(record.revoked_at),
            "reason": reason,
            "exp": int(record.exp),
        }
        if self._hashed:
            record_payload["jti_key"] = self._key(record.jti)
        else:
            record_payload["jti"] = record.jti
        payload = json.dumps(record_payload, ensure_ascii=False)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_name(f".{self._path.name}.lock")
        # Acquire an exclusive lock on a sibling .lock file (not the
        # data file itself — portalocker on Windows can't hold an
        # exclusive lock on the file being appended to without racing
        # the append handle).
        with (
            portalocker.Lock(str(lock_path), mode="a", timeout=30),
            self._path.open("a", encoding="utf-8") as fh,
        ):
            fh.write(payload + "\n")
            fh.flush()
            # fsync is best-effort — some filesystems (tmpfs, Windows
            # on network shares) don't support it. The lock release is
            # already a barrier for same-host readers.
            with contextlib.suppress(OSError):
                os.fsync(fh.fileno())

        # Live-cache update — fast-path for the same process issuing
        # the revoke. Filter out exp-past entries symmetrically with
        # the disk load so the fast path behaves like the reload path.
        if int(record.exp) > int(time.time()):
            self._cache.add(self._key(record.jti))
        with contextlib.suppress(OSError):
            self._mtime = self._path.stat().st_mtime
        self._last_poll = time.time()

    def gc_expired(self) -> int:
        """Rewrite the file keeping only entries whose ``exp`` is in the future.

        Returns the count removed. Safe to call on a fresh store
        (returns 0). Takes the same sibling-file lock as :meth:`add`
        so GC doesn't race with revocations.
        """
        if not self._path.exists():
            return 0

        lock_path = self._path.with_name(f".{self._path.name}.lock")
        now = int(time.time())
        with portalocker.Lock(str(lock_path), mode="a", timeout=30):
            kept: list[dict[str, object]] = []
            removed = 0
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Torn line from a crashed writer. Drop it —
                        # it's already useless and the surviving
                        # entries are the source of truth.
                        continue
                    if int(rec.get("exp", 0)) > now:
                        kept.append(rec)
                    else:
                        removed += 1

            # Rewrite atomically via a temp file + replace. Keeps
            # readers from seeing a half-rewritten file if we crash
            # mid-flight.
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in kept:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                with contextlib.suppress(OSError):
                    os.fsync(fh.fileno())
            tmp.replace(self._path)

        # Refresh the cache to match disk.
        self._mtime = -1.0
        self._reload()
        return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _maybe_refresh(self) -> None:
        """Reload from disk when mtime or the poll window says we must."""
        now = time.time()
        try:
            current_mtime = self._path.stat().st_mtime
        except OSError:
            # File doesn't exist — nothing revoked. Cache stays.
            if self._cache and self._mtime >= 0:
                self._cache.clear()
                self._mtime = -1.0
            self._last_poll = now
            return

        if current_mtime != self._mtime or (now - self._last_poll) > _POLL_INTERVAL_SEC:
            self._reload()
            self._last_poll = now

    def _reload(self) -> None:
        """Rebuild the cache from disk. Tolerates torn final lines."""
        new_cache: set[str] = set()
        try:
            stat = self._path.stat()
        except OSError:
            self._cache = new_cache
            self._mtime = -1.0
            return

        now = int(time.time())
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Torn line — most often a partial write after
                        # a crash. Drop silently; the surviving entries
                        # remain authoritative.
                        continue
                    # Accept both the new jti_key (hashed writes) and
                    # the legacy jti (plaintext writes). A half-migrated
                    # file mixing both schemas still loads cleanly.
                    key = rec.get("jti_key") or rec.get("jti")
                    exp = int(rec.get("exp", 0))
                    if isinstance(key, str) and exp > now:
                        new_cache.add(key)
        except OSError:
            # Race with a concurrent writer — next poll tick retries.
            return

        self._cache = new_cache
        self._mtime = stat.st_mtime


class RedisBlocklist:
    """Redis-backed blocklist for multi-host / k8s deploys.

    Each revocation becomes one key at ``{key_prefix}{jti_key}`` with a
    TTL equal to the remaining seconds until the original token's
    ``exp``. Redis handles natural expiry, so :meth:`gc_expired` is a
    no-op. Propagation across peers is instant — no 30 s file-poll lag.

    ``hashed`` mirrors :class:`FileBlocklist`'s knob: when ``True`` the
    stored key is ``sha256(jti).hexdigest()`` rather than the raw
    ``jti``. Both backends use the same hashing scheme, so a cross-
    backend migration (file -> Redis or vice versa) keeps lookup keys
    byte-identical.

    The ``redis-py`` import is lazy so ``soma`` keeps installing
    cleanly without the ``redis-revocation`` extra. Constructing a
    :class:`RedisBlocklist` without the dep raises :class:`ImportError`
    pointing at the install hint.

    Parameters
    ----------
    url:
        Redis connection URL, e.g. ``redis://localhost:6379/0``.
    key_prefix:
        Prefix applied to every stored key. Keeps the blocklist
        namespace well-separated from other data in a shared Redis.
    hashed:
        When ``True`` the key suffix is ``sha256(jti).hexdigest()``
        instead of the raw ``jti``.
    client:
        Optional pre-built client (for tests — pass a
        ``fakeredis.FakeRedis`` here). When ``None`` (production path)
        the class builds one from ``url`` via ``redis.Redis.from_url``.
    """

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str = "soma:jwt:revoked:",
        hashed: bool = False,
        client: Any | None = None,
    ) -> None:
        self._url = str(url)
        self._key_prefix = str(key_prefix)
        self._hashed = bool(hashed)
        if client is not None:
            self._client = client
        else:
            self._client = self._build_client(self._url)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_client(url: str) -> Any:
        """Build a ``redis.Redis`` from ``url`` — lazy import the dep.

        Separated so tests can monkeypatch or bypass via the ``client``
        kwarg. Raises :class:`ImportError` with an install hint when
        the ``redis`` package isn't available — the only path that
        actually needs the dep.
        """
        try:
            import redis as _redis
        except ImportError as exc:  # pragma: no cover — tested via monkeypatch
            raise ImportError(
                "RedisBlocklist requires the 'redis' package. "
                "Install with: pip install 'soma[redis-revocation]'"
            ) from exc
        # ``decode_responses=True`` keeps the reason string a plain str
        # round-trip; our reads don't need raw bytes.
        return _redis.Redis.from_url(url, decode_responses=True)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def hashed(self) -> bool:
        """Whether this instance hashes the ``jti`` before storage."""
        return self._hashed

    @property
    def key_prefix(self) -> str:
        """Key namespace prefix (immutable after construction)."""
        return self._key_prefix

    def _key(self, jti: str) -> str:
        """Lookup/store key for ``jti`` under the configured mode.

        Mirrors :meth:`FileBlocklist._key` so a jti hashed by either
        backend maps to the same 64-char hex suffix — the prefix
        differs but the identity check in tests stays trivial.
        """
        if self._hashed:
            jti = hashlib.sha256(jti.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}{jti}"

    # ------------------------------------------------------------------
    # BlocklistBackend surface
    # ------------------------------------------------------------------
    def is_revoked(self, jti: str) -> bool:
        """True iff the key exists in Redis. TTL handles natural expiry."""
        # Hashed mode also honours legacy plaintext entries (same logic
        # as FileBlocklist) — covers a cross-backend migration where
        # the old file had plaintext records.
        if bool(self._client.exists(self._key(jti))):
            return True
        if self._hashed:
            legacy_key = f"{self._key_prefix}{jti}"
            return bool(self._client.exists(legacy_key))
        return False

    def add(self, record: RevocationRecord) -> None:
        """Write a revocation with a Redis TTL matching the token's exp.

        ``setex(key, ttl, value)`` is atomic, so no extra locking is
        needed. TTL is clamped at >= 1 s because Redis rejects 0-TTL
        writes — a record whose token has already expired is still
        worth keeping for a heartbeat so late requests race-lose
        cleanly rather than sneaking through.
        """
        reason = record.reason
        if len(reason) > _REASON_MAX_LEN:
            reason = reason[:_REASON_MAX_LEN]
        ttl = max(1, int(record.exp) - int(time.time()))
        self._client.setex(self._key(record.jti), ttl, reason or "revoked")

    def gc_expired(self) -> int:
        """No-op — Redis' native TTL auto-expires entries.

        Returned for :class:`BlocklistBackend` protocol symmetry.
        Always returns ``0``.
        """
        return 0


def blocklist_from_env() -> BlocklistBackend:
    """Factory — ``FileBlocklist`` if ``SOMA_JWT_BLOCKLIST_PATH`` set, else null.

    Unit tests can pin the factory with one ``monkeypatch.setenv`` call.
    Future Redis support slots in as a second branch here gated on
    ``SOMA_JWT_BLOCKLIST_BACKEND=redis``.

    Env contract:
    - ``SOMA_JWT_BLOCKLIST_PATH`` — unset => null backend. Set => file
      backend at that path.
    - ``SOMA_JWT_BLOCKLIST_HASHED`` — ``1`` => enable sha256(jti) at rest.
      Anything else (unset, empty, ``0``) keeps plaintext default.
    """
    path = os.environ.get("SOMA_JWT_BLOCKLIST_PATH", "").strip()
    if not path:
        return null_blocklist()
    hashed = os.environ.get("SOMA_JWT_BLOCKLIST_HASHED", "").strip() == "1"
    return FileBlocklist(Path(path), hashed=hashed)


def to_record_dict(rec: RevocationRecord) -> dict[str, object]:
    """Serialise a record to a plain dict (for CLI JSON output)."""
    return asdict(rec)
