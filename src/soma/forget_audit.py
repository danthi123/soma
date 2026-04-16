"""Append-only JSONL audit trail for ``ConversationalMemory.forget`` calls.

Phase 37 closes the GDPR-forgetting track with a compliance substrate:
every forget invocation (dry-run or live) emits one JSON line to an
operator-configured path. Records capture who ran the delete, which
subject (if different), the criteria, whether it was a preview, and
the resulting (or previewed) counts.

Design trade-offs:

- **Append-only**: the sink never truncates, rotates, or redacts. Let
  operators run ``logrotate`` / ``cron`` against the file — SOMA owns
  the *recording* of forget events, not the retention policy around
  the recording.
- **One-line records**: each emit is exactly one ``json.dumps()`` call
  followed by a ``\\n``, so ``tail -f`` / ``grep`` / ``jq -c`` work.
  ``ensure_ascii=False`` keeps non-ASCII text readable but every
  embedded newline in a criterion string is escaped to ``\\n``.
- **Small records**: deletion counts only, not id lists. Ids belong in
  a dry-run preview if the operator needs them; shipping long lists on
  every emit would both bloat the file and push individual writes past
  ``PIPE_BUF`` (4 KiB) where POSIX append-atomicity no longer holds.
- **Failure isolation**: an open/write error logs at WARNING and
  swallows — audit plumbing being broken shouldn't block a legitimate
  ``forget()`` call. Operators who treat the audit as must-land should
  monitor the ``soma.forget_audit`` logger for write failures.
- **Thread-safety**: an internal lock guards ``open-write-flush`` so
  concurrent :meth:`ConversationalMemory.forget` calls from a thread
  pool produce well-formed records. Cross-process writes rely on OS-
  level append atomicity (POSIX O_APPEND for records under PIPE_BUF;
  Windows ``FILE_APPEND_DATA`` for analogous small writes).

Env-driven opt-in:

- ``SOMA_FORGET_AUDIT_PATH``: path to the JSONL file. Parent
  directories are auto-created on first write. Unset or whitespace
  means "no audit" (a :class:`ForgetAuditSink` with ``path=None`` —
  every :meth:`emit` is a no-op).
- ``SOMA_FORGET_AUDIT_DISABLE=1``: forces the no-op mode even if the
  path env var is set, so operators who log via a different pipeline
  can ship the path config without triggering duplicate writes.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from soma.memory.conversational import ForgetPreview, ForgetResult

logger = logging.getLogger("soma.forget_audit")

# Env-var names are public API for operators — change them with a
# deprecation path if they ever need tweaking.
ENV_PATH = "SOMA_FORGET_AUDIT_PATH"
ENV_DISABLE = "SOMA_FORGET_AUDIT_DISABLE"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix.

    ``datetime.isoformat()`` renders ``+00:00`` for UTC, which is
    technically correct but trips naive string matchers. ``Z`` is the
    older-compat form most log aggregators expect, so we emit that.
    """
    now = dt.datetime.now(dt.UTC)
    # Drop microseconds past milliseconds — three decimal places is
    # plenty for ordering and keeps records compact.
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _result_to_dict(result: ForgetResult | ForgetPreview) -> dict[str, Any]:
    """Render either a :class:`ForgetResult` or :class:`ForgetPreview` as counts.

    Counts only — not the raw id lists — so records stay small (under
    the POSIX PIPE_BUF 4 KiB threshold where append-writes are atomic)
    and operators get audit-log compactness. A dry-run can always be
    replayed to get the id list if an investigator needs it.
    """
    # Late import so the audit module stays importable even when the
    # conversational module hasn't been touched yet (e.g. CLI scripts
    # that use :class:`ForgetAuditSink` directly).
    from soma.memory.conversational import ForgetPreview, ForgetResult

    if isinstance(result, ForgetResult):
        return {
            "deleted_turns": len(result.deleted_turns),
            "deleted_facts": len(result.deleted_facts),
            "deleted_summaries": len(result.deleted_summaries),
            "regenerated_summaries": len(result.regenerated_summaries),
            "total_deleted": result.total_deleted,
        }
    if isinstance(result, ForgetPreview):
        return {
            "raw_turns": len(result.raw_turns),
            "derived_facts": len(result.derived_facts),
            "summaries": len(result.summaries),
            "total_vectors": result.total_vectors,
        }
    # Future-proofing: any new shape falls through to a str() for the
    # record so we don't blow up the forget() call.
    return {"shape": type(result).__name__, "repr": repr(result)[:120]}


class ForgetAuditSink:
    """Append-only JSONL sink for forget-event records.

    Instantiate directly with ``path=`` for programmatic use, or call
    :meth:`from_env` in the common operator-driven path. A sink with
    ``path=None`` is a no-op — :meth:`emit` does nothing and never
    raises. Swapping a real sink for a no-op one is the recommended
    way to disable auditing in tests.

    The class holds a lock to serialise writes across threads within a
    process; cross-process serialisation is the OS's job (POSIX
    O_APPEND atomicity for small records). Callers who need stronger
    guarantees (multi-host, high-volume) should point the path at a
    named pipe / FIFO that feeds a dedicated collector.
    """

    __slots__ = ("_path", "_lock")

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        """Build a sink targeting ``path``.

        ``path=None`` (the documented default) yields the no-op sink so
        every construction site can treat the sink uniformly rather
        than branching on "auditing enabled?". :meth:`from_env` returns
        a no-op sink when the env vars are missing / disabled.
        """
        if path is None:
            self._path: Path | None = None
        else:
            # Normalise to :class:`Path` up-front so the rest of the
            # code can skip the str/PathLike branch.
            self._path = Path(os.fspath(path))
        self._lock = threading.Lock()

    @property
    def path(self) -> Path | None:
        """Backing JSONL path, or ``None`` for a no-op sink.

        Exposed mostly for tests that want to assert the sink resolved
        the env config correctly without writing a record.
        """
        return self._path

    @property
    def is_enabled(self) -> bool:
        """True when :meth:`emit` will attempt a write."""
        return self._path is not None

    @classmethod
    def from_env(cls) -> ForgetAuditSink:
        """Build a sink from environment variables.

        See module docstring for the env-var contract. Returns a no-op
        sink (``path=None``) whenever the path is missing, blank, or
        explicitly disabled — every caller can construct this and not
        check for auditing support.
        """
        raw = os.environ.get(ENV_PATH, "").strip()
        disabled = os.environ.get(ENV_DISABLE, "").strip() in ("1", "true", "True")
        if not raw or disabled:
            return cls(path=None)
        return cls(path=raw)

    def emit(
        self,
        *,
        user_id: str,
        target_user_id: str | None,
        criteria: dict[str, Any],
        dry_run: bool,
        result: ForgetResult | ForgetPreview,
    ) -> None:
        """Append one JSON record describing a forget-event.

        Records carry:
          - ``ts``: ISO-8601 UTC timestamp with ``Z`` suffix.
          - ``user_id``: the caller (principal.sub, or ``"anonymous"``
            in open-mode deploys). Always present.
          - ``target_user_id``: the subject, when different from the
            caller; ``None`` when the caller scrubbed their own data.
          - ``criteria``: the dict passed to :meth:`ForgetRequest` /
            :meth:`ConversationalMemory.forget`. Values are rendered
            verbatim — callers must not put secrets in criterion
            strings.
          - ``dry_run``: whether this was a preview-only call.
          - ``result``: counts drawn from either the
            :class:`ForgetResult` or :class:`ForgetPreview`.

        Errors opening the file (bad path, permission denied) or
        writing (disk full, closed FS) are caught and logged at
        WARNING. They never propagate — audit plumbing is advisory,
        not blocking. Operators who need hard guarantees should
        monitor the ``soma.forget_audit`` logger for the warning
        channel.
        """
        if self._path is None:
            return
        record: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "user_id": user_id,
            "target_user_id": target_user_id,
            "criteria": criteria,
            "dry_run": bool(dry_run),
            "result": _result_to_dict(result),
        }
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "forget_audit: could not serialise record (%s); dropping",
                exc,
                extra={
                    "event": "forget_audit_serialise_failure",
                    "user_id": user_id,
                },
            )
            return
        # Guard open+write+flush so interleaved calls from a thread
        # pool don't tear records. This doesn't (and can't) protect
        # cross-process writes — those rely on O_APPEND atomicity.
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
            except OSError as exc:
                # Audit path is advisory. Warn loudly so the operator
                # sees it, but do not raise — a broken audit path must
                # not prevent a user's forget() from landing.
                logger.warning(
                    "forget_audit: write to %s failed (%s); dropping record",
                    self._path,
                    exc,
                    extra={
                        "event": "forget_audit_write_failure",
                        "path": str(self._path),
                    },
                )
