"""Cross-platform PID-aware file lock using portalocker.

Lock file JSON format: ``{"pid": <int>, "ts": <epoch seconds>, "purpose": "<str>"}``.

Staleness: a lock is considered abandoned (reclaimable) when EITHER the owning
PID is no longer alive OR the timestamp is older than 30 minutes. This matches
the autonomous-loop design doc's lock contract.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import portalocker

STALE_AGE_SECONDS = 30 * 60


def is_pid_alive(pid: int) -> bool:
    """Return True iff a process with this PID exists and is visible."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _is_lock_stale(lock_data: dict[str, Any]) -> bool:
    pid = int(lock_data.get("pid", 0))
    ts = float(lock_data.get("ts", 0))
    if not is_pid_alive(pid):
        return True
    return time.time() - ts > STALE_AGE_SECONDS


class FileLock:
    """PID-aware exclusive lock on a file path.

    Uses a separate ``.guard`` sidecar file for portalocker's exclusive-access
    guarantee during the read-modify-write of the main lock payload. Staleness
    reclaim lets us recover from crashed lock holders.
    """

    def __init__(
        self,
        path: Path | str,
        timeout: float = 30.0,
        purpose: str = "",
    ) -> None:
        self.path = Path(path)
        self.timeout = float(timeout)
        self.purpose = purpose
        self._held = False

    def acquire(self) -> bool:
        """Attempt to acquire the lock within ``self.timeout`` seconds.

        Returns True on success, False on timeout.
        """
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                guard = self.path.with_suffix(self.path.suffix + ".guard")
                with portalocker.Lock(
                    guard,
                    mode="a",
                    timeout=0.5,
                    flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
                ):
                    if self.path.exists():
                        try:
                            existing = json.loads(self.path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError):
                            existing = None
                        if existing and not _is_lock_stale(existing):
                            time.sleep(0.5)
                            continue
                    payload = {
                        "pid": os.getpid(),
                        "ts": time.time(),
                        "purpose": self.purpose,
                    }
                    self.path.write_text(json.dumps(payload), encoding="utf-8")
                    self._held = True
                    return True
            except portalocker.exceptions.LockException:
                time.sleep(0.2)
                continue
        return False

    def release(self) -> None:
        if self._held and self.path.exists():
            with contextlib.suppress(OSError):
                self.path.unlink()
        self._held = False

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
