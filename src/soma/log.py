"""Structured JSON logging for SOMA.

Stdlib-only. Imports nothing outside ``logging``, ``json``, ``os``,
``datetime``, so it is always safe to pull in — no optional extras.

Usage:

    from soma.log import configure_json_logging

    # Called at server start-up. A no-op unless ``SOMA_LOG_JSON=1``.
    configure_json_logging()

The formatter emits one JSON object per line with these always-present
keys: ``ts`` (ISO-8601 UTC with a ``Z`` suffix), ``level``, ``name``,
``message``. Anything passed via ``logger.info("msg", extra={...})`` is
merged in as sibling keys, so callers can attach structured context
(bundle name, k, latency, etc.) without formatting it into the message
text. Exceptions surfaced via ``logger.exception(...)`` are rendered
into an ``exc_info`` string field.

Keeping this in stdlib means we can emit structured logs from every
code path without a hard dep (vs ``structlog``), and the format is the
de-facto standard for Loki / Datadog / CloudWatch ingest pipelines.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
from typing import Any

_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "taskName",
}


class JSONFormatter(logging.Formatter):
    """Format a :class:`logging.LogRecord` as a single JSON line.

    Output is a flat object: no nesting, no top-level list. Adding a new
    field to an emitted record is as simple as passing it through
    ``extra=``; the formatter merges any non-reserved ``LogRecord``
    attribute into the output.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = _dt.datetime.fromtimestamp(record.created, tz=_dt.UTC)
        # Strip "+00:00" → "Z" for canonical UTC form.
        iso = ts.isoformat()
        if iso.endswith("+00:00"):
            iso = iso[: -len("+00:00")] + "Z"
        payload: dict[str, Any] = {
            "ts": iso,
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Merge any caller-supplied extras (anything not part of the
        # standard LogRecord attribute set).
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)  # probe: serializable?
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_json_logging() -> None:
    """Swap the root logger's handler formatters to :class:`JSONFormatter`.

    Gated on ``SOMA_LOG_JSON=1``. Idempotent: calling twice is safe.
    Called from ``soma.serve`` at app start-up; callers embedding
    MemoryLayer directly can call it themselves.

    When the env var is unset, this is a no-op — the caller keeps
    whatever logging setup they already had (the stdlib default, or
    whatever pytest / uvicorn installed).
    """
    if os.environ.get("SOMA_LOG_JSON", "").strip() not in ("1", "true", "True"):
        return
    root = logging.getLogger()
    if not root.handlers:
        # Install a baseline handler on stderr so we don't silently drop
        # logs when nothing has configured logging yet.
        handler = logging.StreamHandler(sys.stderr)
        root.addHandler(handler)
    formatter = JSONFormatter()
    for handler in root.handlers:
        handler.setFormatter(formatter)
    # Default INFO so structured lines actually surface.
    if root.level == logging.WARNING or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
