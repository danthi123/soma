"""Tests for soma.log — stdlib JSONFormatter + configure_json_logging.

The formatter is used by the REST surface when ``SOMA_LOG_JSON=1`` so
log aggregators (Loki, Datadog, CloudWatch) can parse each log line as
structured JSON rather than wrestling ad-hoc format strings.
"""

from __future__ import annotations

import json
import logging
import re

from soma.log import JSONFormatter, configure_json_logging


def _make_record(
    name: str = "soma.test",
    level: int = logging.INFO,
    msg: str = "hello",
    extra: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )
    if extra is not None:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


def test_formatter_emits_valid_json_per_line() -> None:
    fmt = JSONFormatter()
    out = fmt.format(_make_record())
    data = json.loads(out)
    assert isinstance(data, dict)


def test_standard_fields_present() -> None:
    fmt = JSONFormatter()
    out = fmt.format(_make_record(name="soma.api", msg="retrieve"))
    data = json.loads(out)
    assert data["level"] == "INFO"
    assert data["name"] == "soma.api"
    assert data["message"] == "retrieve"
    assert "ts" in data


def test_extra_fields_merged() -> None:
    fmt = JSONFormatter()
    rec = _make_record(msg="retrieve", extra={"event": "retrieve", "k": 5})
    data = json.loads(fmt.format(rec))
    assert data["event"] == "retrieve"
    assert data["k"] == 5


def test_exc_info_included() -> None:
    fmt = JSONFormatter()
    logger = logging.getLogger("soma.exc.test")
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logger.makeRecord(
            name="soma.exc.test",
            level=logging.ERROR,
            fn=__file__,
            lno=1,
            msg="oh no",
            args=None,
            exc_info=__import__("sys").exc_info(),
        )
    out = fmt.format(rec)
    data = json.loads(out)
    assert "exc_info" in data
    assert "ValueError" in data["exc_info"]
    assert "boom" in data["exc_info"]


def test_ts_is_iso8601_utc() -> None:
    fmt = JSONFormatter()
    out = fmt.format(_make_record())
    data = json.loads(out)
    # ISO-8601 with 'Z' suffix, e.g. "2026-04-16T12:34:56.789123Z"
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$", data["ts"]), data["ts"]


def test_configure_json_logging_gated_on_env(monkeypatch) -> None:
    # Root formatter should swap to JSONFormatter when the env var is set.
    monkeypatch.setenv("SOMA_LOG_JSON", "1")
    configure_json_logging()
    root = logging.getLogger()
    found = any(
        isinstance(h.formatter, JSONFormatter) for h in root.handlers if h.formatter
    )
    assert found, "expected at least one root handler to carry JSONFormatter"


def test_configure_json_logging_noop_when_unset(monkeypatch) -> None:
    # Start clean: strip anything the previous test installed.
    monkeypatch.delenv("SOMA_LOG_JSON", raising=False)
    root = logging.getLogger()
    # Clear any JSONFormatters from prior tests.
    for h in list(root.handlers):
        if h.formatter and isinstance(h.formatter, JSONFormatter):
            h.setFormatter(logging.Formatter())
    configure_json_logging()
    still_json = any(
        isinstance(h.formatter, JSONFormatter)
        for h in root.handlers
        if h.formatter
    )
    assert not still_json
