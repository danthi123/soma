"""OTel is an opt-in extra gated on SOMA_OTEL_ENABLED=1.

Off by default — no spans, no collector contact, no startup overhead.
When enabled (and ``soma[otel]`` installed), FastAPIInstrumentor is
applied to the app so each HTTP request produces a trace span.
"""

from __future__ import annotations

import importlib

import pytest

_otel_missing = False
try:
    import opentelemetry.instrumentation.fastapi  # noqa: F401
except ImportError:  # pragma: no cover
    _otel_missing = True

pytestmark = pytest.mark.skipif(
    _otel_missing,
    reason="opentelemetry.instrumentation.fastapi not installed (soma[otel])",
)


def _reload_serve() -> object:
    import soma.serve

    return importlib.reload(soma.serve)


def test_otel_not_initialized_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOMA_OTEL_ENABLED", raising=False)
    serve = _reload_serve()
    assert getattr(serve, "_otel_enabled", False) is False


def test_otel_initializes_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOMA_OTEL_ENABLED", "1")
    serve = _reload_serve()
    assert getattr(serve, "_otel_enabled", False) is True
    # Clean up so other tests don't see the instrumentation.
    monkeypatch.delenv("SOMA_OTEL_ENABLED", raising=False)
    _reload_serve()


def test_otel_disabled_with_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOMA_OTEL_ENABLED", "0")
    serve = _reload_serve()
    assert getattr(serve, "_otel_enabled", False) is False
