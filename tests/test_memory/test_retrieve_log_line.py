"""Single JSON log line per MemoryLayer.retrieve() call.

This pins the observability schema documented in ``docs/observability.md``.
Any caller plumbing logs into Loki / Datadog / CloudWatch depends on
the key set here being stable — adding fields is fine, removing or
renaming is a breaking change.
"""

from __future__ import annotations

import json
import logging

import pytest
import torch

from soma.log import JSONFormatter
from soma.memory.api import MemoryLayer


def _hash_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, generator=g)


def test_retrieve_emits_single_json_line_with_schema(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "log_schema_test"  # type: ignore[attr-defined]
    for t in ["alpha", "beta", "gamma"]:
        mem.store(t)

    caplog.set_level(logging.INFO, logger="soma.memory")
    mem.retrieve("alpha-ish", k=2)

    retrieve_records = [
        r for r in caplog.records
        if getattr(r, "event", None) == "retrieve"
    ]
    assert len(retrieve_records) == 1, (
        f"expected exactly one retrieve log line, got {len(retrieve_records)}"
    )
    rec = retrieve_records[0]
    expected = {
        "event",
        "bundle",
        "query_len",
        "k",
        "has_where",
        "hybrid_alpha",
        "rerank_top_n",
        "n_hits",
        "backend",
        "latency_ms",
        "cache_miss",
    }
    for key in expected:
        assert hasattr(rec, key), f"retrieve log line missing {key!r}"
    assert rec.event == "retrieve"
    assert rec.bundle == "log_schema_test"
    assert rec.k == 2
    assert rec.query_len == len("alpha-ish")
    assert rec.has_where is False
    assert rec.hybrid_alpha is None
    assert rec.rerank_top_n is None
    assert rec.backend == "linear"
    assert isinstance(rec.n_hits, int) and rec.n_hits <= 2
    assert isinstance(rec.latency_ms, (int, float)) and rec.latency_ms >= 0
    assert rec.cache_miss is False


def test_retrieve_log_is_valid_json_via_formatter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end: JSONFormatter-formatted retrieve line round-trips through
    json.loads with the schema keys intact."""
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "json_fmt_test"  # type: ignore[attr-defined]
    for t in ["alpha", "beta"]:
        mem.store(t)

    caplog.set_level(logging.INFO, logger="soma.memory")
    mem.retrieve("query", k=1)
    retrieve_records = [
        r for r in caplog.records if getattr(r, "event", None) == "retrieve"
    ]
    assert retrieve_records, "no retrieve log record captured"
    rec = retrieve_records[0]
    formatter = JSONFormatter()
    line = formatter.format(rec)
    data = json.loads(line)
    assert data["event"] == "retrieve"
    assert data["bundle"] == "json_fmt_test"
    assert data["k"] == 1
    assert data["backend"] == "linear"
    assert "latency_ms" in data


def test_retrieve_log_reflects_hybrid_and_where_flags(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "hybrid_where_test"  # type: ignore[attr-defined]
    mem.store("alpha", metadata={"lang": "en"})
    mem.store("beta", metadata={"lang": "en"})
    mem.store("gamma", metadata={"lang": "fr"})

    caplog.set_level(logging.INFO, logger="soma.memory")
    mem.retrieve(
        "something",
        k=2,
        where={"lang": "en"},
        hybrid_alpha=0.3,
    )
    rec = next(
        r for r in caplog.records if getattr(r, "event", None) == "retrieve"
    )
    assert rec.has_where is True
    assert rec.hybrid_alpha == 0.3
