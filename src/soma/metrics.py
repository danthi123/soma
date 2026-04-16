"""Prometheus metrics for SOMA's memory layer.

All metric primitives live here so each call site can do::

    from soma import metrics as m

    with m.RETRIEVE_LATENCY.labels(bundle=name, backend="linear").time():
        ...

without worrying about whether ``prometheus-client`` is installed.

When the ``soma[metrics]`` extra is not installed, every metric falls
back to a :class:`NoopCounter` / :class:`NoopHistogram` /
:class:`NoopGauge` that silently swallows ``inc`` / ``observe`` /
``set`` / ``.time()`` calls. This preserves the "optional extra"
contract: the core ``serve`` path works with zero observability deps.

Metric reference (see ``docs/observability.md`` for a labelled table):

Counters
    ``soma_store_total{bundle}`` — store() calls
    ``soma_store_batch_total{bundle}`` — store_batch() calls
    ``soma_retrieve_total{bundle, backend}`` — retrieve() calls, labelled
        by which rank path (linear vs faiss) actually ran.
    ``soma_forget_total{bundle}`` — forget() calls
    ``soma_faiss_rebuild_total{index_type}`` — FAISS index rebuilds
    ``soma_bm25_rebuild_total`` — BM25 index rebuilds
    ``soma_consolidate_total`` — consolidate() calls
    ``soma_compaction_total{bundle, outcome}`` — consolidate() entry/exit,
        outcome="ok" | "error".
    ``soma_wal_append_total{op}`` — WAL appends (Phase 1 hook)
    ``soma_auth_failures_total{reason}`` — auth rejections on the REST API
        (invalid_token | expired_token | insufficient_perm | missing_credentials
         | revoked_token)

Gauges
    ``soma_entries{bundle}`` — live entries
    ``soma_loaded_bundles`` — MemoryLayer cache depth (serve.py)
    ``soma_faiss_index_size{bundle}`` — FAISS index ntotal

Histograms
    ``soma_retrieve_latency_seconds{bundle, backend}``
    ``soma_embed_latency_seconds{batch_size_bucket}`` — 1/10/100/1000/1000+
    ``soma_faiss_rebuild_seconds``
    ``soma_bm25_rebuild_seconds``
    ``soma_consolidate_seconds``
    ``soma_compaction_seconds{bundle}`` — consolidate() wall-clock,
        ok + error outcomes combined.
    ``soma_wal_flush_seconds``
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

__all__ = [
    "AUTH_FAILURES_TOTAL",
    "BM25_REBUILD_SECONDS",
    "BM25_REBUILD_TOTAL",
    "COMPACTION_SECONDS",
    "COMPACTION_TOTAL",
    "CONSOLIDATE_SECONDS",
    "CONSOLIDATE_TOTAL",
    "EMBED_LATENCY",
    "ENTRIES",
    "FAISS_INDEX_SIZE",
    "FAISS_REBUILD_SECONDS",
    "FAISS_REBUILD_TOTAL",
    "FORGET_TOTAL",
    "LOADED_BUNDLES",
    "NoopCounter",
    "NoopGauge",
    "NoopHistogram",
    "RELOAD_TOTAL",
    "RETRIEVE_LATENCY",
    "RETRIEVE_TOTAL",
    "STORE_BATCH_TOTAL",
    "STORE_TOTAL",
    "WAL_APPEND_TOTAL",
    "WAL_FLUSH_SECONDS",
    "_bundle_label",
    "batch_bucket",
    "prometheus_available",
]


# ---------------------------------------------------------------------------
# Noop fallback primitives. These keep the same API surface as
# prometheus-client's Counter/Histogram/Gauge so call sites don't need
# conditionals. ``.labels(...)`` returns self so chained calls work.
# ---------------------------------------------------------------------------


class _NoopTimer:
    def __enter__(self) -> _NoopTimer:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def __call__(self, fn: Any) -> Any:
        return fn


class NoopCounter:
    """Counter stub used when prometheus-client is not installed."""

    def labels(self, *_args: Any, **_kwargs: Any) -> NoopCounter:
        return self

    def inc(self, _amount: float = 1.0) -> None:
        return None


class NoopHistogram:
    """Histogram stub used when prometheus-client is not installed."""

    def labels(self, *_args: Any, **_kwargs: Any) -> NoopHistogram:
        return self

    def observe(self, _amount: float) -> None:
        return None

    def time(self) -> _NoopTimer:
        return _NoopTimer()


class NoopGauge:
    """Gauge stub used when prometheus-client is not installed."""

    def labels(self, *_args: Any, **_kwargs: Any) -> NoopGauge:
        return self

    def set(self, _value: float) -> None:
        return None

    def inc(self, _amount: float = 1.0) -> None:
        return None

    def dec(self, _amount: float = 1.0) -> None:
        return None


# ---------------------------------------------------------------------------
# Real primitives, iff prometheus-client is installed.
# ---------------------------------------------------------------------------

prometheus_available: bool
try:
    from prometheus_client import Counter, Gauge, Histogram

    prometheus_available = True
except ImportError:  # pragma: no cover
    prometheus_available = False
    Counter = None  # type: ignore[assignment, misc]
    Gauge = None  # type: ignore[assignment, misc]
    Histogram = None  # type: ignore[assignment, misc]


def _counter(name: str, doc: str, labels: tuple[str, ...] = ()) -> Any:
    if not prometheus_available:
        return NoopCounter()
    # Counters must have >0 label names OR be unlabelled; the client
    # rejects .labels() on unlabelled counters, which matches our .labels
    # calls only on labelled counters.
    if labels:
        return Counter(name, doc, list(labels))
    return Counter(name, doc)


def _gauge(name: str, doc: str, labels: tuple[str, ...] = ()) -> Any:
    if not prometheus_available:
        return NoopGauge()
    if labels:
        return Gauge(name, doc, list(labels))
    return Gauge(name, doc)


# Default bucket layout spans sub-ms (FAISS in-proc) to multi-second
# (cross-encoder rerank). Histogram buckets can be tuned at runtime by
# editing the .observe() call sites — the buckets themselves are fixed
# at metric creation time per prometheus-client semantics.
_DEFAULT_LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    5.0,
    float("inf"),
)


def _histogram(
    name: str,
    doc: str,
    labels: tuple[str, ...] = (),
    buckets: tuple[float, ...] | None = None,
) -> Any:
    if not prometheus_available:
        return NoopHistogram()
    buckets = buckets or _DEFAULT_LATENCY_BUCKETS
    if labels:
        return Histogram(name, doc, list(labels), buckets=buckets)
    return Histogram(name, doc, buckets=buckets)


# ---------------------------------------------------------------------------
# Public metrics. Import these, don't construct new ones in call sites.
# ---------------------------------------------------------------------------

# --- Counters ---
STORE_TOTAL = _counter(
    "soma_store_total",
    "Total MemoryLayer.store() calls.",
    ("bundle",),
)
STORE_BATCH_TOTAL = _counter(
    "soma_store_batch_total",
    "Total MemoryLayer.store_batch() calls (one per batch, not per item).",
    ("bundle",),
)
RETRIEVE_TOTAL = _counter(
    "soma_retrieve_total",
    "Total MemoryLayer.retrieve() calls, labelled by backend path.",
    ("bundle", "backend"),
)
FORGET_TOTAL = _counter(
    "soma_forget_total",
    "Total MemoryLayer.forget() calls that actually removed an entry.",
    ("bundle",),
)
FAISS_REBUILD_TOTAL = _counter(
    "soma_faiss_rebuild_total",
    "FAISS index rebuilds, labelled by index type.",
    ("index_type",),
)
BM25_REBUILD_TOTAL = _counter(
    "soma_bm25_rebuild_total",
    "BM25 lexical index rebuilds.",
)
CONSOLIDATE_TOTAL = _counter(
    "soma_consolidate_total",
    "Total MemoryLayer.consolidate() calls.",
)
WAL_APPEND_TOTAL = _counter(
    "soma_wal_append_total",
    "WAL records appended, labelled by op (store | forget | update_metadata).",
    ("op",),
)
RELOAD_TOTAL = _counter(
    "soma_reload_total",
    "reload_if_stale() invocations that applied ≥1 peer-committed record.",
    ("bundle",),
)
AUTH_FAILURES_TOTAL = _counter(
    "soma_auth_failures_total",
    "REST auth failures, labelled by reason "
    "(invalid_token | expired_token | insufficient_perm | missing_credentials | revoked_token).",
    ("reason",),
)

# --- Gauges ---
ENTRIES = _gauge(
    "soma_entries",
    "Number of live entries per MemoryLayer bundle.",
    ("bundle",),
)
LOADED_BUNDLES = _gauge(
    "soma_loaded_bundles",
    "Number of MemoryLayer instances currently cached by the REST server.",
)
FAISS_INDEX_SIZE = _gauge(
    "soma_faiss_index_size",
    "Entries in the active FAISS index for a bundle (0 when on linear path).",
    ("bundle",),
)

# --- Histograms ---
RETRIEVE_LATENCY = _histogram(
    "soma_retrieve_latency_seconds",
    "MemoryLayer.retrieve() wall-clock latency.",
    ("bundle", "backend"),
)
EMBED_LATENCY = _histogram(
    "soma_embed_latency_seconds",
    "Embed latency, labelled by input batch-size bucket (1/10/100/1000/1000+).",
    ("batch_size_bucket",),
)
FAISS_REBUILD_SECONDS = _histogram(
    "soma_faiss_rebuild_seconds",
    "Wall-clock time to rebuild the FAISS index.",
)
BM25_REBUILD_SECONDS = _histogram(
    "soma_bm25_rebuild_seconds",
    "Wall-clock time to rebuild the BM25 index.",
)
CONSOLIDATE_SECONDS = _histogram(
    "soma_consolidate_seconds",
    "Wall-clock time per consolidate() call.",
)
WAL_FLUSH_SECONDS = _histogram(
    "soma_wal_flush_seconds",
    "WAL flush duration (fsync cost).",
)
# Compaction metrics — MemoryLayer.consolidate() entry/exit. Outcome
# label distinguishes normal completion ("ok") from an exception bubbling
# out of the body ("error"). Buckets span the seconds-to-minutes range
# a graph-consolidation pass can plausibly take on a warm store.
COMPACTION_TOTAL = _counter(
    "soma_compaction_total",
    "Total number of consolidation cycles by outcome.",
    ("bundle", "outcome"),
)
COMPACTION_SECONDS = _histogram(
    "soma_compaction_seconds",
    "Duration of consolidation cycles.",
    ("bundle",),
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def batch_bucket(n: int) -> str:
    """Return the 1/10/100/1000/1000+ bucket label for a batch size.

    Used as the ``batch_size_bucket`` label on ``soma_embed_latency_seconds``
    so operators can see how latency scales with batch width without
    generating one series per distinct batch size.
    """
    if n <= 1:
        return "1"
    if n <= 10:
        return "10"
    if n <= 100:
        return "100"
    if n <= 1000:
        return "1000"
    return "1000+"


def _bundle_label(name: str) -> str:
    """Return the ``bundle`` label value, honouring the cardinality escape.

    Deploys hosting 10k+ bundles can set ``SOMA_METRICS_BUNDLE_LABEL_DISABLE=1``
    to collapse every ``.labels(bundle=...)`` emission into a single
    ``"_disabled"`` series. Keeps the dashboards + alerts wired without
    melting Prometheus's TSDB when cardinality blows up.

    Every MemoryLayer / REST call site that emits a ``bundle=`` label
    routes through this helper — grep for ``_bundle_label(`` to audit.
    The env check is per-call (O(1) dict lookup) so tests can toggle
    between calls without reloading the module.
    """
    if os.environ.get("SOMA_METRICS_BUNDLE_LABEL_DISABLE") == "1":
        return "_disabled"
    return name


# Exposed so call sites that only want to record a duration can use the
# histogram's ``.time()`` context manager via a shorter alias in pinch
# code. The Noop* fallbacks already support ``.time()`` as a no-op.
__timer_factory = contextlib.nullcontext  # mypy: keep module importable
