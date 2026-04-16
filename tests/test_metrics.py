"""Tests for soma.metrics — prometheus-client primitives + Noop fallbacks.

The metrics module must gracefully degrade when ``prometheus-client``
is not installed (it lives under the ``soma[metrics]`` extra), so user
code that happens to call ``metrics.STORE_TOTAL.labels(...).inc()``
does not blow up at import time.
"""

from __future__ import annotations

import importlib

import pytest

_prom_missing = False
try:
    import prometheus_client  # noqa: F401
except ImportError:  # pragma: no cover
    _prom_missing = True


# -------- Noop fallback: always runs (no prometheus-client required) --------


def test_noop_counter_is_callable_and_safe() -> None:
    from soma.metrics import NoopCounter

    c = NoopCounter()
    c.inc()
    c.inc(5)
    c.labels(bundle="x").inc()
    c.labels(bundle="x").inc(3)


def test_noop_histogram_observe_is_safe() -> None:
    from soma.metrics import NoopHistogram

    h = NoopHistogram()
    h.observe(0.1)
    h.labels(bundle="x").observe(0.2)
    with h.time():
        pass
    with h.labels(bundle="x").time():
        pass


def test_noop_gauge_set_is_safe() -> None:
    from soma.metrics import NoopGauge

    g = NoopGauge()
    g.set(42)
    g.labels(bundle="x").set(7)
    g.inc()
    g.dec()


def test_batch_bucket_helper() -> None:
    from soma.metrics import batch_bucket

    assert batch_bucket(1) == "1"
    assert batch_bucket(5) == "10"
    assert batch_bucket(10) == "10"
    assert batch_bucket(50) == "100"
    assert batch_bucket(100) == "100"
    assert batch_bucket(500) == "1000"
    assert batch_bucket(1000) == "1000"
    assert batch_bucket(10000) == "1000+"


# ----------- Real prometheus-client path: skipped if extra missing ------------

pytestmark_prom = pytest.mark.skipif(
    _prom_missing, reason="prometheus-client not installed (install soma[metrics])"
)


@pytestmark_prom
def test_metrics_module_imports() -> None:
    mod = importlib.import_module("soma.metrics")
    assert hasattr(mod, "STORE_TOTAL")
    assert hasattr(mod, "RETRIEVE_LATENCY")
    assert hasattr(mod, "ENTRIES")


@pytestmark_prom
def test_counters_increment() -> None:
    from soma import metrics as m

    m.STORE_TOTAL.labels(bundle="__test__").inc()
    m.STORE_TOTAL.labels(bundle="__test__").inc()
    val = m.STORE_TOTAL.labels(bundle="__test__")._value.get()
    assert val >= 2


@pytestmark_prom
def test_store_batch_counter() -> None:
    from soma import metrics as m

    m.STORE_BATCH_TOTAL.labels(bundle="__batch_test__").inc()
    assert m.STORE_BATCH_TOTAL.labels(bundle="__batch_test__")._value.get() >= 1


@pytestmark_prom
def test_retrieve_counter_labeled_by_backend() -> None:
    from soma import metrics as m

    m.RETRIEVE_TOTAL.labels(bundle="__b__", backend="linear").inc()
    m.RETRIEVE_TOTAL.labels(bundle="__b__", backend="faiss").inc()
    v_lin = m.RETRIEVE_TOTAL.labels(bundle="__b__", backend="linear")._value.get()
    v_faiss = m.RETRIEVE_TOTAL.labels(bundle="__b__", backend="faiss")._value.get()
    assert v_lin >= 1 and v_faiss >= 1


@pytestmark_prom
def test_histograms_observe() -> None:
    from soma import metrics as m

    m.RETRIEVE_LATENCY.labels(bundle="__h__", backend="linear").observe(0.02)
    m.RETRIEVE_LATENCY.labels(bundle="__h__", backend="linear").observe(0.1)
    # Prometheus-client exposes per-label _sum via the underlying
    # value wrapper; we read it via the public collect() API below.
    target = {"bundle": "__h__", "backend": "linear"}
    total = 0.0
    for fam in m.RETRIEVE_LATENCY.collect():
        for s in fam.samples:
            if s.name.endswith("_sum") and s.labels == target:
                total = float(s.value)
    assert total >= 0.12


@pytestmark_prom
def test_gauges_set() -> None:
    from soma import metrics as m

    m.ENTRIES.labels(bundle="__g__").set(42)
    assert m.ENTRIES.labels(bundle="__g__")._value.get() == 42


@pytestmark_prom
def test_bm25_and_faiss_counters() -> None:
    from soma import metrics as m

    m.BM25_REBUILD_TOTAL.inc()
    m.FAISS_REBUILD_TOTAL.labels(index_type="flat").inc()
    assert m.BM25_REBUILD_TOTAL._value.get() >= 1
    assert m.FAISS_REBUILD_TOTAL.labels(index_type="flat")._value.get() >= 1


@pytestmark_prom
def test_consolidate_counter_and_histogram() -> None:
    from soma import metrics as m

    m.CONSOLIDATE_TOTAL.inc()
    m.CONSOLIDATE_SECONDS.observe(0.5)
    assert m.CONSOLIDATE_TOTAL._value.get() >= 1
    assert m.CONSOLIDATE_SECONDS._sum.get() >= 0.5


@pytestmark_prom
def test_embed_latency_labeled_by_batch_bucket() -> None:
    from soma import metrics as m

    m.EMBED_LATENCY.labels(batch_size_bucket="1").observe(0.005)
    m.EMBED_LATENCY.labels(batch_size_bucket="1000+").observe(0.5)
    assert m.EMBED_LATENCY.labels(batch_size_bucket="1")._sum.get() >= 0.005


@pytestmark_prom
def test_wal_counters_and_flush_histogram() -> None:
    from soma import metrics as m

    m.WAL_APPEND_TOTAL.labels(op="store").inc()
    m.WAL_APPEND_TOTAL.labels(op="forget").inc()
    m.WAL_FLUSH_SECONDS.observe(0.001)
    assert m.WAL_APPEND_TOTAL.labels(op="store")._value.get() >= 1


@pytestmark_prom
def test_loaded_bundles_gauge() -> None:
    from soma import metrics as m

    m.LOADED_BUNDLES.set(3)
    assert m.LOADED_BUNDLES._value.get() == 3


@pytestmark_prom
def test_forget_counter() -> None:
    from soma import metrics as m

    m.FORGET_TOTAL.labels(bundle="__f__").inc()
    assert m.FORGET_TOTAL.labels(bundle="__f__")._value.get() >= 1


@pytestmark_prom
def test_faiss_index_size_gauge() -> None:
    from soma import metrics as m

    m.FAISS_INDEX_SIZE.labels(bundle="__x__").set(1234)
    assert m.FAISS_INDEX_SIZE.labels(bundle="__x__")._value.get() == 1234
