"""Integration tests: MemoryLayer hot paths must update soma.metrics.

Each test asserts on the prometheus-client internal ``_value`` / ``_sum``
accumulators to confirm a counter / histogram moved after a call. Skip
entire module when prometheus-client is not installed, since the Noop*
fallbacks deliberately record nothing.
"""

from __future__ import annotations

import pytest
import torch

_prom_missing = False
try:
    import prometheus_client  # noqa: F401
except ImportError:  # pragma: no cover
    _prom_missing = True

pytestmark = pytest.mark.skipif(
    _prom_missing, reason="prometheus-client not installed"
)


def _hash_embed(text: str) -> torch.Tensor:
    seed = abs(hash(text)) % (2**31)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, generator=g)


def _counter_val(c) -> float:
    # Works for both labelled and unlabelled Counters. ._value.get()
    # exists on both after .labels() resolves.
    return float(c._value.get())


def _histogram_count(metric, labels: dict | None = None) -> float:
    """Return the current observation count for a Histogram, optionally
    filtered by exact label set. Uses ``.collect()`` so we don't depend
    on private attributes across prometheus-client versions."""
    target_labels = labels or {}
    for fam in metric.collect():
        for s in fam.samples:
            if not s.name.endswith("_count"):
                continue
            if target_labels and s.labels != target_labels:
                continue
            return float(s.value)
    return 0.0


def test_store_increments_counter() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import STORE_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "test_store_ctr"  # type: ignore[attr-defined]
    before = _counter_val(STORE_TOTAL.labels(bundle="test_store_ctr"))
    mem.store("hello world")
    mem.store("another fact")
    after = _counter_val(STORE_TOTAL.labels(bundle="test_store_ctr"))
    assert after - before >= 2


def test_store_batch_increments_counter() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import STORE_BATCH_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "test_batch_ctr"  # type: ignore[attr-defined]
    before = _counter_val(STORE_BATCH_TOTAL.labels(bundle="test_batch_ctr"))
    mem.store_batch(["one", "two", "three"])
    after = _counter_val(STORE_BATCH_TOTAL.labels(bundle="test_batch_ctr"))
    # One batch call → one counter tick (not 3).
    assert after - before >= 1


def test_retrieve_observes_latency() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import RETRIEVE_LATENCY, RETRIEVE_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "test_retrieve_lat"  # type: ignore[attr-defined]
    for t in ["alpha", "beta", "gamma"]:
        mem.store(t)
    before_total = _counter_val(
        RETRIEVE_TOTAL.labels(bundle="test_retrieve_lat", backend="linear")
    )
    labels = {"bundle": "test_retrieve_lat", "backend": "linear"}
    before_count = _histogram_count(RETRIEVE_LATENCY, labels)
    mem.retrieve("alpha-ish", k=2)
    after_total = _counter_val(
        RETRIEVE_TOTAL.labels(bundle="test_retrieve_lat", backend="linear")
    )
    after_count = _histogram_count(RETRIEVE_LATENCY, labels)
    assert after_total - before_total == 1
    assert after_count - before_count == 1


def test_retrieve_backend_label_linear() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import RETRIEVE_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "linear_bundle"  # type: ignore[attr-defined]
    mem.store("one")
    mem.store("two")
    before = _counter_val(
        RETRIEVE_TOTAL.labels(bundle="linear_bundle", backend="linear")
    )
    mem.retrieve("anything", k=1)
    after = _counter_val(
        RETRIEVE_TOTAL.labels(bundle="linear_bundle", backend="linear")
    )
    assert after - before == 1


def test_forget_increments_counter() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import FORGET_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "forget_bundle"  # type: ignore[attr-defined]
    nid = mem.store("ephemeral")
    before = _counter_val(FORGET_TOTAL.labels(bundle="forget_bundle"))
    assert mem.forget(nid) is True
    after = _counter_val(FORGET_TOTAL.labels(bundle="forget_bundle"))
    assert after - before == 1


def test_forget_unknown_id_does_not_increment() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import FORGET_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "forget_unknown"  # type: ignore[attr-defined]
    before = _counter_val(FORGET_TOTAL.labels(bundle="forget_unknown"))
    # Unknown id → returns False; counter should stay put.
    assert mem.forget("nope") is False
    after = _counter_val(FORGET_TOTAL.labels(bundle="forget_unknown"))
    assert after == before


def test_consolidate_noop_still_observes() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import CONSOLIDATE_SECONDS, CONSOLIDATE_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    before_ct = _counter_val(CONSOLIDATE_TOTAL)
    before_obs = _histogram_count(CONSOLIDATE_SECONDS)
    mem.consolidate()  # no SOMA attached → returns 0 but still records
    after_ct = _counter_val(CONSOLIDATE_TOTAL)
    after_obs = _histogram_count(CONSOLIDATE_SECONDS)
    assert after_ct - before_ct == 1
    assert after_obs - before_obs == 1


def test_entries_gauge_tracks_store_count() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import ENTRIES

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    mem._bundle_name = "entries_gauge"  # type: ignore[attr-defined]
    for i in range(4):
        mem.store(f"e{i}")
    g_val = float(ENTRIES.labels(bundle="entries_gauge")._value.get())
    assert g_val == 4
    mem.forget(mem._ids[0])
    g_val2 = float(ENTRIES.labels(bundle="entries_gauge")._value.get())
    assert g_val2 == 3


def test_embed_latency_recorded_with_bucket_label() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import EMBED_LATENCY

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    labels = {"batch_size_bucket": "1"}
    before = _histogram_count(EMBED_LATENCY, labels)
    mem.store("solo")  # single → bucket "1"
    after = _histogram_count(EMBED_LATENCY, labels)
    assert after - before >= 1


def test_bm25_rebuild_counter_fires_on_hybrid_retrieve() -> None:
    from soma.memory.api import MemoryLayer
    from soma.metrics import BM25_REBUILD_TOTAL

    mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
    for t in ["alpha beta", "gamma delta", "alpha gamma"]:
        mem.store(t)
    before = _counter_val(BM25_REBUILD_TOTAL)
    mem.retrieve("alpha", k=2, hybrid_alpha=0.5)
    after = _counter_val(BM25_REBUILD_TOTAL)
    # One rebuild on first hybrid call.
    assert after - before >= 1
