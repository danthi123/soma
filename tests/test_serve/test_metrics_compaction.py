"""Phase 8 Task 1 + 3 — compaction metrics + bundle-label cardinality escape.

Covers:
- ``soma_compaction_total{bundle, outcome}`` increments on both ok and
  error outcomes (Task 1).
- ``soma_compaction_seconds{bundle}`` observes a timing sample on every
  consolidate() call, regardless of outcome (Task 1).
- ``SOMA_METRICS_BUNDLE_LABEL_DISABLE=1`` collapses the ``bundle`` label
  to ``"_disabled"`` at every call site via the ``_bundle_label`` helper
  (Task 3).

Skipped wholesale when prometheus-client is absent — the Noop* fallbacks
record nothing, so there's nothing to assert on.
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
    return float(c._value.get())


def _histogram_count(metric, labels: dict | None = None) -> float:
    target_labels = labels or {}
    for fam in metric.collect():
        for s in fam.samples:
            if not s.name.endswith("_count"):
                continue
            if target_labels and s.labels != target_labels:
                continue
            return float(s.value)
    return 0.0


# ------------------------------------------------------------------
# Task 1 — compaction counter + histogram
# ------------------------------------------------------------------
class TestCompactionMetrics:
    def test_compaction_total_increments_on_consolidate_ok(self) -> None:
        from soma.memory.api import MemoryLayer
        from soma.metrics import COMPACTION_SECONDS, COMPACTION_TOTAL

        mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
        mem._bundle_name = "ok_bundle"  # type: ignore[attr-defined]
        before_ct = _counter_val(
            COMPACTION_TOTAL.labels(bundle="ok_bundle", outcome="ok")
        )
        before_obs = _histogram_count(
            COMPACTION_SECONDS, {"bundle": "ok_bundle"}
        )
        mem.consolidate()  # no SOMA attached -> returns 0, ok outcome
        after_ct = _counter_val(
            COMPACTION_TOTAL.labels(bundle="ok_bundle", outcome="ok")
        )
        after_obs = _histogram_count(
            COMPACTION_SECONDS, {"bundle": "ok_bundle"}
        )
        assert after_ct - before_ct == 1
        assert after_obs - before_obs == 1

    def test_compaction_total_records_error_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soma.memory.api import MemoryLayer
        from soma.metrics import COMPACTION_SECONDS, COMPACTION_TOTAL

        mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
        mem._bundle_name = "err_bundle"  # type: ignore[attr-defined]

        def _boom() -> int:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(mem, "_consolidate_impl", _boom)

        before_ct = _counter_val(
            COMPACTION_TOTAL.labels(bundle="err_bundle", outcome="error")
        )
        before_obs = _histogram_count(
            COMPACTION_SECONDS, {"bundle": "err_bundle"}
        )
        with pytest.raises(RuntimeError, match="kaboom"):
            mem.consolidate()
        after_ct = _counter_val(
            COMPACTION_TOTAL.labels(bundle="err_bundle", outcome="error")
        )
        after_obs = _histogram_count(
            COMPACTION_SECONDS, {"bundle": "err_bundle"}
        )
        # Error outcome still ticks the counter AND records a timing.
        assert after_ct - before_ct == 1
        assert after_obs - before_obs == 1

    def test_compaction_ok_and_error_labelled_distinctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ok call and an error call on the same bundle touch two
        distinct label series (outcome=ok vs outcome=error)."""
        from soma.memory.api import MemoryLayer
        from soma.metrics import COMPACTION_TOTAL

        mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
        mem._bundle_name = "mixed_bundle"  # type: ignore[attr-defined]
        before_ok = _counter_val(
            COMPACTION_TOTAL.labels(bundle="mixed_bundle", outcome="ok")
        )
        before_err = _counter_val(
            COMPACTION_TOTAL.labels(bundle="mixed_bundle", outcome="error")
        )
        mem.consolidate()  # ok
        monkeypatch.setattr(
            mem, "_consolidate_impl", lambda: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with pytest.raises(RuntimeError):
            mem.consolidate()
        after_ok = _counter_val(
            COMPACTION_TOTAL.labels(bundle="mixed_bundle", outcome="ok")
        )
        after_err = _counter_val(
            COMPACTION_TOTAL.labels(bundle="mixed_bundle", outcome="error")
        )
        assert after_ok - before_ok == 1
        assert after_err - before_err == 1


# ------------------------------------------------------------------
# Task 3 — bundle-label cardinality escape
# ------------------------------------------------------------------
class TestBundleLabelDisabled:
    def test_bundle_label_helper_returns_raw_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SOMA_METRICS_BUNDLE_LABEL_DISABLE", raising=False)
        from soma.metrics import _bundle_label

        assert _bundle_label("my_bundle") == "my_bundle"

    def test_bundle_label_helper_collapses_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOMA_METRICS_BUNDLE_LABEL_DISABLE", "1")
        from soma.metrics import _bundle_label

        assert _bundle_label("any_bundle") == "_disabled"
        assert _bundle_label("other_bundle") == "_disabled"

    def test_bundle_label_helper_value_other_than_1_is_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exactly "1" activates the escape; "true"/"yes"/"0" do not.
        monkeypatch.setenv("SOMA_METRICS_BUNDLE_LABEL_DISABLE", "true")
        from soma.metrics import _bundle_label

        assert _bundle_label("my_bundle") == "my_bundle"
        monkeypatch.setenv("SOMA_METRICS_BUNDLE_LABEL_DISABLE", "0")
        assert _bundle_label("my_bundle") == "my_bundle"

    def test_store_emits_disabled_label_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: setting the env at store() time collapses the
        bundle label on the resulting STORE_TOTAL series."""
        monkeypatch.setenv("SOMA_METRICS_BUNDLE_LABEL_DISABLE", "1")
        from soma.memory.api import MemoryLayer
        from soma.metrics import STORE_TOTAL

        mem_a = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
        mem_a._bundle_name = "bundle_A"  # type: ignore[attr-defined]
        mem_b = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
        mem_b._bundle_name = "bundle_B"  # type: ignore[attr-defined]

        before_disabled = _counter_val(STORE_TOTAL.labels(bundle="_disabled"))
        mem_a.store("hello from A")
        mem_b.store("hello from B")
        after_disabled = _counter_val(STORE_TOTAL.labels(bundle="_disabled"))
        # Both stores collapse into the "_disabled" series.
        assert after_disabled - before_disabled >= 2

    def test_compaction_emits_disabled_label_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOMA_METRICS_BUNDLE_LABEL_DISABLE", "1")
        from soma.memory.api import MemoryLayer
        from soma.metrics import COMPACTION_TOTAL

        mem = MemoryLayer(embed_fn=_hash_embed, embed_dim=16)
        mem._bundle_name = "cardinality_big"  # type: ignore[attr-defined]
        before = _counter_val(
            COMPACTION_TOTAL.labels(bundle="_disabled", outcome="ok")
        )
        mem.consolidate()
        after = _counter_val(
            COMPACTION_TOTAL.labels(bundle="_disabled", outcome="ok")
        )
        assert after - before == 1
