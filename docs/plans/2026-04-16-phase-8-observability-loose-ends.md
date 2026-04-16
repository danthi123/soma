# Phase 8: Observability Loose Ends Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Close the three items flagged in the Phase 3 plan Risks but explicitly deferred: compaction metrics, `/metrics` auth gate, and cardinality escape for high-bundle-count deploys.

**Architecture:** All changes land in `src/soma/metrics.py` (new metric definitions, label-disabling logic), `src/soma/serve.py` (auth gate on `/metrics`, compaction instrumentation wrapper), and `src/soma/memory/api.py` (emit timing around `consolidate()`). Tests in `tests/test_serve/` and `tests/test_memory/`. No API breakage — all three changes are opt-in via env.

**Tech Stack:** existing `prometheus-client` and `prometheus-fastapi-instrumentator` (no new deps), FastAPI `Depends(require_auth)`.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/observability.md`.

---

### Task 1: `soma_compaction_total` + `soma_compaction_seconds`

**Files:**
- Modify: `src/soma/metrics.py` — add `COMPACTION_TOTAL` Counter and `COMPACTION_SECONDS` Histogram
- Modify: `src/soma/memory/api.py` — wrap `consolidate()` body with start-time/exception tracking, emit metrics
- Create: `tests/test_serve/test_metrics_compaction.py` — test metric emission via `generate_latest()`

**Step 1: Write failing tests.**
```python
# tests/test_serve/test_metrics_compaction.py
def test_compaction_total_increments_on_consolidate():
    # Build a stubbed MemoryLayer with SOMA attached, call .consolidate()
    # Assert soma_compaction_total{outcome="ok"} +=1 and
    # soma_compaction_seconds_count +=1.

def test_compaction_total_records_error_outcome():
    # Inject a consolidate() that raises; assert outcome="error".
```

**Step 2:** Run tests → expect ImportError / missing-metric failure.

**Step 3: Implement.**
```python
# metrics.py additions
COMPACTION_TOTAL = Counter(
    "soma_compaction_total",
    "Total number of consolidation cycles by outcome.",
    ["bundle", "outcome"],
)
COMPACTION_SECONDS = Histogram(
    "soma_compaction_seconds",
    "Duration of consolidation cycles.",
    ["bundle"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
```

In `api.py` wrap `consolidate` so that it emits `outcome="ok" | "error"` and a histogram observation. Use `_bundle_label()` helper to honor the cardinality escape (see Task 3).

**Step 4:** Tests pass.

**Step 5: Commit.**
```bash
git commit -m "feat(metrics): soma_compaction_total + soma_compaction_seconds"
```

---

### Task 2: `SOMA_METRICS_PUBLIC=0` gate

**Files:**
- Modify: `src/soma/serve.py` — existing `/metrics` route; wrap with `Depends(require_auth)` when gate is off
- Create: `tests/test_serve/test_metrics_auth.py`

**Step 1: Write failing tests.**
```python
def test_metrics_public_by_default_no_auth():
    # env unset → /metrics returns 200 without bearer token.

def test_metrics_gated_requires_bearer():
    # SOMA_METRICS_PUBLIC=0 + no bearer → 401.
    # With valid admin bearer → 200.

def test_metrics_gated_read_perm_sufficient():
    # Read-scoped bearer suffices (bundle="*").
```

**Step 2:** Tests fail (env toggle not yet wired).

**Step 3: Implement.**
- Read `SOMA_METRICS_PUBLIC` at module import time. Default "1" (public).
- If "0", register the `/metrics` route with `Depends(require_auth)` with a read-only scope check; otherwise public.
- Refactor should be minimal — `prometheus-fastapi-instrumentator`'s `.expose()` accepts a FastAPI dependencies arg.

**Step 4:** Tests pass.

**Step 5: Commit.**
```bash
git commit -m "feat(serve): SOMA_METRICS_PUBLIC=0 gates /metrics behind bearer"
```

---

### Task 3: `SOMA_METRICS_BUNDLE_LABEL_DISABLE=1` cardinality escape

**Files:**
- Modify: `src/soma/metrics.py` — wrap every `_total`/histogram label emission with a `_bundle_label()` helper that returns "_disabled" when the env is set
- Add test to `tests/test_serve/test_metrics_compaction.py` (or new file) to verify

**Step 1: Write failing test.**
```python
def test_bundle_label_disabled_env_collapses_cardinality(monkeypatch):
    monkeypatch.setenv("SOMA_METRICS_BUNDLE_LABEL_DISABLE", "1")
    # Emit two metrics with different bundles; scrape /metrics;
    # both collapse to bundle="_disabled".
```

**Step 2:** Fails.

**Step 3: Implement.** Centralise via:
```python
def _bundle_label(name: str) -> str:
    return "_disabled" if os.environ.get("SOMA_METRICS_BUNDLE_LABEL_DISABLE") == "1" else name
```
Apply at every Counter/Histogram `.labels(bundle=...)` call site. Re-read env per call so tests can toggle (trivial cost).

**Step 4:** Tests pass.

**Step 5: Commit.**
```bash
git commit -m "feat(metrics): SOMA_METRICS_BUNDLE_LABEL_DISABLE cardinality escape"
```

---

### Final sanity

```bash
ruff check src/ tests/
pytest tests/test_serve tests/test_memory -q
```

Expected: 0 failures, 0 ruff errors.
