# Phase 39: Sustained-Load Test Harness

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Catch what a 1M cold benchmark misses: memory drift, GC
pauses, FD leaks, connection-pool exhaustion. Add a harness that runs
10 QPS for 1 hour (configurable) against a live `uvicorn` server and
reports the time-series distribution of latency + memory + GC pauses.
One bake run per release; nightly cron tracks trend vs. golden.

**Architecture:**
- `benchmarks/run_sustained_load.py` — harness launches `uvicorn` in
  a subprocess, issues HTTP requests at a configurable QPS from a
  thread pool, samples the server's process stats every 10 s.
- Request mix is configurable; default is weighted like a real chat
  workload: 70% `/retrieve`, 20% `/store`, 5% `/related`, 5%
  `/forget`. Mix tuneable via `--mix store=0.5,retrieve=0.5`.
- Sampler collects from `psutil`:
  * RSS / VMS (memory)
  * Open file descriptors
  * Thread count
  * `gc.get_stats()` snapshot (collections per generation, uncollectable)
- Client collects per-request latency into a histogram bucketed at
  10 ms / 50 ms / 100 ms / 500 ms / 1s / 5s / 10s / inf.
- Output:
  * `benchmarks/reports/sustained_load.md` — markdown summary
    (headline table: duration, QPS, p50/p95/p99/p99.9 latency, RSS
    delta, FD delta, pass/fail).
  * `benchmarks/reports/sustained_load.json` — full time-series for
    regression-check + Grafana import.
  * `benchmarks/reports/sustained_load_charts/*.png` — optional
    matplotlib plots (RSS over time, latency CDF, GC pause
    distribution). Gated on `matplotlib` being installed.
- Pass/fail criteria (configurable tolerance):
  * RSS drift: <20% growth between minute-30 and minute-60.
  * p95 latency drift: <2× between first and last quartile.
  * FD leak: `abs(final_fds - initial_fds) / initial_fds < 0.5`.
  * Zero 500s observed on any endpoint.
- Default run: 1 minute smoke (always-on). Long run: gated by
  `SOMA_SUSTAINED_LOAD_DURATION=3600` env.

**Why not just reuse `run_scale_vs_chroma`:** that harness is a
cold, single-shot measurement of store throughput. Sustained-load is
a different question — "does the server's resource footprint stay
bounded under steady load?". Different signal, different harness.

**Out-of-scope (central merge):** `CHANGELOG.md`, `deferred-items.md`.

---

### Task 1: Harness skeleton + sampler

**Files:**
- Create: `benchmarks/run_sustained_load.py`
- Create: `benchmarks/harness/sustained_load.py` (request generator,
  sampler, collector — keep the CLI thin).
- Create: `benchmarks/tests/test_sustained_load.py`

**API sketch:**
```python
@dataclass
class LoadResult:
    duration_s: float
    total_requests: int
    rss_samples: list[tuple[float, int]]   # (t_seconds, bytes)
    fd_samples: list[tuple[float, int]]
    latency_hist: dict[str, int]           # bucket_label -> count
    errors_by_code: dict[int, int]

def run_sustained_load(
    *, duration_s: float, qps: float, mix: dict[str, float],
    server_url: str, embed_stub: bool = True,
) -> LoadResult: ...
```

**Tests:**
```python
def test_smoke_5_seconds_passes(monkeypatch, tmp_path):
    # Actually spins up uvicorn, runs 5 s at 10 QPS, asserts
    # zero 500s + some latency samples collected.
    # Uses SOMA_EMBED_MODEL=stub for fast responses.

def test_sampler_measures_rss_over_time(fake_process):
    # Unit-test the sampler against a stubbed psutil.Process.

def test_pass_fail_criteria_detect_leak(fake_result):
    # RSS grows 2x between minute 30 and minute 60 → FAIL.
    # RSS growth 5% → PASS.

def test_request_mix_honoured(fake_client):
    # Mix = {"store": 0.8, "retrieve": 0.2}; over 1000 requests,
    # assert ~80% were POST /store and ~20% were GET /retrieve
    # (within Bernoulli tolerance).
```

**Step 5:** `git commit -m "feat(bench): sustained_load harness skeleton + sampler"`

---

### Task 2: CLI + JSON report

**Files:**
- Modify: `benchmarks/run_sustained_load.py` (CLI entrypoint)

**Shape:**
```
python -m benchmarks.run_sustained_load \
    [--duration 60] [--qps 10] [--mix WEIGHTS] \
    [--json-out PATH] [--md-out PATH] [--charts-dir PATH] \
    [--port 8420] [--tolerance-rss 0.20] [--tolerance-latency 2.0]
```

Report structure mirrors `run_retrieval.py` + `run_scale_vs_chroma.py`
for consistency with Phase 21's regression harness. JSON sidecar uses
the same schema the regression checker already reads — enables
nightly-cron comparison-against-golden once a golden exists.

**Tests:**
```python
def test_cli_runs_end_to_end(tmp_path):
    # 5-second smoke via CLI; JSON + markdown produced; both parse.

def test_json_schema_matches_regression_checker_expectation(tmp_path):
    # Shape compatible with scripts/check_bench_regressions.py.
```

**Step 5:** `git commit -m "feat(bench): sustained_load CLI + JSON report"`

---

### Task 3: Golden snapshot + nightly CI hook

**Files:**
- Create: `benchmarks/golden/sustained_load.json` (captured from a
  real 1-hour run on the dev box — this is operator-work-product; the
  agent commits a placeholder JSON with a "to be captured on target
  runner" note until the operator regenerates it).
- Modify: `.gitea/workflows/bench-regression.yml` — add a gated
  sustained-load step that runs only on the weekly nightly (not
  per-PR, since 1 hour is too long for PR CI).

**Gating:** add `if: ${{ github.event_name == 'schedule' }}` to the
sustained-load step so PRs skip it. Weekly runs call it with
`--duration 3600`.

**Tests:** none — this is CI plumbing.

**Step 5:** `git commit -m "ci(bench): weekly sustained-load regression"`

---

### Task 4: Docs

**Files:**
- Modify: `docs/observability.md` — new "Sustained-load testing"
  section describing when to run, what the pass/fail criteria mean,
  how to interpret a failure.

**Step 5:** `git commit -m "docs(observability): sustained-load test guide"`

---

### Final sanity

```bash
ruff check benchmarks
pytest benchmarks/tests/test_sustained_load.py -q
python -m benchmarks.run_sustained_load --duration 5  # real smoke
```

Baseline post-Phase-38: all suites green. Target +~8 new tests, 0
regressions in default pytest. Operator commits the 1-hour golden
after the first real capture.

**Gotchas:**
- uvicorn spin-up adds ~2 s to the harness; subtract it from the
  measurement window or the first few latency samples look bad.
- `psutil.Process(pid).open_files()` on Windows returns only files
  opened by *this* process; on Linux it includes socket FDs. Use
  `connections()` separately for a cleaner signal.
- `gc.get_stats()` is per-process; sample the target, not the
  client. Use `psutil` to introspect the server subprocess.
- 1 hour at 10 QPS = 36000 requests. That's enough to surface most
  O(N) leaks but not sub-percent ones. Document the sensitivity
  ceiling.
- Default `--mix` should match the dominant agent-memory path
  (retrieval-heavy). Operators with ingest-heavy workloads override.
