# Phase 21: Benchmark Scheduler / CI Regression Cron

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Re-run the SOMA benchmark matrix automatically so perf regressions don't hide. Two triggers: (a) nightly cron on main, (b) any PR that touches `src/soma/memory/**`.

**Architecture:** GitHub Actions workflow that invokes the existing harnesses (`benchmarks/run_scale_vs_chroma.py`, `benchmarks/run_retrieval.py`, `benchmarks/run_backend_matrix.py`). Output compared against a checked-in `benchmarks/golden/*.json` via a new `scripts/check_bench_regressions.py` helper that fails the job if any metric regresses by more than a configured tolerance (default ±20% on latency, ±5% on recall).

**Design principles:**
- Default to non-blocking on PRs (post comment, don't fail CI) for the first iteration — lets operators decide whether a regression is real before we gate merges
- Blocking on nightly main runs — fires an alert/issue on regression
- Keep it cheap (<10 min wall-clock for the small-N matrix); the 1M benchmark stays manual

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/observability.md`.

---

### Task 1: `scripts/check_bench_regressions.py`

**Files:**
- Create: `scripts/check_bench_regressions.py`
- Create: `benchmarks/golden/` directory with committed JSON snapshots (copy from current `benchmarks/reports/*.md`'s extracted numbers — or regenerate against current main and commit as the starting line)
- Create: `tests/test_scripts/test_check_bench_regressions.py`

**Script contract:**
```bash
python scripts/check_bench_regressions.py \
  --current benchmarks/reports/scale_vs_chroma.json \
  --golden benchmarks/golden/scale_vs_chroma.json \
  --tolerance-latency 0.20 \
  --tolerance-recall 0.05

# exit 0 on within-tolerance; exit 1 with a diff-table on regression
```

The harnesses should emit JSON sidecars alongside the existing markdown. If they don't today, add the emit — but only in a way that doesn't change the markdown output (both written in the same pass).

**Step 1: Tests.**
```python
def test_regression_under_tolerance_passes():
    ...

def test_regression_above_tolerance_fails():
    ...

def test_missing_golden_file_creates_it(tmp_path, capsys):
    # First run on a new bench — create the golden instead of failing.
    ...

def test_new_metric_in_current_is_recorded(capsys):
    # Golden has {a, b}; current has {a, b, c} — record c as a new
    # entry, don't regress on it.
```

**Step 5:** `git commit -m "feat(bench): regression checker script + tests"`

---

### Task 2: Harness JSON sidecar output

**Files:**
- Modify: `benchmarks/run_scale_vs_chroma.py`, `benchmarks/run_retrieval.py`, `benchmarks/run_backend_matrix.py` — emit a parallel `.json` alongside the `.md` report.

Keep the markdown unchanged; add a `--json-out` flag (or auto-derive from `--out`) that writes a JSON serialisation of the rows. Already done in `run_backend_matrix.py` (writes `backend_matrix.json`). Extend the other two.

**Step 5:** `git commit -m "bench: JSON sidecar output on scale + retrieval harnesses"`

---

### Task 3: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/bench-regression.yml`

**Workflow shape:**
```yaml
name: Benchmark Regression Check

on:
  pull_request:
    paths:
      - 'src/soma/memory/**'
      - 'src/soma/io/**'
      - 'benchmarks/**'
      - '.github/workflows/bench-regression.yml'
  schedule:
    - cron: '0 8 * * *'   # 08:00 UTC nightly
  workflow_dispatch:

jobs:
  bench:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - name: Install
        run: pip install -e ".[dev]"
      - name: Run bench matrix
        run: |
          python -m benchmarks.run_scale_vs_chroma --json-out /tmp/scale.json
          python -m benchmarks.run_retrieval --json-out /tmp/retrieval.json
          python -m benchmarks.run_backend_matrix --json-out /tmp/backend.json
      - name: Check regressions
        id: check
        run: |
          python scripts/check_bench_regressions.py \
            --current /tmp/scale.json --golden benchmarks/golden/scale.json \
            --current /tmp/retrieval.json --golden benchmarks/golden/retrieval.json \
            --current /tmp/backend.json --golden benchmarks/golden/backend.json
        continue-on-error: ${{ github.event_name == 'pull_request' }}
      - name: Post PR comment
        if: github.event_name == 'pull_request' && steps.check.outcome == 'failure'
        uses: actions/github-script@v7
        with:
          script: |
            # post the diff table as a PR comment
```

**Step 5:** `git commit -m "ci: benchmark regression workflow"`

---

### Task 4: Docs

**Files:**
- Modify: `benchmarks/README.md` — one paragraph explaining the nightly / PR-gated regression check.

**Step 5:** `git commit -m "docs(bench): regression check guide"`

---

### Final sanity

- `ruff check scripts/ tests/test_scripts`
- `pytest tests/test_scripts -q`
- YAML validity: `python -c "import yaml; yaml.safe_load(open('.github/workflows/bench-regression.yml'))"`
- Verify `benchmarks/golden/*.json` files are committed.

No changes to `src/soma/` — this phase is pure operational tooling.
