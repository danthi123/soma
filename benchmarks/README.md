# SOMA Benchmarks

Performance harnesses for the SOMA memory layer. Each script in
`benchmarks/` runs a fixed workload across one or more adapters and
emits a markdown report under `benchmarks/reports/`. The three scripts
wired into CI also drop a JSON sidecar the regression checker can read.

## Scripts

| Script | Comparison | Report | JSON sidecar |
| --- | --- | --- | --- |
| `run_retrieval.py` | SOMA vs Chroma + graph ablations | `reports/retrieval.md` | `reports/retrieval.json` |
| `run_scale_vs_chroma.py` | Pure efficiency across {1K, 5K, 20K} | `reports/scale_vs_chroma.md` | `reports/scale_vs_chroma.json` |
| `run_backend_matrix.py` | InProc / Qdrant / LanceDB adapters | `reports/backend_matrix.md` | `reports/backend_matrix.json` |
| `run_locomo.py` | LoCoMo long-context QA | `reports/locomo.md` | — |
| `run_scale_enterprise.py` | 5K / 20K / 100K / 1M sbert | `reports/scale_enterprise_*.md` | — |
| `run_plasticity_scale.py` | Consolidation plasticity at scale | `reports/plasticity_scale.md` | — |

Every harness script takes `--out` (markdown path). The three
CI-tracked ones take `--json-out` plus a `--lite` switch that skips
expensive arms.

## Regression CI (Phase 21)

`.github/workflows/bench-regression.yml` runs the three JSON-emitting
harnesses on:

- **Pull requests** touching `src/soma/memory/**`, `src/soma/io/**`,
  `benchmarks/**`, the checker script, or the workflow itself. Result
  is non-blocking — a diff table is posted as a PR comment if any
  metric drifts past tolerance.
- **Nightly cron** at 08:00 UTC on `main`. Blocking — a regression
  fails the job so the red build is the signal.
- **Manual dispatch** via `workflow_dispatch`.

The checker (`scripts/check_bench_regressions.py`) compares each
current JSON row-by-row against the matching `benchmarks/golden/*.json`
snapshot. Default tolerances: ±20% on latency / disk metrics, ±5% on
recall / MRR / NDCG. See `benchmarks/golden/README.md` for how the
goldens were generated and how to refresh them after an intentional
perf change.

The CI workflow stays under the 10-minute budget by:
- running `run_scale_vs_chroma` in full-matrix mode (it's the fastest)
- running `run_retrieval` with `--lite` (skips the slow graph-ablation
  arms)
- running `run_backend_matrix` with `--smoke` (InProcFlat only)

Large-scale runs (100K, 1M) stay manual — they take hours and aren't
appropriate for a per-PR loop.
