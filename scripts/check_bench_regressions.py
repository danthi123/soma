"""Compare current benchmark JSON output against a golden snapshot.

Part of Phase 21 (benchmark regression CI). Invoked from the
``.github/workflows/bench-regression.yml`` workflow after the harnesses
run on PR / nightly / manual dispatch. Also usable locally::

    python scripts/check_bench_regressions.py \\
        --current benchmarks/reports/scale_vs_chroma.json \\
        --golden  benchmarks/golden/scale_vs_chroma.json \\
        --current benchmarks/reports/retrieval.json \\
        --golden  benchmarks/golden/retrieval.json \\
        --tolerance-latency 0.20 \\
        --tolerance-recall  0.05

Contract:

- ``--current`` / ``--golden`` pairs may be repeated; every current JSON
  is checked against its matching golden snapshot.
- For numeric metrics: "latency/time/disk" metrics tolerate a relative
  drift of ``--tolerance-latency`` (default ±20%) in the *worse* direction
  (slower / larger). "recall/mrr/ndcg" metrics tolerate
  ``--tolerance-recall`` (default 5%) drift *downward*.
- If a ``--golden`` file does not exist, the checker writes the current
  JSON to that path and exits 0. This is how you bootstrap the first
  snapshot on a new bench.
- Metrics that appear in ``--current`` but not in ``--golden`` are
  recorded as new (pass). Metrics that disappear from current are
  flagged (fail).
- Exit 0 on pass, 1 on any regression; prints a diff table to stdout.

The JSON shape is whatever the harness writes — typically a list of
row-dataclasses serialised via ``dataclasses.asdict``. The checker
walks every list entry, matches rows by the shared key returned by
:func:`_row_key` (``system``+``n`` works across every current harness),
then compares each numeric field.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Fields that use "lower is better" — regression = got bigger by >tol
LATENCY_FIELDS = frozenset(
    {
        "store_total_s",
        "store_avg_ms",
        "retrieve_avg_ms",
        "retrieve_p50_ms",
        "retrieve_p95_ms",
        "retrieve_p99_ms",
        "disk_bytes",
        "disk_kb",
        "disk_mb",
    }
)

# Fields that use "higher is better" — regression = got smaller by >tol
QUALITY_FIELDS = frozenset(
    {
        "recall_at_k",
        "recall_at_10",
        "mrr_at_k",
        "ndcg_at_k",
    }
)

# Numeric fields to ignore entirely (identity / sample-count markers).
IGNORE_FIELDS = frozenset({"n", "num_entries"})


@dataclass
class Diff:
    """One metric diff (current vs golden) for one row."""

    dataset: str
    row_key: str
    metric: str
    golden: float | None
    current: float | None
    delta_pct: float | None
    status: str  # "pass" | "regression" | "new" | "missing"


def _row_key(row: dict[str, Any]) -> str:
    """Produce a stable key for a benchmark row.

    We combine ``system`` (or ``backend``) with scale (``n`` /
    ``num_entries``) so multi-scale harnesses (scale_vs_chroma has
    1K/5K/20K × soma-flat/soma-hnsw/chroma) produce unique keys.
    """
    sys_key = row.get("system") or row.get("backend") or "row"
    scale = row.get("n") if "n" in row else row.get("num_entries", "")
    if scale != "" and scale is not None:
        return f"{sys_key}@{scale}"
    return str(sys_key)


def _iter_numeric_fields(row: dict[str, Any]) -> Iterable[tuple[str, float]]:
    """Yield ``(name, value)`` for every numeric leaf field we care about."""
    for k, v in row.items():
        if k in IGNORE_FIELDS:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            yield k, float(v)


def _load(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file and normalise to a list-of-row-dicts."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Some harnesses might wrap rows: {"rows": [...]}. Accept either.
        for key in ("rows", "results", "data"):
            if key in data and isinstance(data[key], list):
                return list(data[key])
        # Fall back to single-row treatment.
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"unexpected JSON shape in {path}: {type(data).__name__}")


def _compare_metric(
    *,
    dataset: str,
    row_key: str,
    metric: str,
    golden: float | None,
    current: float | None,
    tolerance_latency: float,
    tolerance_recall: float,
) -> Diff:
    """Classify one metric diff as pass / regression / new / missing."""
    if golden is None and current is not None:
        return Diff(dataset, row_key, metric, None, current, None, "new")
    if current is None and golden is not None:
        return Diff(dataset, row_key, metric, golden, None, None, "missing")
    assert golden is not None and current is not None

    if golden == 0:
        # Avoid divide-by-zero; only regress if we jumped way up in abs
        # terms. Fine for "disk_mb = 0" HTTP rows.
        delta_pct = 0.0 if current == 0 else float("inf")
    else:
        delta_pct = (current - golden) / abs(golden)

    status = "pass"
    if metric in LATENCY_FIELDS and delta_pct > tolerance_latency:
        # Higher = worse.
        status = "regression"
    elif metric in QUALITY_FIELDS and -delta_pct > tolerance_recall:
        # Lower = worse. Quality regression is negative delta.
        status = "regression"
    # Anything else: skip (don't gate on unknown metrics).
    return Diff(dataset, row_key, metric, golden, current, delta_pct, status)


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_row_key(r): r for r in rows}


def check_pair(
    *,
    current_path: Path,
    golden_path: Path,
    tolerance_latency: float,
    tolerance_recall: float,
) -> tuple[list[Diff], bool]:
    """Compare one current-vs-golden pair. Returns (diffs, created_golden).

    If ``golden_path`` does not exist, copy ``current_path`` to it and
    return an empty diff list with ``created_golden=True``.
    """
    current_path = Path(current_path)
    golden_path = Path(golden_path)

    if not current_path.exists():
        raise FileNotFoundError(f"--current file missing: {current_path}")

    if not golden_path.exists():
        # Bootstrap: seed golden from current.
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            current_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(
            f"[check-bench] {golden_path} did not exist; "
            f"seeded from {current_path}",
            file=sys.stderr,
        )
        return [], True

    current_rows = _index_rows(_load(current_path))
    golden_rows = _index_rows(_load(golden_path))

    diffs: list[Diff] = []
    dataset = current_path.stem

    all_keys = sorted(set(current_rows) | set(golden_rows))
    for key in all_keys:
        cur = current_rows.get(key)
        gold = golden_rows.get(key)
        if cur is None and gold is not None:
            diffs.append(Diff(dataset, key, "<row>", None, None, None, "missing"))
            continue
        if gold is None and cur is not None:
            # A new row entirely — classify every metric as "new".
            for metric, value in _iter_numeric_fields(cur):
                diffs.append(Diff(dataset, key, metric, None, value, None, "new"))
            continue
        assert cur is not None and gold is not None
        cur_metrics = dict(_iter_numeric_fields(cur))
        gold_metrics = dict(_iter_numeric_fields(gold))
        all_metrics = sorted(set(cur_metrics) | set(gold_metrics))
        for metric in all_metrics:
            diffs.append(
                _compare_metric(
                    dataset=dataset,
                    row_key=key,
                    metric=metric,
                    golden=gold_metrics.get(metric),
                    current=cur_metrics.get(metric),
                    tolerance_latency=tolerance_latency,
                    tolerance_recall=tolerance_recall,
                )
            )
    return diffs, False


def format_diff_table(diffs: list[Diff]) -> str:
    """Render a markdown diff table suitable for PR comment + stdout."""
    # Filter to interesting rows: any regression, plus one "pass"
    # sample per dataset just so the operator can eyeball the numbers.
    rows = [
        d
        for d in diffs
        if d.status in ("regression", "missing", "new")
    ]
    if not rows:
        return "All metrics within tolerance.\n"

    header = (
        "| Dataset | Row | Metric | Golden | Current | Δ | Status |"
    )
    sep = "| --- | --- | --- | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for d in rows:
        g = "-" if d.golden is None else f"{d.golden:.4g}"
        c = "-" if d.current is None else f"{d.current:.4g}"
        delta = (
            "-" if d.delta_pct is None else f"{d.delta_pct * 100:+.1f}%"
        )
        lines.append(
            f"| {d.dataset} | {d.row_key} | {d.metric} | {g} | {c} "
            f"| {delta} | {d.status} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare benchmark JSON against a golden snapshot.",
    )
    p.add_argument(
        "--current",
        action="append",
        required=True,
        type=Path,
        help="path to a current benchmark JSON (repeatable)",
    )
    p.add_argument(
        "--golden",
        action="append",
        required=True,
        type=Path,
        help="path to the matching golden JSON (repeatable)",
    )
    p.add_argument(
        "--tolerance-latency",
        type=float,
        default=0.20,
        help="allowed relative drift on latency/disk (default 0.20 = ±20%%)",
    )
    p.add_argument(
        "--tolerance-recall",
        type=float,
        default=0.05,
        help="allowed relative drift on recall/mrr/ndcg (default 0.05 = 5%%)",
    )
    args = p.parse_args(argv)

    if len(args.current) != len(args.golden):
        p.error(
            f"--current / --golden count mismatch: "
            f"{len(args.current)} vs {len(args.golden)}"
        )

    all_diffs: list[Diff] = []
    any_regression = False
    for cur_path, gold_path in zip(args.current, args.golden, strict=True):
        diffs, created = check_pair(
            current_path=cur_path,
            golden_path=gold_path,
            tolerance_latency=args.tolerance_latency,
            tolerance_recall=args.tolerance_recall,
        )
        if created:
            # Bootstrap — still record zero diffs, no regression.
            continue
        all_diffs.extend(diffs)
        if any(d.status in ("regression", "missing") for d in diffs):
            any_regression = True

    print(format_diff_table(all_diffs))
    if any_regression:
        print("FAIL: regression detected (see table above).", file=sys.stderr)
        return 1
    print("PASS: all metrics within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
