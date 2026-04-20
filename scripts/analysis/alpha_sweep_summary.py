"""Summarize the hybrid-alpha sweep jsonl outputs.

Aggregates rank-1 frac, hit@5 rate, mean rank, and per-question-type
breakdowns for each alpha in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.

Usage::

    python -m scripts.analysis.alpha_sweep_summary \\
        --glob 'benchmarks/industry/longmemeval/results/alpha_sweep_alpha*_n500_alpha_sweep.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_alpha(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    hits = [r for r in rows if r["hit_at_k"] == 1]
    ranks = [r["gold_rank"] for r in hits]
    rank1 = [r for r in hits if r["gold_rank"] == 1]
    per_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_type[r["question_type"]].append(r)

    by_type = {}
    for qtype, qs in per_type.items():
        qhits = [r for r in qs if r["hit_at_k"] == 1]
        qrank1 = [r for r in qhits if r["gold_rank"] == 1]
        by_type[qtype] = {
            "n": len(qs),
            "hit@5": len(qhits) / max(1, len(qs)),
            "rank1_frac": len(qrank1) / max(1, len(qs)),
            "mean_rank_given_hit": (
                sum(r["gold_rank"] for r in qhits) / len(qhits)
                if qhits else 0.0
            ),
        }

    return {
        "n": n,
        "hit@5": len(hits) / max(1, n),
        "rank1_frac": len(rank1) / max(1, n),
        "mean_rank_given_hit": sum(ranks) / len(ranks) if ranks else 0.0,
        "median_rank_given_hit": sorted(ranks)[len(ranks)//2] if ranks else 0,
        "by_type": by_type,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--glob",
        default="benchmarks/industry/longmemeval/results/alpha_sweep_alpha*_n500_alpha_sweep.jsonl",
    )
    p.add_argument("--out",
                   default="benchmarks/industry/longmemeval/results/alpha_sweep_summary_n500.md")
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"No files match {args.glob!r}")
        return

    summaries = []
    for path in paths:
        rows = _load_rows(path)
        if not rows:
            continue
        # alpha is in filename; split on "alpha" gives chunks, first
        # chunk after the prefix starts with the numeric value. Use
        # the row's ``alpha`` field as source of truth when available.
        alpha = rows[0].get("alpha")
        if alpha is None:
            stem = Path(path).stem
            # filename like alpha_sweep_alpha0.10_n500_alpha_sweep
            parts = stem.split("alpha")
            alpha_str = parts[-2].rstrip("_") if len(parts) >= 2 else ""
            try:
                alpha = float(alpha_str)
            except ValueError:
                continue
        summary = summarize_alpha(rows)
        summary["alpha"] = alpha
        summaries.append(summary)

    summaries.sort(key=lambda s: s["alpha"])

    lines = []
    lines.append("# Hybrid alpha sweep — N=500 LongMemEval small-variant\n")
    lines.append(
        "Retrieval-only; ``alpha`` blends BM25 and cosine "
        "(``alpha=0.0`` pure BM25, ``1.0`` pure cosine). Gold session rank "
        "probe.\n"
    )
    lines.append(
        "| alpha | N | hit@5 | rank=1 frac | mean rank | median rank |\n"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for s in summaries:
        lines.append(
            f"| {s['alpha']:.2f} | {s['n']} | {s['hit@5']:.4f} | "
            f"{s['rank1_frac']:.4f} | {s['mean_rank_given_hit']:.3f} | "
            f"{s['median_rank_given_hit']} |\n"
        )

    # Per-type rank-1 frac matrix
    all_types = sorted({t for s in summaries for t in s.get("by_type", {})})
    if all_types and summaries:
        lines.append("\n## Rank-1 fraction per question type\n")
        lines.append("| alpha | " + " | ".join(all_types) + " |\n")
        lines.append("| ---: | " + " | ".join(["---:" for _ in all_types]) + " |\n")
        for s in summaries:
            row = [f"{s['alpha']:.2f}"]
            for qtype in all_types:
                v = s["by_type"].get(qtype, {}).get("rank1_frac", 0.0)
                row.append(f"{v:.3f}")
            lines.append("| " + " | ".join(row) + " |\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    for line in lines:
        print(line, end="")


if __name__ == "__main__":
    main()
