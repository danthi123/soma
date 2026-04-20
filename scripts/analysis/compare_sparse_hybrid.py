#!/usr/bin/env python3
"""Paired analysis of sparse-retrieval variants on LongMemEval.

Reads the per-item jsonl files produced by
``benchmarks/industry/longmemeval/run_sparse_retrieval.py`` and
computes per-question-type deltas of rank-1 frac and hit@5, plus a
paired breakdown (where does hybrid_plus_sparse win vs lose vs tie
relative to hybrid_only).

Usage::

    python -m scripts.analysis.compare_sparse_hybrid \\
        --suffix _n100_sparse
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def load_rows(variant: str, suffix: str) -> list[dict]:
    path = RESULTS_DIR / f"sparse_retrieval_{variant}{suffix}.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="_n100_sparse")
    parser.add_argument("--variants", nargs="+",
                        default=["hybrid_only", "hybrid_plus_sparse", "sparse_only"])
    args = parser.parse_args()

    rows_by_variant = {v: load_rows(v, args.suffix) for v in args.variants}
    for v, rows in rows_by_variant.items():
        print(f"[{v}] loaded {len(rows)} rows")

    # Index by question_id for paired comparison.
    by_qid = {v: {r["question_id"]: r for r in rows} for v, rows in rows_by_variant.items()}
    common = set(by_qid["hybrid_only"].keys())
    for v in args.variants[1:]:
        common &= set(by_qid[v].keys())
    common = sorted(common)
    print(f"\nCommon items across all {len(args.variants)} variants: {len(common)}")

    # Aggregate overall + per-type.
    types = defaultdict(list)
    for qid in common:
        types[by_qid["hybrid_only"][qid]["question_type"]].append(qid)

    def summarise(variant: str, qids: list[str]) -> dict[str, float]:
        rs = [by_qid[variant][q] for q in qids]
        if not rs:
            return {"n": 0}
        hits = [r for r in rs if r["hit_at_k"] == 1]
        r1 = sum(1 for r in rs if r["gold_rank"] == 1)
        return {
            "n": len(rs),
            "hit": len(hits) / len(rs),
            "r1": r1 / len(rs),
            "mean_rank_hit": (sum(r["gold_rank"] for r in hits) / len(hits)) if hits else 0.0,
        }

    print("\n" + "=" * 96)
    print(f"{'segment':<30s} | {'N':>4s} | " + " | ".join(
        f"{v[:18]:>18s}" for v in args.variants
    ))
    print("-" * 96)

    for seg_name, qids in [("ALL", common)] + sorted(types.items()):
        print(f"{seg_name:<30s} | {len(qids):>4d}", end="")
        for v in args.variants:
            s = summarise(v, qids)
            print(f" | hit={s['hit']:.3f} r1={s['r1']:.3f}", end="")
        print()
    print("=" * 96)

    # Paired hybrid_only vs hybrid_plus_sparse (the key comparison).
    if "hybrid_plus_sparse" in by_qid:
        wins, losses, ties = 0, 0, 0
        for qid in common:
            r_hyb = by_qid["hybrid_only"][qid]["gold_rank"]
            r_sp = by_qid["hybrid_plus_sparse"][qid]["gold_rank"]
            # gold_rank=0 means miss. Lower positive rank = better.
            # miss vs hit: hit is strictly better. Within hits, lower rank better.
            if r_hyb == r_sp:
                ties += 1
            elif r_sp > 0 and (r_hyb == 0 or r_sp < r_hyb):
                wins += 1
            else:
                losses += 1
        print(f"\nPaired comparison (hybrid_plus_sparse vs hybrid_only):")
        print(f"  sparse wins:  {wins}  ({wins/len(common)*100:.1f}%)")
        print(f"  sparse loses: {losses} ({losses/len(common)*100:.1f}%)")
        print(f"  ties:         {ties}   ({ties/len(common)*100:.1f}%)")
        print(f"  net:          {wins - losses:+d}")


if __name__ == "__main__":
    main()
