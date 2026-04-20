"""Compute aggregate F1/EM/R@5 + per-type breakdown from a qa_compare jsonl.

Used for sanity-checking a partial run before summary.json is generated,
or for re-scoring after run_qa_compare exits without writing the summary.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from benchmarks.industry.longmemeval.metrics import exact_match, token_f1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, type=Path)
    args = p.parse_args()

    rows: list[dict] = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    f1s: list[float] = []
    ems: list[int] = []
    hits: list[int] = []
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        f1 = token_f1(r["hypothesis"], r["answer"])
        em = 1 if exact_match(r["hypothesis"], r["answer"]) else 0
        f1s.append(f1)
        ems.append(em)
        hits.append(int(r.get("hit_at_k", 0)))
        r["_f1"] = f1
        r["_em"] = em
        by_type[r["question_type"]].append(r)

    print(f"# {args.jsonl.name}")
    print(f"\nN = {len(rows)}")
    print(f"F1 = {mean(f1s):.4f}")
    print(f"EM = {mean(ems):.4f}")
    print(f"R@5 = {mean(hits):.4f}")

    print("\n## Per-type breakdown\n")
    print("| Type | N | F1 | EM | R@5 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for t in sorted(by_type, key=lambda k: -len(by_type[k])):
        trows = by_type[t]
        print(
            f"| {t} | {len(trows)} | "
            f"{mean(r['_f1'] for r in trows):.4f} | "
            f"{mean(r['_em'] for r in trows):.4f} | "
            f"{mean(r.get('hit_at_k', 0) for r in trows):.4f} |"
        )


if __name__ == "__main__":
    main()
