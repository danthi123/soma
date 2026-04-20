"""Compute what fraction of SOMA's F1 lift comes from each retrieval-partition cell.

Runs on the n500_strict paired jsonl. For each of (0,0), (0,1), (1,0), (1,1):
- Sum(SOMA_f1 - chroma_f1) within the cell
- Report that as fraction of total F1 lift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.industry.longmemeval.metrics import token_f1

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def _load(mode: str, suffix: str) -> dict[str, dict]:
    path = RESULTS_DIR / f"qa_compare_{mode}{suffix}.jsonl"
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["f1"] = token_f1(row["hypothesis"], row["answer"])
            out[row["question_id"]] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="_n500_strict")
    args = p.parse_args()
    a = _load("chroma_cosine", args.suffix)
    b = _load("soma_hybrid", args.suffix)
    print(f"suffix={args.suffix}")

    cells: dict[str, list[tuple[str, float, float]]] = {
        "00": [], "01": [], "10": [], "11": [],
    }
    for qid, row_a in a.items():
        if qid not in b:
            continue
        row_b = b[qid]
        key = f"{row_a['hit_at_k']}{row_b['hit_at_k']}"
        cells[key].append((qid, row_a["f1"], row_b["f1"]))

    total_delta = 0.0
    cell_deltas: dict[str, float] = {}
    for k, rows in cells.items():
        delta = sum(fb - fa for _, fa, fb in rows)
        cell_deltas[k] = delta
        total_delta += delta

    print(f"Total F1 lift (soma - chroma summed): {total_delta:.3f}")
    print(f"Per-item mean F1 lift: {total_delta / 500:.4f}\n")
    print("| Cell | N | sum(SOMA_f1 - chroma_f1) | fraction of total lift |")
    print("| --- | ---: | ---: | ---: |")
    labels = {
        "00": "neither retrieves gold",
        "01": "only SOMA retrieves",
        "10": "only chroma retrieves",
        "11": "both retrieve gold",
    }
    for k in ["00", "01", "10", "11"]:
        n = len(cells[k])
        delta = cell_deltas[k]
        frac = 100.0 * delta / total_delta if total_delta else 0.0
        print(
            f"| ({k[0]},{k[1]}) {labels[k]} | {n} | {delta:+.3f} | {frac:+.1f}% |"
        )


if __name__ == "__main__":
    main()
