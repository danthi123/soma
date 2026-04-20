"""Count "I don't know" responses by mode and retrieval outcome.

If SOMA's ranking mechanism is real, we should see chroma say
"I don't know" MORE often than SOMA on items where BOTH retrieve
the gold session — because chroma buries gold deeper in the context.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

IDK_PHRASES = (
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "not in context",
    "not mentioned",
    "not available",
)


def _is_idk(hypothesis: str) -> bool:
    h = hypothesis.strip().lower()
    return any(p in h for p in IDK_PHRASES)


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["question_id"]] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="_n500_strict")
    args = p.parse_args()

    a_path = Path(
        f"benchmarks/industry/longmemeval/results/"
        f"qa_compare_chroma_cosine{args.suffix}.jsonl"
    )
    b_path = Path(
        f"benchmarks/industry/longmemeval/results/"
        f"qa_compare_soma_hybrid{args.suffix}.jsonl"
    )
    a = _load(a_path)
    b = _load(b_path)

    print(f"# 'I don't know' analysis ({args.suffix})\n")

    # Global IDK rate
    a_idk = sum(1 for r in a.values() if _is_idk(r["hypothesis"]))
    b_idk = sum(1 for r in b.values() if _is_idk(r["hypothesis"]))
    print(f"- chroma_cosine IDK rate: {a_idk}/{len(a)} = {100.0*a_idk/len(a):.1f}%")
    print(f"- soma_hybrid  IDK rate: {b_idk}/{len(b)} = {100.0*b_idk/len(b):.1f}%\n")

    # By retrieval-partition cell
    cells: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "a_idk": 0, "b_idk": 0}
    )
    for qid, row_a in a.items():
        if qid not in b:
            continue
        row_b = b[qid]
        key = f"({row_a['hit_at_k']},{row_b['hit_at_k']})"
        cells[key]["n"] += 1
        if _is_idk(row_a["hypothesis"]):
            cells[key]["a_idk"] += 1
        if _is_idk(row_b["hypothesis"]):
            cells[key]["b_idk"] += 1

    print("## IDK rate by retrieval cell\n")
    print("| cell (chroma_hit, soma_hit) | N | chroma IDK | soma IDK |")
    print("| --- | ---: | ---: | ---: |")
    for key in ["(1,1)", "(1,0)", "(0,1)", "(0,0)"]:
        if key not in cells:
            continue
        v = cells[key]
        n = v["n"]
        a_pct = 100.0 * v["a_idk"] / n if n else 0.0
        b_pct = 100.0 * v["b_idk"] / n if n else 0.0
        print(
            f"| {key} | {n} | "
            f"{v['a_idk']} ({a_pct:.1f}%) | "
            f"{v['b_idk']} ({b_pct:.1f}%) |"
        )


if __name__ == "__main__":
    main()
