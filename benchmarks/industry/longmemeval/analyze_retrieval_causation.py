"""Analyze the *causal* link between retrieval quality and F1 on paired runs.

For two modes (A and B) that ran the SAME questions:
- Partition items by (A_hit, B_hit) in {00, 01, 10, 11}
- Within each cell, compare F1 and see who wins, ties, or both fail
- Report whether F1 wins track retrieval wins

This answers: "when SOMA retrieves the gold and chroma doesn't, does SOMA
actually produce a better answer? Or does the LLM hallucinate the same
wrong thing either way?"

Usage::

    python -m benchmarks.industry.longmemeval.analyze_retrieval_causation \\
        --suffix _n500_strict \\
        --mode-a chroma_cosine --mode-b soma_hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.industry.longmemeval.metrics import token_f1

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def _load_jsonl(mode: str, suffix: str) -> dict[str, dict[str, Any]]:
    path = RESULTS_DIR / f"qa_compare_{mode}{suffix}.jsonl"
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["f1"] = token_f1(row["hypothesis"], row["answer"])
            out[row["question_id"]] = row
    return out


def _partition(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    """Return cells keyed by (a_hit, b_hit) in {'00','01','10','11'}."""
    cells: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {
        "00": [],
        "01": [],
        "10": [],
        "11": [],
    }
    for qid, row_a in a.items():
        row_b = b.get(qid)
        if row_b is None:
            continue
        key = f"{row_a['hit_at_k']}{row_b['hit_at_k']}"
        cells[key].append((qid, row_a, row_b))
    return cells


def _summarize_cell(
    label: str,
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]],
    mode_a: str,
    mode_b: str,
) -> str:
    n = len(rows)
    if n == 0:
        return f"- **{label}**: 0 items\n"
    a_wins = b_wins = ties_correct = ties_fail = 0
    a_f1s: list[float] = []
    b_f1s: list[float] = []
    for _, row_a, row_b in rows:
        fa, fb = row_a["f1"], row_b["f1"]
        a_f1s.append(fa)
        b_f1s.append(fb)
        if fa > fb:
            a_wins += 1
        elif fb > fa:
            b_wins += 1
        elif fa > 0.5:
            ties_correct += 1
        else:
            ties_fail += 1
    return (
        f"- **{label}**: {n} items | "
        f"{mode_a} F1={mean(a_f1s):.3f} vs {mode_b} F1={mean(b_f1s):.3f} | "
        f"{mode_a} wins={a_wins} {mode_b} wins={b_wins} "
        f"tied-correct={ties_correct} tied-fail={ties_fail}\n"
    )


def _per_type_win_rate(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    mode_a: str,
    mode_b: str,
) -> str:
    by_type: dict[str, dict[str, int]] = {}
    for qid, row_a in a.items():
        row_b = b.get(qid)
        if row_b is None:
            continue
        t = row_a["question_type"]
        bucket = by_type.setdefault(
            t, {"a_wins": 0, "b_wins": 0, "tied": 0, "n": 0}
        )
        bucket["n"] += 1
        if row_a["f1"] > row_b["f1"]:
            bucket["a_wins"] += 1
        elif row_b["f1"] > row_a["f1"]:
            bucket["b_wins"] += 1
        else:
            bucket["tied"] += 1

    lines = [
        f"| Question type | N | {mode_a} wins | {mode_b} wins | tied | "
        f"{mode_b} win advantage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t, v in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        adv = v["b_wins"] - v["a_wins"]
        sign = "+" if adv >= 0 else ""
        lines.append(
            f"| {t} | {v['n']} | {v['a_wins']} | {v['b_wins']} | "
            f"{v['tied']} | {sign}{adv} |"
        )
    return "\n".join(lines)


def _hit_delta_correlation(
    a: dict[str, dict[str, Any]],
    b: dict[str, dict[str, Any]],
    mode_a: str,
    mode_b: str,
) -> str:
    """For items where retrieval differed, did F1 follow retrieval?"""
    a_retr_only = [
        qid for qid, row_a in a.items()
        if qid in b and row_a["hit_at_k"] == 1 and b[qid]["hit_at_k"] == 0
    ]
    b_retr_only = [
        qid for qid, row_a in a.items()
        if qid in b and row_a["hit_at_k"] == 0 and b[qid]["hit_at_k"] == 1
    ]

    def _f1_win_rate(qids: list[str], winner: str) -> tuple[int, int, int]:
        a_wins = b_wins = ties = 0
        for qid in qids:
            fa, fb = a[qid]["f1"], b[qid]["f1"]
            if fa > fb:
                a_wins += 1
            elif fb > fa:
                b_wins += 1
            else:
                ties += 1
        return a_wins, b_wins, ties

    lines = [
        "### Does F1 follow retrieval?",
        "",
        f"- When only **{mode_a}** retrieves: N={len(a_retr_only)}",
    ]
    if a_retr_only:
        aw, bw, t = _f1_win_rate(a_retr_only, mode_a)
        lines.append(
            f"  - {mode_a} F1 wins={aw}, {mode_b} F1 wins={bw}, tied={t} "
            f"(retrieval advantage translated: {aw}/{len(a_retr_only)} "
            f"= {100.0*aw/len(a_retr_only):.1f}%)"
        )
    lines.append(f"- When only **{mode_b}** retrieves: N={len(b_retr_only)}")
    if b_retr_only:
        aw, bw, t = _f1_win_rate(b_retr_only, mode_b)
        lines.append(
            f"  - {mode_a} F1 wins={aw}, {mode_b} F1 wins={bw}, tied={t} "
            f"(retrieval advantage translated: {bw}/{len(b_retr_only)} "
            f"= {100.0*bw/len(b_retr_only):.1f}%)"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="_n500_strict")
    p.add_argument("--mode-a", default="chroma_cosine")
    p.add_argument("--mode-b", default="soma_hybrid")
    args = p.parse_args()

    a = _load_jsonl(args.mode_a, args.suffix)
    b = _load_jsonl(args.mode_b, args.suffix)

    print(f"# Retrieval-to-F1 causal analysis ({args.mode_a} vs {args.mode_b})")
    print(f"\nSuffix: `{args.suffix}`  |  N_a={len(a)}  N_b={len(b)}\n")

    print("## Retrieval x F1 partition")
    print()
    cells = _partition(a, b)
    print("(a_hit, b_hit) -> (0,0) = neither retrieved gold, etc.")
    print()
    print(_summarize_cell(
        "(0,0) neither retrieved gold", cells["00"], args.mode_a, args.mode_b))
    print(_summarize_cell(
        f"(0,1) only {args.mode_b} retrieved", cells["01"],
        args.mode_a, args.mode_b))
    print(_summarize_cell(
        f"(1,0) only {args.mode_a} retrieved", cells["10"],
        args.mode_a, args.mode_b))
    print(_summarize_cell(
        "(1,1) both retrieved gold", cells["11"], args.mode_a, args.mode_b))

    print("\n## Per-type F1 win distribution\n")
    print(_per_type_win_rate(a, b, args.mode_a, args.mode_b))

    print("\n## Retrieval -> F1 causation check\n")
    print(_hit_delta_correlation(a, b, args.mode_a, args.mode_b))


if __name__ == "__main__":
    main()
