"""Compare LLM-judge accuracy across two modes for a paired run.

Reads the *_judged_* jsonl files produced by judge_predictions.py and
produces headline accuracy tables + per-type breakdown + delta vs F1
(where F1 and judge disagree, we see the metric artifacts).

Usage::

    python -m scripts.analysis.compare_judge_accuracy \\
        --suffix _n500_strict_judged_qwen4b \\
        --mode-a chroma_cosine --mode-b soma_hybrid
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from benchmarks.industry.longmemeval.metrics import token_f1

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def _load_judged(mode: str, suffix: str) -> dict[str, dict]:
    path = RESULTS_DIR / f"qa_compare_{mode}{suffix}.jsonl"
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["f1"] = token_f1(r["hypothesis"], r["answer"])
            out[r["question_id"]] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="_n500_strict_judged_qwen4b")
    p.add_argument("--mode-a", default="chroma_cosine")
    p.add_argument("--mode-b", default="soma_hybrid")
    args = p.parse_args()

    a = _load_judged(args.mode_a, args.suffix)
    b = _load_judged(args.mode_b, args.suffix)

    print(f"# Judge-vs-F1 comparison ({args.mode_a} vs {args.mode_b})\n")
    print(f"Suffix: `{args.suffix}`\n")

    # Headline: judge accuracy + F1
    a_judge = mean(r["judge_correct"] for r in a.values())
    b_judge = mean(r["judge_correct"] for r in b.values())
    a_f1 = mean(r["f1"] for r in a.values())
    b_f1 = mean(r["f1"] for r in b.values())

    print("## Headline\n")
    print(f"| Mode | F1 | judge_accuracy |")
    print(f"| --- | ---: | ---: |")
    print(f"| {args.mode_a} | {a_f1:.4f} | {a_judge:.4f} |")
    print(f"| {args.mode_b} | {b_f1:.4f} | {b_judge:.4f} |")
    print()
    if a_f1 > 0 and a_judge > 0:
        print(f"**SOMA lift: F1 +{(b_f1/a_f1 - 1)*100:.1f}%  |  "
              f"judge-accuracy +{(b_judge/a_judge - 1)*100:.1f}%**\n")

    # Per-type breakdown
    print("## Per-type breakdown\n")
    shared = set(a.keys()) & set(b.keys())
    by_type: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "a_f1": 0.0, "b_f1": 0.0, "a_j": 0, "b_j": 0}
    )
    for qid in shared:
        t = a[qid]["question_type"]
        bucket = by_type[t]
        bucket["n"] += 1
        bucket["a_f1"] += a[qid]["f1"]
        bucket["b_f1"] += b[qid]["f1"]
        bucket["a_j"] += a[qid]["judge_correct"]
        bucket["b_j"] += b[qid]["judge_correct"]

    print("| Type | N | A F1 | B F1 | F1 lift | A judge | B judge | judge lift |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for t, v in sorted(by_type.items(), key=lambda kv: -kv[1]["n"]):
        n = v["n"]
        a_f1_m = v["a_f1"] / n
        b_f1_m = v["b_f1"] / n
        a_j_m = v["a_j"] / n
        b_j_m = v["b_j"] / n
        f1_lift = f"+{(b_f1_m/a_f1_m - 1)*100:.0f}%" if a_f1_m > 0 else "—"
        j_lift = f"+{(b_j_m/a_j_m - 1)*100:.0f}%" if a_j_m > 0 else "—"
        print(
            f"| {t} | {n} | {a_f1_m:.3f} | {b_f1_m:.3f} | {f1_lift} | "
            f"{a_j_m:.3f} | {b_j_m:.3f} | {j_lift} |"
        )

    # Judge-F1 agreement analysis
    print("\n## Where judge and F1 disagree\n")
    print("Items where judge says CORRECT but F1 is low (<0.5), or "
          "judge says WRONG but F1 is high (>0.5):\n")
    disagree_a_judge_right_f1_low = 0
    disagree_a_judge_wrong_f1_high = 0
    disagree_b_judge_right_f1_low = 0
    disagree_b_judge_wrong_f1_high = 0
    for qid in shared:
        ra, rb = a[qid], b[qid]
        if ra["judge_correct"] == 1 and ra["f1"] < 0.5:
            disagree_a_judge_right_f1_low += 1
        if ra["judge_correct"] == 0 and ra["f1"] > 0.5:
            disagree_a_judge_wrong_f1_high += 1
        if rb["judge_correct"] == 1 and rb["f1"] < 0.5:
            disagree_b_judge_right_f1_low += 1
        if rb["judge_correct"] == 0 and rb["f1"] > 0.5:
            disagree_b_judge_wrong_f1_high += 1
    print(f"| Mode | judge=1 & F1<0.5 (F1 underscored) | "
          f"judge=0 & F1>0.5 (F1 flattered) |")
    print(f"| --- | ---: | ---: |")
    print(f"| {args.mode_a} | {disagree_a_judge_right_f1_low} | "
          f"{disagree_a_judge_wrong_f1_high} |")
    print(f"| {args.mode_b} | {disagree_b_judge_right_f1_low} | "
          f"{disagree_b_judge_wrong_f1_high} |")
    print()
    print("High 'judge=1 & F1<0.5' count => the mode is giving "
          "semantically-correct but lexically-different answers. High "
          "'judge=0 & F1>0.5' => judge disagrees with F1 on overlap-heavy "
          "but wrong-meaning answers (rare).")


if __name__ == "__main__":
    main()
