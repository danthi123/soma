"""Compare SOMA vs chroma 9b strict judge accuracy with per-type breakdown.

Mirror of ``compare_judge_accuracy.py`` but pinned to the N=500 paired
qwen9b-strict runs that were judged by qwen4b.

Usage::

    python -m scripts.analysis.compare_judge_accuracy_9b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["question_id"]] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--soma",
        default="benchmarks/industry/longmemeval/results/qa_compare_soma_hybrid_n500_qwen9b_strict_judged_qwen4b.jsonl",
    )
    p.add_argument(
        "--chroma",
        default="benchmarks/industry/longmemeval/results/qa_compare_chroma_cosine_n500_qwen9b_strict_judged_qwen4b.jsonl",
    )
    p.add_argument(
        "--out",
        default="research/developmental/results/longmemeval_judge_findings_qwen9b.md",
    )
    args = p.parse_args()

    soma = _load(args.soma)
    chroma = _load(args.chroma)
    shared = set(soma) & set(chroma)

    # Overall
    soma_correct = sum(soma[q]["judge_correct"] for q in shared)
    chroma_correct = sum(chroma[q]["judge_correct"] for q in shared)
    n = len(shared)
    soma_acc = soma_correct / n
    chroma_acc = chroma_correct / n

    # Per-type
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "s": 0, "c": 0})
    for q in shared:
        qt = soma[q]["question_type"]
        by_type[qt]["n"] += 1
        by_type[qt]["s"] += soma[q]["judge_correct"]
        by_type[qt]["c"] += chroma[q]["judge_correct"]

    lines: list[str] = []
    lines.append(
        "# LongMemEval — qwen9b strict predictions, 4b judge (N=500)\n\n"
    )
    lines.append(
        f"Judged {n} paired items. Judge model: qwen3.5:4b-q8_0.\n\n"
    )
    lines.append("## Headline\n\n")
    lines.append("| System | Judge-accuracy |\n")
    lines.append("| --- | ---: |\n")
    lines.append(f"| chroma cosine @ qwen9b | {chroma_acc:.4f} |\n")
    lines.append(f"| **SOMA hybrid @ qwen9b** | **{soma_acc:.4f}** |\n")
    lift = soma_acc - chroma_acc
    rel = lift / max(chroma_acc, 1e-9)
    lines.append(f"| Delta | {lift:+.4f} ({rel:+.2%}) |\n\n")

    lines.append("## Per-question-type accuracy\n\n")
    lines.append(
        "| Type | N | chroma | SOMA | abs lift | rel lift |\n"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for qt in sorted(by_type):
        d = by_type[qt]
        n_t = d["n"]
        s = d["s"] / n_t
        c = d["c"] / n_t
        abs_lift = s - c
        rel_lift = abs_lift / max(c, 1e-9)
        lines.append(
            f"| {qt} | {n_t} | {c:.3f} | {s:.3f} | "
            f"{abs_lift:+.3f} | {rel_lift:+.2%} |\n"
        )

    # Disagreement analysis
    lines.append("\n## Where SOMA and chroma disagree\n\n")
    wins_soma = [q for q in shared
                 if soma[q]["judge_correct"] == 1 and chroma[q]["judge_correct"] == 0]
    wins_chroma = [q for q in shared
                   if soma[q]["judge_correct"] == 0 and chroma[q]["judge_correct"] == 1]
    ties_right = [q for q in shared
                  if soma[q]["judge_correct"] == 1 and chroma[q]["judge_correct"] == 1]
    ties_wrong = [q for q in shared
                  if soma[q]["judge_correct"] == 0 and chroma[q]["judge_correct"] == 0]
    lines.append(
        f"- SOMA correct, chroma wrong: **{len(wins_soma)}**\n"
        f"- Chroma correct, SOMA wrong: **{len(wins_chroma)}**\n"
        f"- Both correct: {len(ties_right)}\n"
        f"- Both wrong: {len(ties_wrong)}\n"
    )
    lines.append(
        f"\nNet items SOMA correct where chroma isn't: "
        f"**{len(wins_soma) - len(wins_chroma):+d}**\n"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}\n")
    for line in lines:
        print(line, end="")


if __name__ == "__main__":
    main()
