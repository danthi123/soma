"""Analyze the output of run_qa_compare.py.

Reads the per-mode JSON results and produces:
1. Head-to-head comparison tables
2. R@K → F1 correlation
3. Cost-per-question in tokens
4. Win/loss breakdown (which mode answered correctly more often)

Usage::

    python -m benchmarks.industry.longmemeval.analyze_qa_compare \\
        --suffix _n100 \\
        --modes chroma_cosine chroma_rerank soma_hybrid full_context
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def _load_mode(mode: str, suffix: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"qa_compare_{mode}{suffix}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _headline_table(mode_data: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Mode | F1 | R@5 | input tok | retrieve+LLM ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, d in mode_data.items():
        total_ms = d["avg_retrieval_ms"] + d["avg_llm_ms"]
        lines.append(
            f"| {mode} | {d['scores']['f1']:.4f} | "
            f"{d['retrieval_recall_at_k']:.3f} | "
            f"{int(d['avg_input_tokens'])} | {total_ms:.0f} |"
        )
    return "\n".join(lines)


def _pairwise_win_loss(mode_data: dict[str, dict[str, Any]], suffix: str) -> str:
    """For each pair of modes, count items where one has strictly higher F1."""
    modes = list(mode_data.keys())
    # Build qid -> {mode: f1} map
    from benchmarks.industry.longmemeval.metrics import token_f1

    qid_f1: dict[str, dict[str, float]] = {}
    for mode in mode_data:
        # Recompute per-item F1 from predictions (need gold)
        # Load from jsonl which has both hyp and gold
        jsonl_path = RESULTS_DIR / f"qa_compare_{mode}{suffix}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    qid = row["question_id"]
                    f1 = token_f1(row["hypothesis"], row["answer"])
                    qid_f1.setdefault(qid, {})[mode] = f1

    lines = ["### Pairwise wins (one mode strictly higher F1)", ""]
    lines.append("| A vs B | A wins | B wins | tied |")
    lines.append("| --- | ---: | ---: | ---: |")
    for i, a in enumerate(modes):
        for b in modes[i+1:]:
            a_wins = b_wins = ties = 0
            for qid, scores in qid_f1.items():
                if a not in scores or b not in scores:
                    continue
                if scores[a] > scores[b]:
                    a_wins += 1
                elif scores[b] > scores[a]:
                    b_wins += 1
                else:
                    ties += 1
            lines.append(f"| {a} vs {b} | {a_wins} | {b_wins} | {ties} |")
    return "\n".join(lines)


def _cost_table(mode_data: dict[str, dict[str, Any]]) -> str:
    """Compare dollar cost per query assuming cloud LLM rates."""
    # gpt-4o-mini indicative: $0.15 / 1M input, $0.60 / 1M output
    # Use input-only for retrieval cost comparison
    input_price_per_mtok = 0.15  # $/1M input tokens
    output_tokens = 100  # rough average

    lines = [
        "### Cost per 1000 queries (gpt-4o-mini rates, $0.15/Mtok input)",
        "",
        "| Mode | avg input tok | $/1K queries (input) |",
        "| --- | ---: | ---: |",
    ]
    for mode, d in mode_data.items():
        tok = d["avg_input_tokens"]
        cost_per_1k = tok * 1000 * input_price_per_mtok / 1e6
        lines.append(f"| {mode} | {int(tok)} | ${cost_per_1k:.3f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="_n100")
    p.add_argument(
        "--modes", nargs="+",
        default=["chroma_cosine", "chroma_rerank", "soma_hybrid", "full_context"],
    )
    args = p.parse_args()

    mode_data: dict[str, dict[str, Any]] = {}
    for mode in args.modes:
        d = _load_mode(mode, args.suffix)
        if d is None:
            print(f"[skip] no data for {mode}{args.suffix}")
            continue
        mode_data[mode] = d

    if not mode_data:
        print("No data found.")
        return

    print("# QA comparison analysis")
    print()
    print("## Headline")
    print()
    print(_headline_table(mode_data))
    print()
    print("## Cost analysis")
    print()
    print(_cost_table(mode_data))
    print()
    print("## Per-question pairwise comparison")
    print()
    print(_pairwise_win_loss(mode_data, args.suffix))


if __name__ == "__main__":
    main()
