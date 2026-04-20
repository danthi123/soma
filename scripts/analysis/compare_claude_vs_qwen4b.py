"""Compare SOMA lift under Claude vs qwen4b on LongMemEval predictions.

Reads the paired jsonl predictions for:

- ``qa_compare_{mode}_n500_strict.jsonl`` (qwen4b baseline)
- ``qa_compare_{mode}_n500_claude_strict.jsonl`` (Claude run)

For each mode (chroma_rerank, soma_hybrid), computes:

- F1, EM, rouge per LLM
- Paired count: items where Claude is right / qwen4b is wrong, etc.
- Per-question-type breakdown of lift

Outputs a markdown table suitable for dropping into a findings doc.

Usage::

    python -m scripts.analysis.compare_claude_vs_qwen4b \\
        --qwen4b-prefix qa_compare \\
        --claude-prefix qa_compare \\
        --qwen4b-suffix _n500_strict \\
        --claude-suffix _n500_claude_strict

All files are read from
``benchmarks/industry/longmemeval/results/``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def _load_by_qid(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["question_id"]] = r
    return rows


def _exact_match(hyp: str, gold: str) -> int:
    return 1 if hyp.strip().lower() == gold.strip().lower() else 0


def _token_f1(hyp: str, gold: str) -> float:
    """Minimal token-F1 (same as metrics.py computes)."""
    h = hyp.strip().lower().split()
    g = gold.strip().lower().split()
    if not h or not g:
        return 0.0
    common = {}
    gh = dict((w, h.count(w)) for w in set(h))
    gg = dict((w, g.count(w)) for w in set(g))
    for w in gh:
        common[w] = min(gh[w], gg.get(w, 0))
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(h)
    r = n_common / len(g)
    return 2 * p * r / (p + r)


def summarize_mode(rows: dict[str, dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    ems = [_exact_match(r["hypothesis"], r["answer"]) for r in rows.values()]
    f1s = [_token_f1(r["hypothesis"], r["answer"]) for r in rows.values()]
    hits = [r.get("hit_at_k", 0) for r in rows.values()]

    # Per-type breakdown
    by_type: dict[str, dict] = defaultdict(lambda: {"n": 0, "f1": [], "em": []})
    for r in rows.values():
        t = r.get("question_type", "unknown")
        by_type[t]["n"] += 1
        by_type[t]["f1"].append(_token_f1(r["hypothesis"], r["answer"]))
        by_type[t]["em"].append(_exact_match(r["hypothesis"], r["answer"]))

    per_type = {
        t: {
            "n": v["n"],
            "f1": sum(v["f1"]) / v["n"],
            "em": sum(v["em"]) / v["n"],
        }
        for t, v in by_type.items()
    }

    return {
        "n": n,
        "f1": sum(f1s) / n,
        "em": sum(ems) / n,
        "hit@5": sum(hits) / n,
        "by_type": per_type,
    }


def compare_modes(
    qwen4b_files: dict[str, Path],
    claude_files: dict[str, Path],
) -> dict:
    """Compare (qwen4b, claude) × (chroma, soma) = 4 cells."""
    out = {}
    for llm, files in [("qwen4b", qwen4b_files), ("claude", claude_files)]:
        out[llm] = {}
        for mode, path in files.items():
            rows = _load_by_qid(path)
            out[llm][mode] = summarize_mode(rows)
    return out


def build_markdown(comp: dict, modes: list[str]) -> str:
    """Build a side-by-side markdown table."""
    lines = [
        "# LongMemEval QA — Claude vs qwen4b on identical retrieval",
        "",
    ]
    # Headline
    lines.append("## Headline F1")
    lines.append("")
    lines.append("| Mode | qwen4b F1 | Claude F1 | Claude/qwen4b |")
    lines.append("| --- | ---: | ---: | ---: |")
    for mode in modes:
        q = comp["qwen4b"].get(mode) or {}
        c = comp["claude"].get(mode) or {}
        qf1 = q.get("f1", 0.0)
        cf1 = c.get("f1", 0.0)
        qn = q.get("n", 0)
        cn = c.get("n", 0)
        ratio = cf1 / qf1 if qf1 else 0
        lines.append(
            f"| {mode} (N_q={qn}, N_c={cn}) | "
            f"{qf1:.4f} | {cf1:.4f} | +{(ratio - 1) * 100:.1f}% |"
        )
    lines.append("")
    # SOMA lift per LLM
    if "chroma_rerank" in modes and "soma_hybrid" in modes:
        lines.append("## SOMA lift (soma_hybrid / chroma_rerank)")
        lines.append("")
        lines.append("| LLM | chroma F1 | SOMA F1 | Relative lift |")
        lines.append("| --- | ---: | ---: | ---: |")
        for llm in ["qwen4b", "claude"]:
            c = comp[llm].get("chroma_rerank") or {}
            s = comp[llm].get("soma_hybrid") or {}
            cf1 = c.get("f1", 0.0)
            sf1 = s.get("f1", 0.0)
            lift = (sf1 / cf1 - 1) * 100 if cf1 else 0
            lines.append(
                f"| {llm} | {cf1:.4f} | {sf1:.4f} | **+{lift:.1f}%** |"
            )
        lines.append("")
    # Per-type F1 for Claude + SOMA
    lines.append("## Per-type F1 (Claude)")
    lines.append("")
    # Gather all types
    types = set()
    for llm in comp.values():
        for mode_result in llm.values():
            if "by_type" in mode_result:
                types.update(mode_result["by_type"].keys())
    header = "| Type | " + " | ".join(modes) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(modes)) + " |"
    lines.append(header)
    lines.append(sep)
    for t in sorted(types):
        row = [t]
        for mode in modes:
            pts = comp["claude"].get(mode, {}).get("by_type", {}).get(t)
            if pts:
                row.append(f"{pts['f1']:.3f} (n={pts['n']})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen4b-prefix", default="qa_compare")
    p.add_argument("--claude-prefix", default="qa_compare")
    p.add_argument("--qwen4b-suffix", default="_n500_strict")
    p.add_argument("--claude-suffix", default="_n500_claude_strict")
    p.add_argument("--modes", nargs="+",
                   default=["chroma_rerank", "soma_hybrid"])
    p.add_argument("--out", default="benchmarks/industry/longmemeval/"
                                    "results/claude_vs_qwen4b_summary.md")
    args = p.parse_args()

    qwen4b_files = {
        m: RESULTS_DIR / f"{args.qwen4b_prefix}_{m}{args.qwen4b_suffix}.jsonl"
        for m in args.modes
    }
    claude_files = {
        m: RESULTS_DIR / f"{args.claude_prefix}_{m}{args.claude_suffix}.jsonl"
        for m in args.modes
    }

    comp = compare_modes(qwen4b_files, claude_files)
    md = build_markdown(comp, args.modes)

    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
