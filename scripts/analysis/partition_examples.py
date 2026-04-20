"""Pull concrete per-item examples from each retrieval partition cell.

For (0,1) only-SOMA-retrieves: show items where SOMA's F1 >> chroma's F1.
For (1,1) both-retrieve-gold: show items where SOMA wins F1 despite tied retrieval.
For (1,0) only-chroma-retrieves: show items where chroma's retrieval doesn't translate.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.industry.longmemeval.metrics import token_f1

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def _load(mode: str) -> dict[str, dict]:
    path = RESULTS_DIR / f"qa_compare_{mode}_n500_strict.jsonl"
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["f1"] = token_f1(row["hypothesis"], row["answer"])
            out[row["question_id"]] = row
    return out


def _need_gold_question(qid: str, variant: str = "s") -> tuple[str, str]:
    """Look up the question text + gold answer for a qid."""
    # data_loader returns tuples; easier to just load the raw json
    # in CLI usage we only need this for a few examples
    import gzip
    raw_path = Path("benchmarks/industry/longmemeval/data/longmemeval_s.json")
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            if item["question_id"] == qid:
                return item["question"], item["answer"]
    return "<question_text_not_loaded>", "<gold_not_loaded>"


def _fmt_row(qid: str, row: dict, label: str) -> str:
    return (
        f"- **{label}**: `{qid}` ({row['question_type']})\n"
        f"  - hypothesis: `{row['hypothesis'][:120]}`\n"
        f"  - gold answer: `{row['answer'][:120]}`\n"
        f"  - F1={row['f1']:.3f} | R@5_hit={row['hit_at_k']}\n"
    )


def main() -> None:
    a = _load("chroma_cosine")
    b = _load("soma_hybrid")

    cells: dict[str, list[tuple[str, dict, dict]]] = {
        "01": [], "10": [], "11_soma_wins": [], "11_chroma_wins": [],
    }
    for qid, row_a in a.items():
        if qid not in b:
            continue
        row_b = b[qid]
        if row_a["hit_at_k"] == 0 and row_b["hit_at_k"] == 1:
            cells["01"].append((qid, row_a, row_b))
        elif row_a["hit_at_k"] == 1 and row_b["hit_at_k"] == 0:
            cells["10"].append((qid, row_a, row_b))
        elif row_a["hit_at_k"] == 1 and row_b["hit_at_k"] == 1:
            if row_b["f1"] > row_a["f1"]:
                cells["11_soma_wins"].append((qid, row_a, row_b))
            elif row_a["f1"] > row_b["f1"]:
                cells["11_chroma_wins"].append((qid, row_a, row_b))

    def _top_delta(lst, k=5, soma_wins=True):
        lst = sorted(
            lst,
            key=lambda t: (t[2]["f1"] - t[1]["f1"]) * (1 if soma_wins else -1),
            reverse=True,
        )
        return lst[:k]

    print("# Concrete examples from each retrieval partition\n")

    print("## (0,1): only SOMA retrieves gold — top 5 F1 deltas\n")
    for qid, row_a, row_b in _top_delta(cells["01"], k=5, soma_wins=True):
        print(_fmt_row(qid, row_a, "chroma_cosine"))
        print(_fmt_row(qid, row_b, "soma_hybrid"))
        print(f"  - delta F1 = +{row_b['f1'] - row_a['f1']:.3f}\n")

    print("## (1,1) both retrieve gold, SOMA wins F1 — top 5 deltas\n")
    for qid, row_a, row_b in _top_delta(cells["11_soma_wins"], k=5, soma_wins=True):
        print(_fmt_row(qid, row_a, "chroma_cosine"))
        print(_fmt_row(qid, row_b, "soma_hybrid"))
        print(f"  - delta F1 = +{row_b['f1'] - row_a['f1']:.3f}\n")

    print("## (1,1) both retrieve gold, chroma wins F1 — top 5 deltas\n")
    for qid, row_a, row_b in _top_delta(cells["11_chroma_wins"], k=5, soma_wins=False):
        print(_fmt_row(qid, row_a, "chroma_cosine"))
        print(_fmt_row(qid, row_b, "soma_hybrid"))
        print(f"  - delta F1 = -{row_a['f1'] - row_b['f1']:.3f}\n")

    print("## (1,0): only chroma retrieves gold\n")
    for qid, row_a, row_b in cells["10"][:6]:
        print(_fmt_row(qid, row_a, "chroma_cosine"))
        print(_fmt_row(qid, row_b, "soma_hybrid"))
        print(f"  - delta F1 = {row_b['f1'] - row_a['f1']:+.3f}\n")


if __name__ == "__main__":
    main()
