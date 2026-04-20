"""Check whether IDK answers correlate with low rank in retrieval.

Join the partial rank_probe chroma data (190 items) with the
qa_compare chroma strict jsonl to see: among items where chroma
retrieved gold, does the LLM IDK more often when gold is at rank
4-5 (truncation zone) vs rank 1-2?

If truncation is the mechanism, we expect monotonically increasing
IDK rate with rank.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


IDK_PHRASES = (
    "i don't know", "i do not know", "i don't have",
    "i do not have", "not in context", "not mentioned",
    "not available",
)


def _is_idk(hypothesis: str) -> bool:
    h = hypothesis.strip().lower()
    return any(p in h for p in IDK_PHRASES)


def main() -> None:
    rp_path = Path(
        "benchmarks/industry/longmemeval/results/"
        "rank_probe_chroma_cosine_n500_rank_probe.jsonl"
    )
    qa_path = Path(
        "benchmarks/industry/longmemeval/results/"
        "qa_compare_chroma_cosine_n500_strict.jsonl"
    )

    # Load rank probe: qid -> gold_rank
    rank_by_qid: dict[str, int] = {}
    for line in open(rp_path, encoding="utf-8"):
        r = json.loads(line)
        rank_by_qid[r["question_id"]] = r["gold_rank"]

    # Load QA: qid -> hypothesis + f1 info
    qa_by_qid: dict[str, dict] = {}
    for line in open(qa_path, encoding="utf-8"):
        r = json.loads(line)
        qa_by_qid[r["question_id"]] = r

    # Join on shared qids
    joined: list[tuple[int, str, str, str]] = []
    for qid, rank in rank_by_qid.items():
        if qid not in qa_by_qid:
            continue
        qa = qa_by_qid[qid]
        joined.append((rank, qa["hypothesis"], qa["answer"], qid))

    print(f"Joined {len(joined)} items (partial probe N={len(rank_by_qid)}, "
          f"QA N={len(qa_by_qid)})\n")

    # Group by rank
    by_rank: dict[int, list] = defaultdict(list)
    for rank, hyp, gold, qid in joined:
        by_rank[rank].append((hyp, gold, qid))

    print("## Chroma: IDK rate vs gold_rank\n")
    print("| gold_rank | N | IDK count | IDK rate |")
    print("| --- | ---: | ---: | ---: |")
    for r in sorted(by_rank):
        rows = by_rank[r]
        idks = sum(1 for hyp, _, _ in rows if _is_idk(hyp))
        label = f"{r}" if r > 0 else "0 (missed)"
        print(f"| {label} | {len(rows)} | {idks} | "
              f"{100.0*idks/len(rows):.1f}% |")


if __name__ == "__main__":
    main()
