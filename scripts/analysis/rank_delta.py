"""Compare per-item gold_rank between chroma and SOMA.

Reads rank_probe jsonls for both modes and for each question_id
computes (chroma_rank, soma_rank). Partitions into:
- both rank 1 (no change possible)
- SOMA improves rank (chroma > soma, both retrieve)
- SOMA degrades rank (chroma < soma, both retrieve)
- SOMA rescues (chroma missed, soma hit)
- SOMA loses (chroma hit, soma missed)
- both miss

This is the direct measurement of Mechanism 2 (ranking) from the
causation findings doc — does SOMA place gold higher than chroma
does, item-by-item?
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _load(mode: str, suffix: str) -> dict[str, dict]:
    path = Path(
        f"benchmarks/industry/longmemeval/results/"
        f"rank_probe_{mode}{suffix}.jsonl"
    )
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["question_id"]] = r
    return out


def main() -> None:
    suffix = "_n500_rank_probe"
    a = _load("chroma_cosine", suffix)
    b = _load("soma_hybrid", suffix)

    shared = sorted(set(a) & set(b))
    print(f"# Rank-delta: chroma vs SOMA  |  shared N={len(shared)}\n")

    # Scatter distribution
    pair_counts = Counter(
        (a[q]["gold_rank"], b[q]["gold_rank"]) for q in shared
    )

    chroma_miss_soma_hit = 0
    chroma_hit_soma_miss = 0
    both_miss = 0
    soma_rank_lower = 0  # soma > chroma (worse)
    soma_rank_same = 0   # soma == chroma
    soma_rank_higher = 0  # soma < chroma (better)

    for q in shared:
        ca, cb = a[q]["gold_rank"], b[q]["gold_rank"]
        if ca == 0 and cb == 0:
            both_miss += 1
        elif ca == 0 and cb > 0:
            chroma_miss_soma_hit += 1
        elif ca > 0 and cb == 0:
            chroma_hit_soma_miss += 1
        elif ca == cb:
            soma_rank_same += 1
        elif cb < ca:
            soma_rank_higher += 1
        else:
            soma_rank_lower += 1

    print("## Partition\n")
    print("| Cell | N | % |")
    print("| --- | ---: | ---: |")
    n = len(shared)
    for label, count in [
        ("both retrieve, same rank", soma_rank_same),
        ("SOMA places gold at lower (better) rank", soma_rank_higher),
        ("SOMA places gold at higher (worse) rank", soma_rank_lower),
        ("SOMA rescues (chroma miss, SOMA hit)", chroma_miss_soma_hit),
        ("SOMA loses (chroma hit, SOMA miss)", chroma_hit_soma_miss),
        ("both miss", both_miss),
    ]:
        print(f"| {label} | {count} | {100.0*count/n:.1f}% |")

    # Pair breakdown (chroma_rank, soma_rank)
    print("\n## Pair distribution (chroma_rank, soma_rank)\n")
    print("Rows = chroma rank, columns = SOMA rank. 0 = missed.\n")
    max_rank = max(max(pair_counts, key=lambda kv: max(kv))[0], 5)
    print("| chroma \\ soma | 0 | 1 | 2 | 3 | 4 | 5 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ca in range(0, max_rank + 1):
        row = f"| rank {ca} |"
        for cb in range(0, 6):
            row += f" {pair_counts.get((ca, cb), 0)} |"
        print(row)

    # Mean ranks
    chroma_ranks = [a[q]["gold_rank"] for q in shared if a[q]["gold_rank"] > 0]
    soma_ranks = [b[q]["gold_rank"] for q in shared if b[q]["gold_rank"] > 0]
    print(f"\nMean chroma rank (hits only): "
          f"{sum(chroma_ranks)/len(chroma_ranks):.3f}  "
          f"(N={len(chroma_ranks)})")
    print(f"Mean SOMA rank (hits only):   "
          f"{sum(soma_ranks)/len(soma_ranks):.3f}  "
          f"(N={len(soma_ranks)})")


if __name__ == "__main__":
    main()
