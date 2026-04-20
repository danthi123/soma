"""Sweep hybrid_alpha on LongMemEval to find the optimal blend of BM25 and cosine.

alpha=0.0 => pure BM25
alpha=1.0 => pure cosine
alpha=0.3 => SOMA's current default (from LoCoMo ablation)

For each alpha, runs retrieval-only (no LLM) on all items and records
hit_at_5 + gold_rank. Gives us a per-type F1 surface map so we can pick
the best alpha per benchmark corpus.

Usage::

    python -m benchmarks.industry.longmemeval.alpha_sweep \\
        --variant small --limit 500 --alphas 0.0 0.1 0.2 0.3 0.5 0.7 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from benchmarks.industry.longmemeval.data_loader import load_dataset
from benchmarks.industry.longmemeval.run_qa_compare import (
    _build_sbert,
    _compute_gold_rank,
    _session_text,
)
from soma.memory import MemoryLayer

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


def run_alpha(
    alpha: float,
    items: list[Any],
    embed_fn,
    dim: int,
    top_k: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run SOMA with fixed alpha on all items."""
    rows: list[dict] = []
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add(r["question_id"])
                rows.append(r)
        logger.info("Resuming alpha=%.2f: %d already done", alpha, len(done))

    t_start = time.perf_counter()
    for i, item in enumerate(items):
        if item.question_id in done:
            continue
        raw_ids = item.haystack_session_ids
        uniq_ids = [f"{sid}__{j}" for j, sid in enumerate(raw_ids)]
        uniq_to_orig = {u: orig for u, orig in zip(uniq_ids, raw_ids, strict=True)}
        mem = MemoryLayer.ephemeral(embed_fn=embed_fn, embed_dim=dim)
        for j, sess in enumerate(item.haystack_sessions):
            sess_date = (
                item.haystack_dates[j] if j < len(item.haystack_dates) else ""
            )
            mem.store(_session_text(sess, sess_date),
                      metadata={"sid": uniq_ids[j]})
        t_r = time.perf_counter()
        hits = mem.retrieve(item.question, k=top_k, hybrid_alpha=alpha)
        retr_ms = (time.perf_counter() - t_r) * 1000
        retrieved_origs = [uniq_to_orig.get(h.metadata["sid"], h.metadata["sid"])
                           for h in hits]
        gold = set(item.answer_session_ids or [])
        gold_rank = _compute_gold_rank(retrieved_origs, gold, top_k)
        row = {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "alpha": alpha,
            "hit_at_k": 1 if gold_rank > 0 else 0,
            "gold_rank": gold_rank,
            "retrieval_ms": retr_ms,
        }
        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            logger.info("[alpha=%.2f] %d/%d | %.1fs", alpha, i + 1,
                        len(items), elapsed)

    total_s = time.perf_counter() - t_start
    hits = [r for r in rows if r["hit_at_k"] == 1]
    ranks = [r["gold_rank"] for r in hits]
    return {
        "alpha": alpha,
        "n": len(rows),
        "hit_rate": len(hits) / max(1, len(rows)),
        "mean_rank_given_hit": mean(ranks) if ranks else 0.0,
        "rank1_frac": sum(1 for r in ranks if r == 1) / max(1, len(rows)),
        "total_s": total_s,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--out-suffix", default="_n500_alpha_sweep")
    args = p.parse_args()

    items = load_dataset(args.variant, limit=args.limit)
    sbert, dim = _build_sbert(device="cpu")

    def embed_fn(t: str) -> torch.Tensor:
        return torch.tensor(sbert.encode(t, convert_to_numpy=True))

    summaries: list[dict[str, Any]] = []
    for alpha in args.alphas:
        out_path = RESULTS_DIR / f"alpha_sweep_alpha{alpha:.2f}{args.out_suffix}.jsonl"
        summary = run_alpha(alpha, items, embed_fn, dim,
                            top_k=args.top_k, out_path=out_path)
        summaries.append(summary)
        logger.info("alpha=%.2f summary: %s", alpha, json.dumps(summary, indent=2))

    summary_path = RESULTS_DIR / f"alpha_sweep_summary{args.out_suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Summary written to %s", summary_path)

    print("\n| alpha | N | R@5 | rank=1 frac | mean_rank_given_hit |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for s in summaries:
        print(f"| {s['alpha']:.2f} | {s['n']} | {s['hit_rate']:.4f} | "
              f"{s['rank1_frac']:.4f} | {s['mean_rank_given_hit']:.3f} |")


if __name__ == "__main__":
    main()
