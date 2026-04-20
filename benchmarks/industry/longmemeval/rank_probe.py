"""Retrieval-only rank probe for LongMemEval.

Runs JUST the retrieval step (no LLM inference) on the LongMemEval
small variant (500 items) across two modes (chroma_cosine, soma_hybrid)
and logs the 1-indexed rank of the first gold session in top-k for
every item. This gives a direct answer to "does SOMA's hybrid score
place gold sessions higher than chroma's pure cosine does?"

Output: per-item jsonl with {question_id, question_type, hit, gold_rank}
plus a rank-distribution summary.

Usage::

    python -m benchmarks.industry.longmemeval.rank_probe \\
        --variant small --limit 500 --top-k 5 --sbert-device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from benchmarks.industry.longmemeval.data_loader import load_dataset
from benchmarks.industry.longmemeval.run_qa_compare import (
    _build_sbert,
    _compute_gold_rank,
    _make_index,
    _session_text,
)
from soma.memory.rerank import CrossEncoderReranker

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def probe_mode(
    mode: str,
    items: list[Any],
    sbert_model: Any,
    dim: int,
    embed_fn: Any,
    reranker: CrossEncoderReranker,
    top_k: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run retrieval-only probe for one mode across all items."""
    rows: list[dict[str, Any]] = []
    t_total = time.perf_counter()

    # Resume support
    done_qids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done_qids.add(row["question_id"])
                rows.append(row)
        logger.info("Resuming %s: %d items already done", mode, len(done_qids))

    for i, item in enumerate(items):
        if item.question_id in done_qids:
            continue
        raw_ids = item.haystack_session_ids
        uniq_ids = [f"{sid}__{j}" for j, sid in enumerate(raw_ids)]
        uniq_to_orig = {u: orig for u, orig in zip(uniq_ids, raw_ids, strict=True)}
        texts = [
            _session_text(
                sess,
                item.haystack_dates[j] if j < len(item.haystack_dates) else "",
            )
            for j, sess in enumerate(item.haystack_sessions)
        ]
        idx = _make_index(
            mode, sbert_model, dim, embed_fn, reranker,
            max_context_tokens=3800,
        )
        idx.ingest(texts, uniq_ids)
        t0 = time.perf_counter()
        retrieved = idx.retrieve(item.question, top_k)
        retr_ms = (time.perf_counter() - t0) * 1000
        retrieved_origs = [uniq_to_orig.get(sid, sid) for sid, _ in retrieved]
        gold = set(item.answer_session_ids or [])
        gold_rank = _compute_gold_rank(retrieved_origs, gold, top_k)
        hit = 1 if gold_rank > 0 else 0

        row = {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "hit_at_k": hit,
            "gold_rank": gold_rank,
            "retrieval_ms": retr_ms,
        }
        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 25 == 0:
            logger.info(
                "[%s] %d/%d hits_so_far=%d last_rank=%d",
                mode, i + 1, len(items),
                sum(r["hit_at_k"] for r in rows), gold_rank,
            )

    total_ms = (time.perf_counter() - t_total) * 1000
    hits = [r for r in rows if r["hit_at_k"] == 1]
    ranks = [r["gold_rank"] for r in hits]
    return {
        "mode": mode,
        "n": len(rows),
        "hit_rate": len(hits) / max(1, len(rows)),
        "mean_rank_given_hit": mean(ranks) if ranks else 0.0,
        "median_rank_given_hit": median(ranks) if ranks else 0.0,
        "rank_hist": {
            str(k): sum(1 for r in ranks if r == k) for k in range(1, 6)
        },
        "total_ms": total_ms,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--sbert-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out-suffix", default="_n500_rank_probe")
    p.add_argument(
        "--modes", nargs="+",
        default=["chroma_cosine", "soma_hybrid"],
    )
    args = p.parse_args()

    items = load_dataset(args.variant, limit=args.limit)
    logger.info("Loaded %d items from LongMemEval %s", len(items), args.variant)

    sbert_model, dim = _build_sbert(device=args.sbert_device)

    def embed_fn(text: str) -> torch.Tensor:
        return torch.tensor(sbert_model.encode(text, convert_to_numpy=True))

    reranker = CrossEncoderReranker()

    summaries: dict[str, Any] = {}
    for mode in args.modes:
        out_path = RESULTS_DIR / f"rank_probe_{mode}{args.out_suffix}.jsonl"
        summary = probe_mode(
            mode, items, sbert_model, dim, embed_fn, reranker,
            top_k=args.top_k, out_path=out_path,
        )
        summaries[mode] = summary
        logger.info("%s summary: %s", mode, json.dumps(summary, indent=2))

    summary_path = RESULTS_DIR / f"rank_probe_summary{args.out_suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
