"""LoCoMo retrieval-only rank probe: SOMA hybrid vs chroma cosine.

Mirror of LongMemEval's ``rank_probe.py`` but on LoCoMo. For each QA
pair, ingest all turns of the conversation into a MemoryLayer, then
retrieve and measure whether any evidence turn appears in the top-k
and at what rank.

LoCoMo QA pairs have an ``evidence`` field listing the dia_ids that
should be retrieved. Hit = any evidence dia_id appears in top-k.
Gold rank = position of the FIRST evidence turn in top-k.

Usage::

    python -m benchmarks.industry.locomo.rank_probe --mode soma_hybrid --limit 3
    python -m benchmarks.industry.locomo.rank_probe --mode chroma_cosine --limit 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.industry.locomo.data_loader import load_dataset
from soma.memory import MemoryLayer

RESULTS_DIR = Path("research/audit/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


def _format_turn(speaker: str, text: str, date_time: str,
                 session_idx: int) -> str:
    return f"[{date_time}] [session_{session_idx}] {speaker}: {text}"


def _compute_gold_rank(retrieved_ids: list[str],
                       gold_ids: set[str],
                       top_k: int) -> int:
    for pos, rid in enumerate(retrieved_ids[:top_k], start=1):
        if rid in gold_ids:
            return pos
    return 0


def run_conv(conv, mode: str, sbert_model, dim: int, top_k: int) -> list[dict]:
    """Run the rank probe on all QA pairs of one conversation."""
    def embed_fn(t: str) -> torch.Tensor:
        return torch.tensor(sbert_model.encode(t, convert_to_numpy=True))

    mem = MemoryLayer.ephemeral(embed_fn=embed_fn, embed_dim=dim)
    for session in conv.sessions:
        for turn in session.turns:
            text = _format_turn(
                turn.speaker, turn.text, session.date_time, session.index,
            )
            mem.store(text, metadata={"dia_id": turn.dia_id})

    hybrid_alpha = 0.3 if mode == "soma_hybrid" else None

    rows = []
    for qa in conv.qa_pairs:
        t_r = time.perf_counter()
        if hybrid_alpha is None:
            hits = mem.retrieve(qa.question, k=top_k)
        else:
            hits = mem.retrieve(qa.question, k=top_k, hybrid_alpha=hybrid_alpha)
        retr_ms = (time.perf_counter() - t_r) * 1000
        retrieved_ids = [h.metadata.get("dia_id", "") for h in hits]
        gold = set(qa.evidence)
        gold_rank = _compute_gold_rank(retrieved_ids, gold, top_k)
        rows.append({
            "sample_id": conv.sample_id,
            "question": qa.question,
            "category": qa.category,
            "category_name": qa.category_name,
            "mode": mode,
            "hit_at_k": 1 if gold_rank > 0 else 0,
            "gold_rank": gold_rank,
            "retrieval_ms": retr_ms,
        })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["chroma_cosine", "soma_hybrid"],
                   default="soma_hybrid")
    p.add_argument("--limit", type=int, default=None,
                   help="Max conversations")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out-suffix", default="")
    args = p.parse_args()

    items = load_dataset(limit=args.limit)
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    dim = sbert.get_sentence_embedding_dimension()

    out_path = RESULTS_DIR / f"locomo_rank_probe_{args.mode}{args.out_suffix}.jsonl"
    all_rows: list[dict] = []
    t_start = time.perf_counter()
    for i, conv in enumerate(items):
        rows = run_conv(conv, args.mode, sbert, dim, top_k=args.top_k)
        all_rows.extend(rows)
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elapsed = time.perf_counter() - t_start
        logger.info("[%s] conv %d/%d (%d QAs) | %.1fs elapsed",
                    args.mode, i + 1, len(items), len(rows), elapsed)

    total_s = time.perf_counter() - t_start
    hits = [r for r in all_rows if r["hit_at_k"] == 1]
    rank1 = [r for r in hits if r["gold_rank"] == 1]
    summary = {
        "mode": args.mode,
        "n_conv": len(items),
        "n_qa": len(all_rows),
        "hit_rate": len(hits) / max(1, len(all_rows)),
        "rank1_frac": len(rank1) / max(1, len(all_rows)),
        "mean_rank_given_hit": (
            sum(r["gold_rank"] for r in hits) / max(1, len(hits))
            if hits else 0.0
        ),
        "total_s": total_s,
    }
    summary_path = RESULTS_DIR / f"locomo_rank_probe_{args.mode}{args.out_suffix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
