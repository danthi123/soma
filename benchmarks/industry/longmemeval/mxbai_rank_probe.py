"""Rank probe using mxbai-embed-large (ollama) instead of sbert.

Mirror of ``rank_probe.py`` but swaps the embedding function for an
ollama-served mxbai-embed-large (1024-dim). Answers: "does SOMA's
+0.16 mean-rank advantage over chroma survive a stronger embedder?"

If yes, the lift is embedder-agnostic (baked into the ranking
mechanism, not just a quirk of the 384-dim MiniLM). If no, SOMA only
helps when cosine is already weak.

Retrieval-only. No LLM involvement.

Usage::

    python -m benchmarks.industry.longmemeval.mxbai_rank_probe \\
        --variant small --limit 100 --mode soma_hybrid
    python -m benchmarks.industry.longmemeval.mxbai_rank_probe \\
        --variant small --limit 100 --mode chroma_cosine
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
import torch

from benchmarks.industry.longmemeval.data_loader import load_dataset
from benchmarks.industry.longmemeval.run_qa_compare import (
    _compute_gold_rank,
    _session_text,
)
from soma.memory import MemoryLayer

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/embed"
_EMBED_MODEL = "mxbai-embed-large"
_EMBED_DIM = 1024


def _mxbai_embed(text: str, max_chars: int = 2000) -> torch.Tensor:
    """Single-text embed via ollama /api/embed. Returns 1024-d torch tensor.

    Truncates to ``max_chars`` to respect mxbai-embed-large's ~512-token
    limit. Retries up to 3 times, aggressively truncating on HTTP 400.
    Returns a zero vector on persistent failure (with warning) so the
    probe doesn't crash on a single problematic item.
    """
    # Empty or whitespace-only text -> zero vec (avoid 400 from ollama)
    stripped = text.strip()
    if not stripped:
        logger.warning("Empty input text; returning zero vector")
        return torch.zeros(_EMBED_DIM, dtype=torch.float32)
    if len(stripped) > max_chars:
        stripped = stripped[:max_chars]
    last_err: Exception | None = None
    current_text = stripped
    for attempt in range(3):
        try:
            resp = requests.post(
                _OLLAMA_URL,
                json={"model": _EMBED_MODEL, "input": current_text},
                timeout=120,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings") or []
            if not embeddings or not embeddings[0]:
                raise RuntimeError(
                    f"Empty embedding for text: {current_text[:100]!r}"
                )
            return torch.tensor(embeddings[0], dtype=torch.float32)
        except (requests.exceptions.HTTPError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_err = exc
            if isinstance(exc, requests.exceptions.HTTPError):
                if exc.response is not None and exc.response.status_code == 400:
                    # Aggressive truncation on 400 — model may be rejecting
                    # due to token count. Try half length.
                    current_text = current_text[: max(200, len(current_text) // 2)]
                    logger.warning(
                        "HTTP 400 on embed, truncating to %d chars (attempt %d)",
                        len(current_text), attempt + 1,
                    )
    # All retries failed; return zero vector to avoid crash
    logger.error("mxbai embed failed persistently (last err: %s); "
                 "returning zero vector for %r",
                 last_err, current_text[:80])
    return torch.zeros(_EMBED_DIM, dtype=torch.float32)


def run_one(
    mode: str,
    items: list[Any],
    top_k: int,
    out_path: Path,
) -> dict[str, Any]:
    """Run a single mode (chroma_cosine | soma_hybrid) at mxbai-embed."""
    rows: list[dict] = []
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add(r["question_id"])
                rows.append(r)
        logger.info("Resuming %s: %d already done", mode, len(done))

    hybrid_alpha = 0.3 if mode == "soma_hybrid" else None

    t_start = time.perf_counter()
    for i, item in enumerate(items):
        if item.question_id in done:
            continue
        raw_ids = item.haystack_session_ids
        uniq_ids = [f"{sid}__{j}" for j, sid in enumerate(raw_ids)]
        uniq_to_orig = {u: orig for u, orig in zip(uniq_ids, raw_ids, strict=True)}
        mem = MemoryLayer.ephemeral(embed_fn=_mxbai_embed, embed_dim=_EMBED_DIM)
        for j, sess in enumerate(item.haystack_sessions):
            sess_date = item.haystack_dates[j] if j < len(item.haystack_dates) else ""
            mem.store(_session_text(sess, sess_date), metadata={"sid": uniq_ids[j]})
        t_r = time.perf_counter()
        if hybrid_alpha is None:
            hits = mem.retrieve(item.question, k=top_k)
        else:
            hits = mem.retrieve(item.question, k=top_k, hybrid_alpha=hybrid_alpha)
        retr_ms = (time.perf_counter() - t_r) * 1000
        retrieved_origs = [
            uniq_to_orig.get(h.metadata["sid"], h.metadata["sid"]) for h in hits
        ]
        gold = set(item.answer_session_ids or [])
        gold_rank = _compute_gold_rank(retrieved_origs, gold, top_k)
        row = {
            "question_id": item.question_id,
            "question_type": item.question_type,
            "mode": mode,
            "hit_at_k": 1 if gold_rank > 0 else 0,
            "gold_rank": gold_rank,
            "retrieval_ms": retr_ms,
        }
        rows.append(row)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            logger.info(
                "[%s mxbai] %d/%d | %.1fs | last gold_rank=%d",
                mode, i + 1, len(items), elapsed, gold_rank,
            )

    total_s = time.perf_counter() - t_start
    hits = [r for r in rows if r["hit_at_k"] == 1]
    ranks = [r["gold_rank"] for r in hits]
    return {
        "mode": mode,
        "embedder": _EMBED_MODEL,
        "n": len(rows),
        "hit_rate": len(hits) / max(1, len(rows)),
        "mean_rank_given_hit": sum(ranks) / len(ranks) if ranks else 0.0,
        "rank1_frac": sum(1 for r in ranks if r == 1) / max(1, len(rows)),
        "total_s": total_s,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mode", choices=["chroma_cosine", "soma_hybrid"],
                   default="soma_hybrid")
    p.add_argument("--out-suffix", default="_n100_mxbai_rank_probe")
    args = p.parse_args()

    items = load_dataset(args.variant, limit=args.limit)
    out_path = RESULTS_DIR / f"mxbai_rank_probe_{args.mode}_n{args.limit}{args.out_suffix.replace('_n100_mxbai_rank_probe','')}.jsonl"
    summary = run_one(args.mode, items, top_k=args.top_k, out_path=out_path)
    logger.info("summary: %s", json.dumps(summary, indent=2))

    summary_path = RESULTS_DIR / f"mxbai_rank_probe_{args.mode}_n{args.limit}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
