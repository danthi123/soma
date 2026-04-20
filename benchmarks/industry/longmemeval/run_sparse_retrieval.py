"""Sparse-retrieval probe for LongMemEval — Path B Phase 3.

Measures whether adding a sparse-overlap retrieval score (from the
Path B numpy primitives in :mod:`soma.memory.sparse_codes`) to SOMA's
existing hybrid retrieval produces a rank-1 lift. Self-contained
retrieval — does NOT use MemoryLayer, so the scoring is fully
transparent and reproducible.

Variants tested (design reference:
docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md § Phase 3):

- ``hybrid_only``        : 0.7·cosine + 0.3·bm25 (current SOMA default)
- ``hybrid_plus_sparse`` : 0.5·cosine + 0.2·bm25 + 0.3·sparse_overlap
- ``sparse_only``        : sparse_overlap (sanity check)

Outputs per-item jsonl with the gold rank under each variant for paired
analysis.

Usage::

    python -m benchmarks.industry.longmemeval.run_sparse_retrieval \\
        --variant small --limit 100 --top-k 5 \\
        --sparse-k 32 --sparse-dim 4096 \\
        --out-suffix _n100_sparse
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from benchmarks.industry.longmemeval.data_loader import load_dataset
from benchmarks.industry.longmemeval.run_qa_compare import (
    _build_sbert,
    _compute_gold_rank,
    _session_text,
)
from soma.memory.bm25 import BM25Index
from soma.memory.sparse_codes import SparseCode, code_similarity, kwta

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


VARIANT_WEIGHTS = {
    "hybrid_only":        (0.7, 0.3, 0.0),
    "hybrid_plus_sparse": (0.5, 0.2, 0.3),
    "sparse_only":        (0.0, 0.0, 1.0),
}


class SparseAugmentedIndex:
    """In-memory index that stores dense embedding, BM25 tokens, and
    sparse code per document. Scores each query with the three signals
    and returns their weighted combination ranked top-k.

    Scores are each min-max normalised to [0, 1] across the current
    document set before combining — that keeps the weights
    interpretable regardless of the absolute magnitudes each signal
    produces on a given haystack."""

    def __init__(
        self,
        embed_fn: Any,
        sparse_k: int,
        sparse_dim: int,
        sparse_seed: int,
    ) -> None:
        self.embed_fn = embed_fn
        self.sparse_k = sparse_k
        self.sparse_dim = sparse_dim
        # One projection matrix shared across all docs / queries.
        # Generated lazily once we know the embed dim.
        self._projection: np.ndarray | None = None
        self._sparse_seed = sparse_seed

        self._ids: list[str] = []
        self._texts: list[str] = []
        self._dense: np.ndarray | None = None        # shape (n_docs, embed_dim)
        self._sparse: list[SparseCode] = []
        self._bm25: BM25Index | None = None

    def _ensure_projection(self, embed_dim: int) -> None:
        if self._projection is None:
            rng = np.random.default_rng(self._sparse_seed)
            self._projection = rng.standard_normal(
                (self.sparse_dim, embed_dim)
            ).astype(np.float32)

    def _encode_dense(self, text: str) -> np.ndarray:
        emb = self.embed_fn(text)
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        return np.asarray(emb, dtype=np.float32)

    def _encode_sparse(self, dense: np.ndarray) -> SparseCode:
        assert self._projection is not None
        return kwta(
            dense, k=self.sparse_k, dim=self.sparse_dim,
            projection=self._projection,
        )

    def ingest(self, texts: list[str], ids: list[str]) -> None:
        self._ids = list(ids)
        self._texts = list(texts)
        denses = [self._encode_dense(t) for t in texts]
        embed_dim = denses[0].shape[0]
        self._ensure_projection(embed_dim)

        self._dense = np.stack(denses).astype(np.float32)
        self._sparse = [self._encode_sparse(d) for d in denses]

        self._bm25 = BM25Index()
        self._bm25.build(texts)

    def retrieve(self, query: str, k: int, variant: str) -> list[tuple[str, str, float]]:
        assert self._bm25 is not None and self._dense is not None
        w_cos, w_bm, w_sparse = VARIANT_WEIGHTS[variant]
        n = len(self._ids)

        # --- Dense cosine ---
        q_dense = self._encode_dense(query)
        q_norm = q_dense / (np.linalg.norm(q_dense) + 1e-12)
        doc_norms = self._dense / (np.linalg.norm(self._dense, axis=1, keepdims=True) + 1e-12)
        cos_scores = doc_norms @ q_norm  # shape (n,)

        # --- BM25 ---
        # BM25Index.search returns (doc_index, score); doc_index indexes
        # into the list passed to .build(). Same order as self._ids.
        bm25_hits = self._bm25.search(query, k=n)
        bm_scores = np.zeros(n, dtype=np.float32)
        for doc_idx, score in bm25_hits:
            bm_scores[doc_idx] = score

        # --- Sparse overlap ---
        sparse_scores = np.zeros(n, dtype=np.float32)
        if w_sparse > 0:
            q_sparse = self._encode_sparse(q_dense)
            for i, ds in enumerate(self._sparse):
                sparse_scores[i] = code_similarity(q_sparse, ds)

        # --- Normalise each to [0, 1] across this query's doc set ---
        def _norm(arr: np.ndarray) -> np.ndarray:
            lo, hi = float(arr.min()), float(arr.max())
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        cos_n = _norm(cos_scores) if w_cos > 0 else np.zeros(n, dtype=np.float32)
        bm_n = _norm(bm_scores) if w_bm > 0 else np.zeros(n, dtype=np.float32)
        sp_n = _norm(sparse_scores) if w_sparse > 0 else np.zeros(n, dtype=np.float32)

        combined = w_cos * cos_n + w_bm * bm_n + w_sparse * sp_n

        # Top-k by combined score.
        order = np.argsort(-combined)[:k]
        return [(self._ids[i], self._texts[i], float(combined[i])) for i in order]


def probe_all_variants(
    variants: list[str],
    items: list[Any],
    index_factory,
    top_k: int,
    out_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Share the index across variants — ingest is expensive, scoring is cheap."""
    per_variant_rows: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    done_qids_by_variant: dict[str, set[str]] = {}
    for v, p in out_paths.items():
        done: set[str] = set()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    done.add(row["question_id"])
                    per_variant_rows[v].append(row)
        done_qids_by_variant[v] = done
        if done:
            logger.info("Resuming %s: %d items already done", v, len(done))

    t_total = time.perf_counter()

    for i, item in enumerate(items):
        # Skip this item only if ALL variants already have it.
        if all(item.question_id in done_qids_by_variant[v] for v in variants):
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
        idx = index_factory()
        idx.ingest(texts, uniq_ids)

        for variant in variants:
            if item.question_id in done_qids_by_variant[variant]:
                continue
            t0 = time.perf_counter()
            retrieved = idx.retrieve(item.question, top_k, variant=variant)
            retr_ms = (time.perf_counter() - t0) * 1000
            retrieved_origs = [uniq_to_orig.get(sid, sid) for sid, _, _ in retrieved]
            gold = set(item.answer_session_ids or [])
            gold_rank = _compute_gold_rank(retrieved_origs, gold, top_k)
            hit = 1 if gold_rank > 0 else 0

            row = {
                "question_id": item.question_id,
                "question_type": item.question_type,
                "variant": variant,
                "hit_at_k": hit,
                "gold_rank": gold_rank,
                "retrieval_ms": retr_ms,
            }
            per_variant_rows[variant].append(row)
            with open(out_paths[variant], "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0:
            msg_parts = []
            for v in variants:
                rows = per_variant_rows[v]
                if rows:
                    hits = sum(r["hit_at_k"] for r in rows)
                    r1 = sum(1 for r in rows if r["gold_rank"] == 1)
                    msg_parts.append(
                        f"{v}: hit={hits/len(rows):.2f} r1={r1/len(rows):.2f}"
                    )
            logger.info("[%d/%d] %s", i + 1, len(items), " | ".join(msg_parts))

    total_ms = (time.perf_counter() - t_total) * 1000

    summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        rows = per_variant_rows[variant]
        hits_rows = [r for r in rows if r["hit_at_k"] == 1]
        ranks = [r["gold_rank"] for r in hits_rows]
        summaries[variant] = {
            "variant": variant,
            "n": len(rows),
            "hit_rate": len(hits_rows) / max(1, len(rows)),
            "rank1_frac": sum(1 for r in ranks if r == 1) / max(1, len(rows)),
            "mean_rank_given_hit": mean(ranks) if ranks else 0.0,
            "median_rank_given_hit": median(ranks) if ranks else 0.0,
            "rank_hist": {str(k): sum(1 for r in ranks if r == k) for k in range(1, 6)},
            "total_ms": total_ms,
        }
    return summaries


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--sparse-k", type=int, default=32)
    p.add_argument("--sparse-dim", type=int, default=4096)
    p.add_argument("--sparse-seed", type=int, default=42)
    p.add_argument("--sbert-device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--out-suffix", default="_n100_sparse")
    p.add_argument("--variants", nargs="+",
                   default=["hybrid_only", "hybrid_plus_sparse", "sparse_only"])
    args = p.parse_args()

    items = load_dataset(args.variant, limit=args.limit)
    logger.info("Loaded %d items from LongMemEval %s", len(items), args.variant)

    sbert_model, dim = _build_sbert(device=args.sbert_device)

    def embed_fn(text: str):
        import torch
        return torch.tensor(sbert_model.encode(text, convert_to_numpy=True))

    def make_index() -> SparseAugmentedIndex:
        return SparseAugmentedIndex(
            embed_fn=embed_fn,
            sparse_k=args.sparse_k,
            sparse_dim=args.sparse_dim,
            sparse_seed=args.sparse_seed,
        )

    out_paths = {
        v: RESULTS_DIR / f"sparse_retrieval_{v}{args.out_suffix}.jsonl"
        for v in args.variants
    }
    summaries = probe_all_variants(
        args.variants, items, make_index,
        top_k=args.top_k, out_paths=out_paths,
    )
    for v, s in summaries.items():
        logger.info("%s summary: %s", v, json.dumps(s, indent=2))

    summary_path = RESULTS_DIR / f"sparse_retrieval_summary{args.out_suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
