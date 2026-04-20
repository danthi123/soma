"""LongMemEval retrieval benchmark — does SOMA's hybrid+rerank win hold?

LoCoMo showed SOMA hybrid+rerank beats chroma+same-rerank by +13% R@5.
LongMemEval is structurally different: ~50 sessions per question, ~500
turns per question, and questions explicitly test LONG-HORIZON memory
(the gold evidence lives in earlier sessions). Good generalization
test for the "hybrid+rerank is real value" claim.

Protocol: per item,
  1. Build a fresh memory (SOMA or chroma).
  2. Ingest each haystack session as one memory entry (session text
     = concatenated turn contents). Metadata: session_id.
  3. Retrieve top-K sessions for the question.
  4. Hit = at least one answer_session_id in the top-K retrieved.

Systems compared:
  - chroma-sbert baseline (cosine)
  - chroma-sbert + cross-encoder rerank (top-20)
  - soma-sbert baseline (cosine via MemoryLayer)
  - soma-sbert hybrid (alpha=0.3)
  - soma-sbert hybrid+rerank (alpha=0.3, top-20)

Uses sentence-transformers/all-MiniLM-L6-v2 throughout. Fast enough to
run all 500 items in ~10 min with sbert batching; mxbai-equivalent
would take hours on per-call Ollama, so sbert is the practical choice
here. The hybrid-vs-cosine DELTA is what generalizes — not the
absolute numbers.

Run::

    python -m benchmarks.industry.longmemeval.run_retrieval

Report lands in benchmarks/industry/longmemeval/results/retrieval_sbert.md.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.industry.longmemeval.data_loader import (
    LongMemEvalItem,
    load_dataset,
)
from soma.memory import MemoryLayer
from soma.memory.rerank import CrossEncoderReranker

REPORT = Path("benchmarks/industry/longmemeval/results/retrieval_sbert.md")
JSON_OUT = Path("benchmarks/industry/longmemeval/results/retrieval_sbert.json")

K_VALUES = (1, 5, 10)


@dataclass
class Result:
    system: str
    r_at_k: dict[int, float]
    per_type_r_at_k: dict[str, dict[int, float]]
    avg_retrieve_ms: float
    total_ingest_s: float


def _session_text(session: list[Any]) -> str:
    """Flatten a session (list of Turn) into a single searchable string."""
    return "\n".join(f"[{t.role}] {t.content}" for t in session)


def _score_system(
    label: str,
    items: list[LongMemEvalItem],
    build_system: Any,  # (session_texts, session_ids) -> obj
    retrieve_fn: Any,  # (obj, query, k) -> list[str] (session_ids)
) -> Result:
    hits_by_k: dict[int, int] = {k: 0 for k in K_VALUES}
    hits_by_type: dict[str, dict[int, int]] = defaultdict(
        lambda: {k: 0 for k in K_VALUES}
    )
    totals_by_type: dict[str, int] = defaultdict(int)
    total_items = 0
    total_retrieve = 0.0
    total_ingest = 0.0

    for it in items:
        # Haystack session_ids may repeat across sessions (same session_id
        # shared by multiple session blocks). Uniquify by appending index;
        # preserve original id in a parallel map so we can compare against
        # answer_session_ids (which use the original ids).
        raw_ids = it.haystack_session_ids
        session_ids = [f"{sid}__{i}" for i, sid in enumerate(raw_ids)]
        uniq_to_orig = {
            f"{sid}__{i}": sid for i, sid in enumerate(raw_ids)
        }
        session_texts = [_session_text(s) for s in it.haystack_sessions]
        t0 = time.perf_counter()
        obj = build_system(session_texts, session_ids)
        total_ingest += time.perf_counter() - t0

        t0 = time.perf_counter()
        retrieved = retrieve_fn(obj, it.question, max(K_VALUES))
        total_retrieve += time.perf_counter() - t0

        gold = set(it.answer_session_ids or [])
        # Map retrieved unique ids back to their original (possibly
        # duplicated) session_ids for comparison against `gold`.
        retrieved_orig = [uniq_to_orig.get(sid, sid) for sid in retrieved]
        total_items += 1
        totals_by_type[it.question_type] += 1
        for k in K_VALUES:
            if any(sid in gold for sid in retrieved_orig[:k]):
                hits_by_k[k] += 1
                hits_by_type[it.question_type][k] += 1

    return Result(
        system=label,
        r_at_k={k: hits_by_k[k] / max(1, total_items) for k in K_VALUES},
        per_type_r_at_k={
            t: {
                k: hits_by_type[t][k] / max(1, totals_by_type[t])
                for k in K_VALUES
            }
            for t in totals_by_type
        },
        avg_retrieve_ms=1000 * total_retrieve / max(1, total_items),
        total_ingest_s=total_ingest,
    )


def _build_sbert_embed_fn() -> tuple[Any, int]:
    """Return (embed_fn, dim) for MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    def _embed(text: str) -> torch.Tensor:
        return torch.tensor(model.encode(text, convert_to_numpy=True))

    return _embed, dim, model


def _chroma_builder(
    sbert_model: Any,
    reranker: CrossEncoderReranker | None,
) -> Any:
    """Returns (build_fn, retrieve_fn) pair for chroma with optional rerank."""
    import chromadb

    import uuid

    def build(session_texts: list[str], session_ids: list[str]) -> Any:
        # Fresh client per item (EphemeralClient shares state across
        # instances, so we need unique collection names to avoid collisions
        # even across chroma runs in the same process)
        client = chromadb.EphemeralClient()
        col = client.create_collection(
            name=f"longmem_{uuid.uuid4().hex[:12]}",
            metadata={"hnsw:space": "cosine"},
        )
        embeds = sbert_model.encode(session_texts, convert_to_numpy=True).tolist()
        col.add(ids=session_ids, documents=session_texts, embeddings=embeds)
        # Return (col, id->text)
        return (col, dict(zip(session_ids, session_texts, strict=True)))

    def retrieve(obj: Any, query: str, k: int) -> list[str]:
        col, id2text = obj
        pool_k = 20 if reranker is not None else k
        q_emb = sbert_model.encode(query, convert_to_numpy=True).tolist()
        result = col.query(query_embeddings=[q_emb], n_results=pool_k)
        candidate_ids: list[str] = result.get("ids", [[]])[0]

        if reranker is not None and candidate_ids:
            docs = [id2text[cid] for cid in candidate_ids]
            scores = reranker.score(query, docs)
            order = sorted(range(len(docs)), key=lambda i: -scores[i])
            return [candidate_ids[i] for i in order[:k]]
        return candidate_ids[:k]

    return build, retrieve


def _soma_builder(
    embed_fn: Any,
    dim: int,
    reranker: CrossEncoderReranker | None,
    *,
    hybrid_alpha: float | None = None,
    rerank_top_n: int | None = None,
) -> Any:
    def build(session_texts: list[str], session_ids: list[str]) -> Any:
        # Ephemeral MemoryLayer per item: no WAL, no bundle.
        mem = MemoryLayer.ephemeral(embed_fn=embed_fn, embed_dim=dim)
        if reranker is not None:
            mem.attach_reranker(reranker)
        for sid, text in zip(session_ids, session_texts, strict=True):
            mem.store(text, metadata={"sid": sid})
        return mem

    def retrieve(mem: Any, query: str, k: int) -> list[str]:
        kwargs: dict[str, Any] = {}
        if hybrid_alpha is not None:
            kwargs["hybrid_alpha"] = hybrid_alpha
        if rerank_top_n is not None:
            kwargs["rerank_top_n"] = rerank_top_n
        hits = mem.retrieve(query, k=k, **kwargs)
        return [h.metadata["sid"] for h in hits]

    return build, retrieve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small", choices=["small", "oracle", "medium"])
    p.add_argument(
        "--limit", type=int, default=None,
        help="limit number of items (quick smoke)",
    )
    p.add_argument(
        "--skip-chroma", action="store_true",
    )
    args = p.parse_args()

    print(f"Loading LongMemEval {args.variant}...")
    items = load_dataset(args.variant)
    if args.limit is not None:
        items = items[: args.limit]
    print(f"  {len(items)} items")
    type_counts = defaultdict(int)
    for i in items:
        type_counts[i.question_type] += 1
    for t, n in sorted(type_counts.items()):
        print(f"    {t}: {n}")

    print("\nLoading sbert all-MiniLM-L6-v2 ...")
    embed_fn, dim, sbert_model = _build_sbert_embed_fn()
    print(f"  dim = {dim}")

    print("Loading cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
    reranker_shared = CrossEncoderReranker()
    # Warm-up
    _ = reranker_shared.score("warmup", ["test"])

    results: list[Result] = []

    if not args.skip_chroma:
        print("\n=== chroma-sbert baseline (cosine) ===")
        build, retrieve = _chroma_builder(sbert_model, reranker=None)
        r = _score_system("chroma-sbert baseline", items, build, retrieve)
        results.append(r)
        print(
            f"  R@1={r.r_at_k[1]:.3f}  R@5={r.r_at_k[5]:.3f}  "
            f"R@10={r.r_at_k[10]:.3f}  retrieve={r.avg_retrieve_ms:.1f}ms  "
            f"ingest={r.total_ingest_s:.1f}s"
        )

        print("\n=== chroma-sbert + cross-encoder rerank (top-20) ===")
        build, retrieve = _chroma_builder(sbert_model, reranker=reranker_shared)
        r = _score_system("chroma-sbert + rerank", items, build, retrieve)
        results.append(r)
        print(
            f"  R@1={r.r_at_k[1]:.3f}  R@5={r.r_at_k[5]:.3f}  "
            f"R@10={r.r_at_k[10]:.3f}  retrieve={r.avg_retrieve_ms:.1f}ms  "
            f"ingest={r.total_ingest_s:.1f}s"
        )

    print("\n=== soma-sbert baseline (cosine) ===")
    build, retrieve = _soma_builder(embed_fn, dim, reranker=None)
    r = _score_system("soma-sbert baseline", items, build, retrieve)
    results.append(r)
    print(
        f"  R@1={r.r_at_k[1]:.3f}  R@5={r.r_at_k[5]:.3f}  "
        f"R@10={r.r_at_k[10]:.3f}  retrieve={r.avg_retrieve_ms:.1f}ms  "
        f"ingest={r.total_ingest_s:.1f}s"
    )

    print("\n=== soma-sbert hybrid (alpha=0.3) ===")
    build, retrieve = _soma_builder(embed_fn, dim, reranker=None, hybrid_alpha=0.3)
    r = _score_system("soma-sbert hybrid (a=0.3)", items, build, retrieve)
    results.append(r)
    print(
        f"  R@1={r.r_at_k[1]:.3f}  R@5={r.r_at_k[5]:.3f}  "
        f"R@10={r.r_at_k[10]:.3f}  retrieve={r.avg_retrieve_ms:.1f}ms  "
        f"ingest={r.total_ingest_s:.1f}s"
    )

    print("\n=== soma-sbert hybrid+rerank (alpha=0.3, top-20) ===")
    build, retrieve = _soma_builder(
        embed_fn, dim, reranker=reranker_shared,
        hybrid_alpha=0.3, rerank_top_n=20,
    )
    r = _score_system(
        "soma-sbert hybrid+rerank (a=0.3, top-20)",
        items, build, retrieve,
    )
    results.append(r)
    print(
        f"  R@1={r.r_at_k[1]:.3f}  R@5={r.r_at_k[5]:.3f}  "
        f"R@10={r.r_at_k[10]:.3f}  retrieve={r.avg_retrieve_ms:.1f}ms  "
        f"ingest={r.total_ingest_s:.1f}s"
    )

    # Summary
    if results:
        base = results[0]  # chroma baseline if present, else soma baseline
        print("\n=== Summary ===")
        print(
            f"{'config':<45}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  "
            f"{'ms':>6}  {'R@5 delta':>11}"
        )
        for r in results:
            d5 = (r.r_at_k[5] - base.r_at_k[5]) * 100
            print(
                f"{r.system:<45}  {r.r_at_k[1]:>6.3f}  "
                f"{r.r_at_k[5]:>6.3f}  {r.r_at_k[10]:>6.3f}  "
                f"{r.avg_retrieve_ms:>6.1f}  {d5:>+10.2f} pp"
            )

    # Write report
    lines = [
        "# LongMemEval retrieval benchmark — SOMA vs Chroma (sbert)",
        "",
        f"Dataset: LongMemEval `{args.variant}` variant, "
        f"{len(items)} items across {len(type_counts)} question types.",
        "",
        "Protocol per item: build a fresh memory, ingest each haystack",
        "session as one memory entry (concatenated turns), retrieve top-K",
        "sessions for the question. Hit = any gold-evidence session_id in",
        "the top-K retrieved.",
        "",
        "Embedder: sentence-transformers/all-MiniLM-L6-v2.",
        "Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (shared across",
        "SOMA and chroma-rerank paths).",
        "",
        "## Results",
        "",
        "| Strategy | R@1 | R@5 | R@10 | Retrieve (ms) | Ingest (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.system} | {r.r_at_k[1]:.3f} | {r.r_at_k[5]:.3f} | "
            f"{r.r_at_k[10]:.3f} | {r.avg_retrieve_ms:.1f} | "
            f"{r.total_ingest_s:.1f} |"
        )
    lines += [
        "",
        "## R@5 by question type",
        "",
        "| Strategy | "
        + " | ".join(
            sorted({t for r in results for t in r.per_type_r_at_k.keys()})
        )
        + " |",
        "| --- | "
        + " | ".join(
            ["---:"]
            * len({t for r in results for t in r.per_type_r_at_k.keys()})
        )
        + " |",
    ]
    for r in results:
        types = sorted(r.per_type_r_at_k.keys())
        row = f"| {r.system} | " + " | ".join(
            f"{r.per_type_r_at_k[t][5]:.3f}" for t in types
        ) + " |"
        lines.append(row)
    lines.append("")
    lines.append(f"Generated by `benchmarks/industry/longmemeval/run_retrieval.py`.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {REPORT}")

    # Also dump JSON
    import json

    payload = {
        "variant": args.variant,
        "n_items": len(items),
        "type_counts": dict(type_counts),
        "results": [
            {
                "system": r.system,
                "r_at_k": r.r_at_k,
                "per_type_r_at_k": r.per_type_r_at_k,
                "avg_retrieve_ms": r.avg_retrieve_ms,
                "total_ingest_s": r.total_ingest_s,
            }
            for r in results
        ],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON: {JSON_OUT}")


if __name__ == "__main__":
    main()
