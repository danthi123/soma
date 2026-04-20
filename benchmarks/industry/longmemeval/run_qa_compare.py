"""Does the retrieval lift translate to answer-quality lift?

We already showed SOMA's built-in BM25+cosine hybrid beats chroma +
cross-encoder-rerank on LongMemEval retrieval (+4.7% R@5, 8.3x faster).
The remaining question: when that retrieved context is fed to the SAME
LLM, does the QA answer get measurably better?

This script isolates retrieval as the ONLY variable. Per item:
  1. Build retrieval index for each mode (fresh per item).
  2. Retrieve top-K sessions for the question.
  3. Pack retrieved session texts into a fixed-size context.
  4. Send to the SAME LLM with the SAME system prompt.
  5. Score the answer via F1/ROUGE/EM.

Modes compared:
  - chroma_rerank   : chroma cosine + cross-encoder rerank top-20 (best chroma)
  - soma_hybrid     : SOMA hybrid alpha=0.3, no rerank (best for LongMemEval)

Both use sbert all-MiniLM-L6-v2 for embedding. Both use
cross-encoder/ms-marco-MiniLM-L-6-v2 (rerank only applied to chroma_rerank).
LLM: qwen3.5:4b-q8_0 via Ollama.

Run::

    python -m benchmarks.industry.longmemeval.run_qa_compare \
        --variant small --limit 50 \
        --modes chroma_rerank soma_hybrid

Outputs to benchmarks/industry/longmemeval/results/qa_compare_*.json
and aggregate qa_compare_summary.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import torch

from benchmarks.industry.longmemeval.data_loader import (
    LongMemEvalItem,
    load_dataset,
)
from benchmarks.industry.longmemeval.metrics import (
    LongMemEvalScores,
    compute_scores,
)
from soma.memory import MemoryLayer
from soma.memory.rerank import CrossEncoderReranker

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b-q8_0"
CHARS_PER_TOKEN = 4
DEFAULT_MAX_CONTEXT_TOKENS = 3800
RETRIEVE_TOP_K = 5
RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")

MODES = (
    "chroma_rerank",
    "soma_hybrid",
    "soma_hybrid_rerank",
    "chroma_cosine",
    "full_context",
)


# ---------------------------------------------------------------------------
# Retrieval backends
# ---------------------------------------------------------------------------


def _session_text(session: list[Any], session_date: str) -> str:
    """Flatten a session into one searchable/LLM-readable string."""
    header = f"[Session date: {session_date}]\n" if session_date else ""
    body = "\n".join(f"{t.role}: {t.content}" for t in session)
    return header + body


def _build_sbert(device: str = "cpu") -> tuple[Any, int]:
    """Build sbert. Defaults to CPU to avoid VRAM contention with ollama LLMs.

    sbert inference on CPU is ~50ms for a session-sized text, which is fine
    for LongMemEval ingest speed (dominated by LLM latency anyway).
    Using GPU sbert alongside a 10+GB LLM in ollama caused CUDA errors
    on a 24GB card.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    dim = model.get_sentence_embedding_dimension()
    return model, dim


class _ChromaIndex:
    def __init__(self, sbert_model: Any, reranker: CrossEncoderReranker | None) -> None:
        self.sbert = sbert_model
        self.reranker = reranker
        self._client = None
        self._col = None
        self._id_to_text: dict[str, str] = {}

    def ingest(self, texts: list[str], ids: list[str]) -> None:
        import chromadb

        self._client = chromadb.EphemeralClient()
        self._col = self._client.create_collection(
            name=f"qa_{uuid.uuid4().hex[:12]}",
            metadata={"hnsw:space": "cosine"},
        )
        embeds = self.sbert.encode(texts, convert_to_numpy=True).tolist()
        self._col.add(ids=ids, documents=texts, embeddings=embeds)
        self._id_to_text = dict(zip(ids, texts, strict=True))

    def retrieve(self, query: str, k: int) -> list[tuple[str, str]]:
        """Returns [(doc_id, doc_text)] top-k."""
        assert self._col is not None, "must ingest first"
        pool_k = 20 if self.reranker is not None else k
        q_emb = self.sbert.encode(query, convert_to_numpy=True).tolist()
        result = self._col.query(query_embeddings=[q_emb], n_results=pool_k)
        candidate_ids: list[str] = result.get("ids", [[]])[0]
        if self.reranker is not None and candidate_ids:
            docs = [self._id_to_text[cid] for cid in candidate_ids]
            scores = self.reranker.score(query, docs)
            order = sorted(range(len(docs)), key=lambda i: -scores[i])
            top_ids = [candidate_ids[i] for i in order[:k]]
        else:
            top_ids = candidate_ids[:k]
        return [(cid, self._id_to_text[cid]) for cid in top_ids]


class _FullContextIndex:
    """No retrieval. Returns all sessions truncated-from-front to fit budget.

    Simulates 'just stuff everything in context' baseline — the default
    approach when LLMs have large context windows. The cost argument is
    direct: how much context do we need for the same F1?
    """

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars
        self._ids: list[str] = []
        self._texts: list[str] = []

    def ingest(self, texts: list[str], ids: list[str]) -> None:
        self._ids = list(ids)
        self._texts = list(texts)

    def retrieve(self, query: str, k: int) -> list[tuple[str, str]]:
        """Return all sessions (truncated-from-front if total exceeds budget).

        Ignores ``query`` and ``k`` — the entire haystack goes in. Oldest
        sessions drop first when we exceed budget.
        """
        items = list(zip(self._ids, self._texts, strict=True))
        total = sum(len(t) for _, t in items) + 2 * max(0, len(items) - 1)
        while items and total > self.max_chars:
            dropped_sid, dropped_text = items.pop(0)
            total -= len(dropped_text) + 2
        return items


class _SomaIndex:
    def __init__(
        self,
        embed_fn: Any,
        dim: int,
        *,
        reranker: CrossEncoderReranker | None,
        hybrid_alpha: float | None,
        rerank_top_n: int | None,
    ) -> None:
        self.embed_fn = embed_fn
        self.dim = dim
        self.reranker = reranker
        self.hybrid_alpha = hybrid_alpha
        self.rerank_top_n = rerank_top_n
        self._mem: MemoryLayer | None = None

    def ingest(self, texts: list[str], ids: list[str]) -> None:
        mem = MemoryLayer.ephemeral(embed_fn=self.embed_fn, embed_dim=self.dim)
        if self.reranker is not None:
            mem.attach_reranker(self.reranker)
        for sid, text in zip(ids, texts, strict=True):
            mem.store(text, metadata={"sid": sid})
        self._mem = mem

    def retrieve(self, query: str, k: int) -> list[tuple[str, str]]:
        assert self._mem is not None, "must ingest first"
        kwargs: dict[str, Any] = {}
        if self.hybrid_alpha is not None:
            kwargs["hybrid_alpha"] = self.hybrid_alpha
        if self.rerank_top_n is not None:
            kwargs["rerank_top_n"] = self.rerank_top_n
        hits = self._mem.retrieve(query, k=k, **kwargs)
        return [(h.metadata["sid"], h.text) for h in hits]


def _make_index(
    mode: str,
    sbert_model: Any,
    dim: int,
    embed_fn: Any,
    reranker: CrossEncoderReranker,
    max_context_tokens: int,
) -> Any:
    if mode == "chroma_cosine":
        return _ChromaIndex(sbert_model, reranker=None)
    if mode == "chroma_rerank":
        return _ChromaIndex(sbert_model, reranker=reranker)
    if mode == "soma_hybrid":
        return _SomaIndex(
            embed_fn, dim,
            reranker=None,
            hybrid_alpha=0.3,
            rerank_top_n=None,
        )
    if mode == "soma_hybrid_rerank":
        return _SomaIndex(
            embed_fn, dim,
            reranker=reranker,
            hybrid_alpha=0.3,
            rerank_top_n=20,
        )
    if mode == "full_context":
        # Reserve space for system prompt + question (~200 tokens)
        return _FullContextIndex(
            max_chars=(max_context_tokens - 200) * CHARS_PER_TOKEN,
        )
    raise ValueError(f"unknown mode: {mode}")


# ---------------------------------------------------------------------------
# LLM pipeline
# ---------------------------------------------------------------------------


def _pack_context(
    retrieved: list[tuple[str, str]],
    max_tokens: int,
) -> str:
    """Pack retrieved (id, text) chunks into a budget-capped context string."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    lines: list[str] = []
    used = 0
    for idx, (sid, text) in enumerate(retrieved):
        block = f"--- Evidence #{idx+1} ---\n{text}"
        cost = len(block) + (2 if lines else 0)  # blank line separator
        if used + cost > max_chars:
            # Try to fit a truncated version
            remaining = max_chars - used - (2 if lines else 0) - 80
            if remaining > 200:
                truncated = block[:remaining] + "\n... (truncated)"
                lines.append(truncated)
            break
        lines.append(block)
        used += cost
    return "\n\n".join(lines)


def _call_llm(
    context: str,
    question: str,
    question_date: str,
    *,
    api_base: str,
    model: str,
) -> str:
    system_prompt = (
        "You are an AI assistant that recalls information from past conversations. "
        "Answer the question using ONLY the evidence provided. Be concise and direct. "
        "If the evidence does not contain the answer, say 'I don't know'."
    )
    user_msg = ""
    if context.strip():
        user_msg += f"Relevant conversation history:\n{context}\n\n"
    user_msg += f"Current date: {question_date}\n"
    user_msg += f"Question: {question}\nAnswer:"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": 512, "temperature": 0.0},
    }
    try:
        resp = requests.post(
            f"{api_base}/api/chat",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError) as exc:
        logger.error("LLM call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def _evaluate_mode(
    mode: str,
    items: list[LongMemEvalItem],
    *,
    sbert_model: Any,
    dim: int,
    embed_fn: Any,
    reranker: CrossEncoderReranker,
    api_base: str,
    model: str,
    max_context_tokens: int,
    top_k: int,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    retrieval_latencies: list[float] = []
    llm_latencies: list[float] = []
    retrieved_hits_at_k: list[int] = []
    input_token_estimates: list[int] = []

    # Resume support: read any partial .jsonl we wrote previously
    already_done: set[str] = set()
    jsonl_path = resume_path
    if jsonl_path and jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                already_done.add(row["question_id"])
                predictions.append(
                    {"question_id": row["question_id"], "hypothesis": row["hypothesis"]}
                )
                references.append({
                    "question_id": row["question_id"],
                    "answer": str(row["answer"]),
                    "question_type": row["question_type"],
                })
                retrieval_latencies.append(row.get("retrieval_ms", 0.0))
                llm_latencies.append(row.get("llm_ms", 0.0))
                retrieved_hits_at_k.append(int(row.get("hit_at_k", 0)))
                input_token_estimates.append(int(row.get("input_tokens", 0)))
        logger.info("Resumed %d items from %s", len(already_done), jsonl_path)

    for i, item in enumerate(items):
        if item.question_id in already_done:
            continue
        # Uniquify haystack ids (LongMemEval can repeat)
        raw_ids = item.haystack_session_ids
        uniq_ids = [f"{sid}__{j}" for j, sid in enumerate(raw_ids)]
        uniq_to_orig = {f"{sid}__{j}": sid for j, sid in enumerate(raw_ids)}
        texts = [
            _session_text(
                sess,
                item.haystack_dates[j] if j < len(item.haystack_dates) else "",
            )
            for j, sess in enumerate(item.haystack_sessions)
        ]

        idx = _make_index(
            mode, sbert_model, dim, embed_fn, reranker, max_context_tokens,
        )
        idx.ingest(texts, uniq_ids)

        t0 = time.perf_counter()
        retrieved = idx.retrieve(item.question, top_k)
        retrieval_latencies.append((time.perf_counter() - t0) * 1000)

        # Bookkeeping: did retrieval find the right evidence?
        gold = set(item.answer_session_ids or [])
        retrieved_origs = [uniq_to_orig.get(sid, sid) for sid, _ in retrieved]
        hit = 1 if any(sid in gold for sid in retrieved_origs[:top_k]) else 0
        retrieved_hits_at_k.append(hit)

        # Pack context, call LLM
        context = _pack_context(retrieved, max_context_tokens)
        # Approximate prompt tokens for cost reporting
        prompt_chars = len(context) + len(item.question) + 300
        input_token_estimates.append(prompt_chars // CHARS_PER_TOKEN)
        t0 = time.perf_counter()
        answer = _call_llm(
            context, item.question, item.question_date,
            api_base=api_base, model=model,
        )
        llm_latencies.append((time.perf_counter() - t0) * 1000)

        predictions.append(
            {"question_id": item.question_id, "hypothesis": answer}
        )
        # Coerce answer to str — LongMemEval has int answers for some count questions
        gold_answer = str(item.answer)
        references.append({
            "question_id": item.question_id,
            "answer": gold_answer,
            "question_type": item.question_type,
        })
        # Incremental save so a later crash doesn't lose work
        if jsonl_path is not None:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "question_id": item.question_id,
                    "hypothesis": answer,
                    "answer": gold_answer,
                    "question_type": item.question_type,
                    "retrieval_ms": retrieval_latencies[-1],
                    "llm_ms": llm_latencies[-1],
                    "hit_at_k": hit,
                    "input_tokens": input_token_estimates[-1],
                }, ensure_ascii=False) + "\n")
        logger.info(
            "[%s] [%d/%d] %s | type=%s | R@%d=%d | retr=%.0fms llm=%.0fms | %s",
            mode, i + 1, len(items),
            item.question_id, item.question_type, top_k, hit,
            retrieval_latencies[-1], llm_latencies[-1],
            answer[:60].replace("\n", " "),
        )

    scores = compute_scores(predictions, references)
    return {
        "method": mode,
        "model": model,
        "scores": scores.to_dict(),
        "retrieval_recall_at_k": sum(retrieved_hits_at_k) / max(1, len(retrieved_hits_at_k)),
        "top_k": top_k,
        "avg_retrieval_ms": sum(retrieval_latencies) / max(1, len(retrieval_latencies)),
        "avg_llm_ms": sum(llm_latencies) / max(1, len(llm_latencies)),
        "avg_input_tokens": sum(input_token_estimates) / max(1, len(input_token_estimates)),
        "n_items": len(items),
        "predictions": predictions,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="LongMemEval QA retrieval comparison")
    p.add_argument("--variant", default="small", choices=["small", "oracle", "medium"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--modes", nargs="+", default=["chroma_rerank", "soma_hybrid"],
                   choices=list(MODES))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    p.add_argument("--top-k", type=int, default=RETRIEVE_TOP_K)
    p.add_argument("--out-suffix", default="",
                   help="appended to output filenames, e.g. '_smoke50'")
    p.add_argument("--sbert-device", default="cpu",
                   choices=["cpu", "cuda"],
                   help="Device for sbert embedder. CPU avoids VRAM "
                        "contention with ollama LLMs on a shared GPU.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"Loading LongMemEval {args.variant}...")
    items = load_dataset(args.variant)
    if args.limit is not None:
        items = items[: args.limit]
    print(f"  {len(items)} items")

    print(f"Loading sbert all-MiniLM-L6-v2 on device={args.sbert_device} ...")
    sbert_model, dim = _build_sbert(device=args.sbert_device)
    embed_fn = lambda t: torch.tensor(sbert_model.encode(t, convert_to_numpy=True))

    print(f"Loading cross-encoder ms-marco-MiniLM-L-6-v2 on device={args.sbert_device} ...")
    reranker = CrossEncoderReranker(device=args.sbert_device)
    _ = reranker.score("warmup", ["test"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        print(f"\n=== {mode} ===")
        jsonl_path = RESULTS_DIR / f"qa_compare_{mode}{args.out_suffix}.jsonl"
        res = _evaluate_mode(
            mode, items,
            sbert_model=sbert_model, dim=dim, embed_fn=embed_fn,
            reranker=reranker,
            api_base=args.api_base, model=args.model,
            max_context_tokens=args.max_context_tokens,
            top_k=args.top_k,
            resume_path=jsonl_path,
        )
        out_path = RESULTS_DIR / f"qa_compare_{mode}{args.out_suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"  F1={res['scores']['f1']:.4f}  R@{args.top_k}={res['retrieval_recall_at_k']:.3f}")
        print(f"  avg retrieval {res['avg_retrieval_ms']:.0f}ms, avg LLM {res['avg_llm_ms']:.0f}ms")
        print(f"  saved {out_path}")
        summary_rows.append({
            "mode": mode,
            "f1": res["scores"]["f1"],
            "em": res["scores"]["em"],
            "rouge1": res["scores"]["rouge1"],
            "rougeL": res["scores"]["rougeL"],
            "recall_at_k": res["retrieval_recall_at_k"],
            "avg_retrieval_ms": res["avg_retrieval_ms"],
            "avg_llm_ms": res["avg_llm_ms"],
            "avg_input_tokens": res["avg_input_tokens"],
        })

    # Summary markdown
    md_path = RESULTS_DIR / f"qa_compare_summary{args.out_suffix}.md"
    lines = [
        "# LongMemEval QA comparison -- retrieval lift -> QA lift?",
        "",
        f"Variant: {args.variant}  |  N={len(items)}  |  top-k={args.top_k}  |  model={args.model}",
        "",
        "| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['mode']} | {r['f1']:.4f} | {r['em']:.4f} | "
            f"{r['rouge1']:.4f} | {r['rougeL']:.4f} | "
            f"{r['recall_at_k']:.3f} | {int(r['avg_input_tokens'])} | "
            f"{r['avg_retrieval_ms']:.0f} | {r['avg_llm_ms']:.0f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
