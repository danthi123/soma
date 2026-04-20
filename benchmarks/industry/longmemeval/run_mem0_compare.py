"""Head-to-head vs Mem0 on LongMemEval QA.

Mem0 is the most prominent agent-memory competitor (80K+ stars, $24M
funded). It's designed around LLM-extracted facts rather than raw-turn
storage — a fundamentally different strategy than SOMA's BM25+cosine
hybrid over raw messages.

This script runs a fair head-to-head:
  - Same LongMemEval item
  - Same LLM for QA (qwen3.5:4b-q8_0 via Ollama)
  - Same embedder (sentence-transformers/all-MiniLM-L6-v2)
  - Mem0 uses its default `infer=True` mode (LLM-extracted facts during
    ingest) — its designed use case.
  - SOMA uses `hybrid_alpha=0.3` — its winning config.

Warning: Mem0 with `infer=True` makes an LLM call PER turn during
ingest. With ~500 turns per LongMemEval item, this is SLOW. Default
scope is small (N=10-20) for a qualitative head-to-head.

Run::

    python -m benchmarks.industry.longmemeval.run_mem0_compare \\
        --variant small --limit 10

Do NOT run while another ollama-consuming job is active — they'll
serialize and both slow down.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from benchmarks.industry.longmemeval.data_loader import (
    LongMemEvalItem,
    load_dataset,
)
from benchmarks.industry.longmemeval.metrics import (
    compute_scores,
    token_f1,
)

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:4b-q8_0"
MAX_CONTEXT_TOKENS = 3800
CHARS_PER_TOKEN = 4
RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


SYSTEM_PROMPT_VERBOSE = (
    "You are an AI assistant that recalls information from past conversations. "
    "Answer the question using ONLY the evidence provided. Be concise and direct. "
    "If the evidence does not contain the answer, say 'I don't know'."
)

SYSTEM_PROMPT_STRICT = (
    "Answer with ONLY the specific fact in 1-5 words. "
    "No explanation, no preamble (e.g. 'Based on...', 'According to...'). "
    "Extract the single value that answers the question. "
    "If the evidence does not contain the answer, reply exactly 'I don't know'. "
    "Examples:\n"
    "  Question: What's my favorite brand?  Answer: Nike\n"
    "  Question: How many pages are left?  Answer: 190\n"
    "  Question: Where did I travel?  Answer: Hawaii"
)


def _call_llm(
    context: str,
    question: str,
    question_date: str,
    *,
    api_base: str,
    model: str,
    strict_prompt: bool = False,
) -> str:
    import requests

    system_prompt = (
        SYSTEM_PROMPT_STRICT if strict_prompt else SYSTEM_PROMPT_VERBOSE
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
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return ""


def _pack_text(chunks: list[str], max_tokens: int) -> str:
    budget = max_tokens * CHARS_PER_TOKEN
    lines, used = [], 0
    for i, text in enumerate(chunks):
        block = f"--- Evidence #{i+1} ---\n{text}"
        cost = len(block) + (2 if lines else 0)
        if used + cost > budget:
            break
        lines.append(block)
        used += cost
    return "\n\n".join(lines)


def _session_text(session, session_date: str) -> str:
    header = f"[Session date: {session_date}]\n" if session_date else ""
    body = "\n".join(f"{t.role}: {t.content}" for t in session)
    return header + body


# ---------------------------------------------------------------------------
# Mem0 mode
# ---------------------------------------------------------------------------


def _run_mem0(item: LongMemEvalItem, *, api_base: str, model: str, top_k: int,
              user_id: str, infer: bool, strict_prompt: bool = False) -> dict[str, Any]:
    from mem0 import Memory

    # Ephemeral tmp dir per item (Mem0 stores to disk)
    tmp_dir = Path(f"./tmp_mem0_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": model,
                "ollama_base_url": api_base,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": f"m0_{uuid.uuid4().hex[:8]}",
                "path": str(tmp_dir),
            },
        },
    }

    t_ing = time.perf_counter()
    mem = Memory.from_config(config)
    # Ingest as messages grouped by session. infer=True makes Mem0 extract
    # atomic facts via LLM. infer=False stores raw messages.
    for i, sess in enumerate(item.haystack_sessions):
        sess_date = (
            item.haystack_dates[i] if i < len(item.haystack_dates) else ""
        )
        messages = [
            {"role": t.role, "content": f"[{sess_date}] {t.content}"}
            for t in sess
        ]
        if messages:
            mem.add(messages, user_id=user_id, infer=infer)
    ingest_s = time.perf_counter() - t_ing

    t_retr = time.perf_counter()
    hits = mem.search(item.question, top_k=top_k, filters={"user_id": user_id})
    retrieve_ms = (time.perf_counter() - t_retr) * 1000

    # Mem0 returns {"results": [...]} with each result having "memory"
    mem_texts = [r["memory"] for r in hits.get("results", [])]
    context = _pack_text(mem_texts, MAX_CONTEXT_TOKENS)

    t_llm = time.perf_counter()
    answer = _call_llm(
        context, item.question, item.question_date,
        api_base=api_base, model=model, strict_prompt=strict_prompt,
    )
    llm_ms = (time.perf_counter() - t_llm) * 1000

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {
        "question_id": item.question_id,
        "question_type": item.question_type,
        "answer": str(item.answer),
        "hypothesis": answer,
        "n_memories_retrieved": len(mem_texts),
        "ingest_s": ingest_s,
        "retrieve_ms": retrieve_ms,
        "llm_ms": llm_ms,
        "input_chars": sum(len(t) for t in mem_texts),
    }


# ---------------------------------------------------------------------------
# SOMA mode (same as run_qa_compare soma_hybrid)
# ---------------------------------------------------------------------------


def _run_soma(item: LongMemEvalItem, *, sbert_model, dim: int, embed_fn,
              api_base: str, model: str, top_k: int,
              strict_prompt: bool = False) -> dict[str, Any]:
    from soma.memory import MemoryLayer

    t_ing = time.perf_counter()
    mem = MemoryLayer.ephemeral(embed_fn=embed_fn, embed_dim=dim)
    raw_ids = item.haystack_session_ids
    uniq_ids = [f"{sid}__{j}" for j, sid in enumerate(raw_ids)]
    for i, sess in enumerate(item.haystack_sessions):
        sess_date = (
            item.haystack_dates[i] if i < len(item.haystack_dates) else ""
        )
        mem.store(_session_text(sess, sess_date), metadata={"sid": uniq_ids[i]})
    ingest_s = time.perf_counter() - t_ing

    t_retr = time.perf_counter()
    hits = mem.retrieve(item.question, k=top_k, hybrid_alpha=0.3)
    retrieve_ms = (time.perf_counter() - t_retr) * 1000

    texts = [h.text for h in hits]
    context = _pack_text(texts, MAX_CONTEXT_TOKENS)

    t_llm = time.perf_counter()
    answer = _call_llm(
        context, item.question, item.question_date,
        api_base=api_base, model=model, strict_prompt=strict_prompt,
    )
    llm_ms = (time.perf_counter() - t_llm) * 1000

    return {
        "question_id": item.question_id,
        "question_type": item.question_type,
        "answer": str(item.answer),
        "hypothesis": answer,
        "n_memories_retrieved": len(texts),
        "ingest_s": ingest_s,
        "retrieve_ms": retrieve_ms,
        "llm_ms": llm_ms,
        "input_chars": sum(len(t) for t in texts),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="small", choices=["small", "oracle", "medium"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--modes", nargs="+", default=["mem0_infer", "mem0_raw", "soma_hybrid"],
                   choices=["mem0_infer", "mem0_raw", "soma_hybrid"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out-suffix", default="")
    p.add_argument("--strict-prompt", action="store_true",
                   help="Use strict 1-5 word answer prompt (matches run_qa_compare)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print(f"Loading LongMemEval {args.variant} (limit={args.limit})...")
    items = load_dataset(args.variant)[: args.limit]
    print(f"  {len(items)} items")

    # Lazy sbert for SOMA mode only
    sbert = dim = embed_fn = None
    if "soma_hybrid" in args.modes:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        dim = sbert.get_sentence_embedding_dimension()
        embed_fn = lambda t: torch.tensor(sbert.encode(t, convert_to_numpy=True))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for mode in args.modes:
        print(f"\n=== {mode} ===")
        rows: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            t0 = time.perf_counter()
            try:
                if mode == "mem0_infer":
                    row = _run_mem0(
                        item, api_base=args.api_base, model=args.model,
                        top_k=args.top_k, user_id=item.question_id, infer=True,
                        strict_prompt=args.strict_prompt,
                    )
                elif mode == "mem0_raw":
                    row = _run_mem0(
                        item, api_base=args.api_base, model=args.model,
                        top_k=args.top_k, user_id=item.question_id, infer=False,
                        strict_prompt=args.strict_prompt,
                    )
                elif mode == "soma_hybrid":
                    row = _run_soma(
                        item, sbert_model=sbert, dim=dim, embed_fn=embed_fn,
                        api_base=args.api_base, model=args.model,
                        top_k=args.top_k,
                        strict_prompt=args.strict_prompt,
                    )
                else:
                    raise ValueError(f"unknown mode: {mode}")
                total_s = time.perf_counter() - t0
                rows.append(row)
                logger.info(
                    "[%s] [%d/%d] %s | %.1fs | ingest=%.1fs retr=%.0fms llm=%.0fms | %s",
                    mode, i+1, len(items), item.question_id, total_s,
                    row["ingest_s"], row["retrieve_ms"], row["llm_ms"],
                    row["hypothesis"][:60].replace("\n", " "),
                )
            except Exception as exc:
                logger.error("Item %s failed in %s: %s", item.question_id, mode, exc)
                continue

        # Score
        preds = [{"question_id": r["question_id"], "hypothesis": r["hypothesis"]} for r in rows]
        refs = [{"question_id": r["question_id"], "answer": r["answer"], "question_type": r["question_type"]} for r in rows]
        scores = compute_scores(preds, refs)

        out_path = RESULTS_DIR / f"mem0_compare_{mode}{args.out_suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "method": mode,
                "model": args.model,
                "scores": scores.to_dict(),
                "n_items": len(rows),
                "avg_ingest_s": sum(r["ingest_s"] for r in rows) / max(1, len(rows)),
                "avg_retrieve_ms": sum(r["retrieve_ms"] for r in rows) / max(1, len(rows)),
                "avg_llm_ms": sum(r["llm_ms"] for r in rows) / max(1, len(rows)),
                "avg_input_chars": sum(r["input_chars"] for r in rows) / max(1, len(rows)),
                "predictions": preds,
                "rows": rows,
            }, f, indent=2, ensure_ascii=False)
        print(f"  F1={scores.f1:.4f}  avg ingest={sum(r['ingest_s'] for r in rows)/max(1,len(rows)):.1f}s")
        summary.append({"mode": mode, "f1": scores.f1, "n": len(rows),
                        "avg_ingest_s": sum(r["ingest_s"] for r in rows)/max(1,len(rows))})

    print("\n== Summary ==")
    for r in summary:
        print(f"  {r['mode']:15s} F1={r['f1']:.4f}  (N={r['n']}, avg ingest {r['avg_ingest_s']:.1f}s)")


if __name__ == "__main__":
    main()
