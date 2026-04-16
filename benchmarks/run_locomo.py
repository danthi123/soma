"""LoCoMo retrieval benchmark — SOMA vs Chroma on real long conversations.

LoCoMo (Maharana et al. 2024) is the de-facto agent-memory benchmark:
10 conversations, ~588 turns each, 1,986 question-answer pairs total
with explicit evidence-turn annotations. The full LoCoMo eval scores
QA accuracy via a GPT-4 judge; we deliberately stop at retrieval
(does the system fetch the evidence turns?) so the benchmark stays
runnable without external API access. Recall@k is the cleanest
signal for the memory layer's job — generation quality is on the
LLM, not the memory.

Per conversation:
1. Store every turn as a separate entry with its dia_id in metadata.
2. For each QA, retrieve top-k against the question text.
3. Score Recall@k = "did at least one evidence turn make it into top-k?"

Aggregate Recall@k across conversations + per category. Report the
SOMA-flat / SOMA-hnsw / Chroma three-way comparison so the same
table answers "is SOMA competitive on a real-world workload?" and
"does the HNSW backend hold up at scale (10K+ entries)?"

Run::

    python -m benchmarks.run_locomo --out benchmarks/reports/locomo.md
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.datasets.locomo import (
    CATEGORY_NAMES,
    LoCoMoQuery,
    LoCoMoTurn,
    load_locomo,
    turns_for_sample,
)
from benchmarks.harness.adapters.chroma import ChromaAdapter
from benchmarks.harness.adapters.soma import ConversationalSomaAdapter, SomaAdapter
from benchmarks.harness.qa_eval import QAEvalResult, evaluate_qa
from soma.llm import backend_from_env
from soma.llm.backends import DryRunBackend

K_VALUES: tuple[int, ...] = (1, 5, 10)


@dataclass
class LoCoMoResult:
    system: str
    n_turns: int
    n_queries: int
    recall_at_k: dict[int, float]
    recall_by_category: dict[str, dict[int, float]]
    store_total_s: float
    retrieve_avg_ms: float
    disk_kb: float
    facts_stored: int = 0
    turns_processed: int = 0
    qa: QAEvalResult | None = None
    # Captured (query, hits, gold) triples per arm so a single --run-qa-eval
    # pass can hit every system fairly — filled only when qa_eval is on.
    _qa_triples: list[tuple[str, list, str]] = field(default_factory=list)


def _score_recall(retrieved_dia_ids: list[str], evidence: list[str], k: int) -> float:
    """1.0 if any evidence dia_id is in retrieved[:k], else 0.0."""
    return 1.0 if any(e in retrieved_dia_ids[:k] for e in evidence) else 0.0


def _run_one_system(
    name: str,
    adapter,
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
    *,
    capture_qa_triples: bool = False,
    qa_max_questions: int | None = None,
    qa_eval_k: int = 10,
) -> LoCoMoResult:
    print(f"  [{name}] preparing...")
    adapter.prepare()

    print(f"  [{name}] storing {len(turns)} turns across 10 conversations...")
    t0 = time.perf_counter()
    # Group turns by sample so we can store per-conversation contexts
    samples = sorted({t.sample_id for t in turns})
    for sid in samples:
        sample_turns = turns_for_sample(turns, sid)
        for turn in sample_turns:
            adapter.store(
                turn.text,
                metadata={"sample_id": sid, "dia_id": turn.dia_id, "speaker": turn.speaker},
            )
    store_total = time.perf_counter() - t0

    # Per-conversation index: collect dia_ids that belong to each sample
    # so we can match retrieved nodes back to evidence dia_ids.
    print(f"  [{name}] running {len(queries)} retrieval queries...")
    recall_sums: dict[int, float] = {k: 0.0 for k in K_VALUES}
    cat_sums: dict[str, dict[int, list[float]]] = {
        CATEGORY_NAMES.get(i, str(i)): {k: [] for k in K_VALUES}
        for i in range(1, 6)
    }
    retrieve_times: list[float] = []
    max_k = max(K_VALUES)

    # Warmup
    adapter.retrieve(queries[0].question, k=max_k)

    qa_triples: list[tuple[str, list, str]] = []
    # Collect an at-most-qa_max_questions slice for the QA eval so we
    # don't burn thousands of LLM calls when real keys are set.
    qa_slice_cap = qa_max_questions if qa_max_questions is not None else len(queries)

    for q in queries:
        t1 = time.perf_counter()
        hits = adapter.retrieve(q.question, k=max_k)
        retrieve_times.append(time.perf_counter() - t1)
        # Filter to same sample (cross-sample matches don't count —
        # would be cheating; LoCoMo's evidence is intra-conversation).
        same_sample = [
            h for h in hits if h.metadata.get("sample_id") == q.sample_id
        ]
        retrieved_dia_ids = [h.metadata.get("dia_id", "") for h in same_sample]
        cat_name = CATEGORY_NAMES.get(q.category, str(q.category))
        for k in K_VALUES:
            score = _score_recall(retrieved_dia_ids, q.evidence, k)
            recall_sums[k] += score
            cat_sums[cat_name][k].append(score)
        if (
            capture_qa_triples
            and len(qa_triples) < qa_slice_cap
            and q.answer  # skip adversarial "no-answer" QAs
        ):
            # qa_eval_k hits from same-sample — wider context lifts accuracy
            # when retrieval recall@5 is already weak on the dataset.
            qa_triples.append((q.question, same_sample[:qa_eval_k], q.answer))

    n = len(queries)
    recall_avg = {k: recall_sums[k] / max(1, n) for k in K_VALUES}
    recall_by_cat = {
        cat: {
            k: sum(scores) / max(1, len(scores))
            for k, scores in by_k.items()
        }
        for cat, by_k in cat_sums.items()
    }
    disk = adapter.disk_footprint_bytes() / 1024.0
    # Conversational adapter exposes extraction counters; defaults to 0.
    facts_stored = int(getattr(adapter, "facts_stored", 0))
    turns_processed = int(getattr(adapter, "turns_processed", 0))
    adapter.teardown()

    return LoCoMoResult(
        system=name,
        n_turns=len(turns),
        n_queries=len(queries),
        recall_at_k=recall_avg,
        recall_by_category=recall_by_cat,
        store_total_s=store_total,
        retrieve_avg_ms=sum(retrieve_times) * 1000 / max(1, len(retrieve_times)),
        disk_kb=disk,
        facts_stored=facts_stored,
        turns_processed=turns_processed,
        _qa_triples=qa_triples,
    )


def _format_main_table(
    results: list[LoCoMoResult],
    *,
    show_facts: bool = False,
    show_qa: bool = False,
) -> str:
    header_cols = ["System"] + [f"R@{k}" for k in K_VALUES] + [
        "Retrieve (ms)", "Store total (s)", "Disk (MB)",
    ]
    if show_facts:
        header_cols.append("Facts / turns")
    if show_qa:
        header_cols.append("QA acc")
    sep = ["---"] + [":---:"] * (len(header_cols) - 1)
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in results:
        cells = [r.system]
        for k in K_VALUES:
            cells.append(f"{r.recall_at_k[k]:.3f}")
        cells.extend([
            f"{r.retrieve_avg_ms:.2f}",
            f"{r.store_total_s:.1f}",
            f"{r.disk_kb / 1024:.1f}",
        ])
        if show_facts:
            if r.turns_processed > 0:
                cells.append(
                    f"{r.facts_stored} / {r.turns_processed}"
                )
            else:
                cells.append("-")
        if show_qa:
            if r.qa is not None and r.qa.n_questions > 0:
                cells.append(
                    f"{r.qa.accuracy:.3f} "
                    f"({r.qa.n_correct}/{r.qa.n_questions})"
                )
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_category_table(results: list[LoCoMoResult], k: int) -> str:
    cat_names = [CATEGORY_NAMES[i] for i in range(1, 6)]
    header_cols = ["System"] + [f"{c} R@{k}" for c in cat_names]
    sep = ["---"] + [":---:"] * len(cat_names)
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in results:
        cells = [r.system]
        for cat in cat_names:
            score = r.recall_by_category.get(cat, {}).get(k, float("nan"))
            cells.append(f"{score:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _maybe_qa_backend(prefer_judge: str | None = None):
    """Pick the best available LLM backend for QA eval, gracefully
    falling back to :class:`DryRunBackend` when nothing is reachable.

    :func:`soma.llm.backend_from_env` falls back to
    :class:`HuggingFaceBackend` by default, which tries to load a
    local HF model on first ``generate`` — that's too heavy for the
    smoke path. For QA eval we want: Ollama > OpenAI > Anthropic >
    DryRun.
    """
    import os

    # Explicit override.
    if prefer_judge:
        return backend_from_env(prefer=prefer_judge)
    # Happy path: one of the three online backends is configured.
    for env_key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(env_key):
            return backend_from_env()
    # Ollama if reachable.
    from soma.llm.backends import _ollama_alive  # noqa: PLC2701

    if _ollama_alive(
        os.environ.get("SOMA_LLM_BASE_URL") or "http://localhost:11434"
    ):
        return backend_from_env()
    # Nothing reachable — the caller will warn + substitute a DryRun.
    return DryRunBackend()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/locomo.md"),
    )
    p.add_argument(
        "--conversational",
        action="store_true",
        help=(
            "Run the --conversational variant: wraps the SOMA adapter in "
            "ConversationalMemory so every turn goes through LLM-driven "
            "fact extraction + reconciliation. Requires an LLM backend "
            "reachable via the SOMA_LLM_BACKEND env (see soma.llm)."
        ),
    )
    p.add_argument(
        "--run-qa-eval",
        action="store_true",
        help=(
            "Enable the LoCoMo-QA LLM-as-judge eval. For each retrieved "
            "context a responder LLM generates an answer; a judge LLM "
            "then compares the answer to the gold annotation. Reports "
            "qa_accuracy per arm. Gated by cost — default off."
        ),
    )
    p.add_argument(
        "--qa-eval-max-questions",
        type=int,
        default=200,
        help=(
            "Cap the number of questions scored per arm. Default 200 so "
            "a typical run stays under $1 on paid APIs. Pass 0 for "
            "unlimited."
        ),
    )
    p.add_argument(
        "--qa-eval-k",
        type=int,
        default=10,
        help=(
            "Number of retrieved hits handed to the responder LLM per "
            "question. Larger k = more context = higher QA accuracy "
            "ceiling but more input tokens. Default 10."
        ),
    )
    p.add_argument(
        "--judge-llm-name",
        type=str,
        default=None,
        help=(
            "Override the judge LLM backend (e.g. 'openai' / 'anthropic' "
            "/ 'ollama'). Same values as SOMA_LLM_BACKEND. Default: "
            "same backend as the responder."
        ),
    )
    args = p.parse_args()

    print("Loading LoCoMo dataset...")
    turns, queries = load_locomo()
    samples = sorted({t.sample_id for t in turns})
    print(
        f"  {len(samples)} conversations, {len(turns)} turns, "
        f"{len(queries)} queries with evidence."
    )

    if args.conversational:
        llm = backend_from_env()
        systems = [
            ("soma-flat", SomaAdapter(use_sbert=True)),
            (
                "soma-conversational",
                ConversationalSomaAdapter(
                    llm=llm, session_id="locomo", summary_every=20,
                ),
            ),
            ("chroma", ChromaAdapter()),
        ]
        args.out = args.out.with_name("locomo_conversational.md")
    else:
        systems = [
            ("soma-flat", SomaAdapter(use_sbert=True)),
            (
                "soma-hnsw",
                SomaAdapter(
                    use_sbert=True, faiss_index_type="hnsw", faiss_threshold=500,
                ),
            ),
            ("chroma", ChromaAdapter()),
        ]

    qa_max = args.qa_eval_max_questions if args.qa_eval_max_questions > 0 else None
    results: list[LoCoMoResult] = []
    for name, adapter in systems:
        print(f"\n=== {name} ===")
        r = _run_one_system(
            name,
            adapter,
            turns,
            queries,
            capture_qa_triples=args.run_qa_eval,
            qa_max_questions=qa_max,
            qa_eval_k=args.qa_eval_k,
        )
        results.append(r)
        print(
            f"  Recall@1={r.recall_at_k[1]:.3f} "
            f"Recall@5={r.recall_at_k[5]:.3f} "
            f"Recall@10={r.recall_at_k[10]:.3f} "
            f"retrieve={r.retrieve_avg_ms:.1f}ms"
        )

    # ---- QA eval (post-hoc so every arm scores on the SAME judge) ----
    if args.run_qa_eval:
        responder = _maybe_qa_backend()
        judge = (
            _maybe_qa_backend(prefer_judge=args.judge_llm_name)
            if args.judge_llm_name
            else responder
        )
        print(
            f"\n=== QA eval === responder={responder.name} judge={judge.name}"
        )
        if isinstance(responder, DryRunBackend):
            print(
                "  WARNING: no live LLM backend reachable "
                "(OPENAI_API_KEY / ANTHROPIC_API_KEY / Ollama). Running "
                "with DryRunBackend -- QA accuracy will be all zeros; "
                "re-run with a real backend to get real numbers."
            )
        total_calls = 0
        total_tokens = 0
        for r in results:
            if not r._qa_triples:
                continue
            print(f"  [{r.system}] scoring {len(r._qa_triples)} questions...")
            r.qa = evaluate_qa(
                r._qa_triples,
                responder_llm=responder,
                judge_llm=judge,
            )
            total_calls += r.qa.llm_calls
            total_tokens += r.qa.est_total_tokens
            print(
                f"    accuracy={r.qa.accuracy:.3f} "
                f"({r.qa.n_correct}/{r.qa.n_questions}) "
                f"llm_calls={r.qa.llm_calls}"
            )
        print(
            f"  TOTAL LLM calls across all arms: {total_calls} "
            f"(~{total_tokens} tokens estimated)"
        )

    show_facts = args.conversational
    show_qa = args.run_qa_eval
    title_suffix = " — conversational mode" if args.conversational else ""
    lines = [
        f"# LoCoMo Retrieval Benchmark — SOMA vs Chroma{title_suffix}",
        "",
        f"**Dataset:** LoCoMo (Maharana et al. 2024) — {len(samples)} "
        f"conversations, {len(turns)} dialogue turns, "
        f"{len(queries)} questions with evidence-turn annotations.",
        "",
        "**What's measured:** retrieval Recall@k (did at least one of "
        "the question's gold-evidence turns make it into the top-k? "
        "Cross-sample retrievals are excluded — LoCoMo's evidence is "
        "intra-conversation so cross-sample hits would be cheating). "
        + (
            "QA accuracy is scored by a judge LLM comparing the "
            "responder's answer against the gold annotation."
            if show_qa
            else "We deliberately do *not* run the LoCoMo paper's GPT-4 "
            "judge for QA accuracy; that part is the LLM's job, not "
            "the memory layer's. Recall@k cleanly isolates the memory "
            "contribution."
        ),
        "",
        "## Headline",
        "",
        _format_main_table(results, show_facts=show_facts, show_qa=show_qa),
        "",
        "## Recall@5 by Question Category",
        "",
        _format_category_table(results, k=5),
        "",
        "## Interpretation",
        "",
        "Single-hop questions (one fact lookup) are the bread-and-butter "
        "vector-retrieval case; SOMA and Chroma should be near-tied. "
        "Multi-hop and temporal questions stress the index more — the "
        "evidence may be split across distant turns. Adversarial "
        "questions are designed to be hard or unanswerable; low Recall "
        "there is expected and the gap between systems is the signal "
        "of interest.",
        "",
        "Open-domain questions are the largest category (841 of 1986); "
        "they ask about facts mentioned anywhere in the conversation "
        "and are the closest to a real RAG-over-conversation workload.",
        "",
        "---",
        "",
        "Generated by `benchmarks/run_locomo.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
