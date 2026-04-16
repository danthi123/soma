"""ConversationalMemory threshold calibration sweep.

Measures how ``near_dup_threshold`` (skip-LLM above this) and
``ambiguous_threshold`` (ADD below this) affect downstream behaviour
on a LoCoMo-like workload. Thresholds default to 0.92 / 0.75 in
:class:`soma.memory.conversational.ConversationalMemory` — sbert
rules-of-thumb inherited from Mem0 that have never been tuned
against our actual data.

The sweep processes the same LoCoMo subset through a fresh
:class:`ConversationalMemory` for every
(near_dup, ambiguous) combination and records:

- ``facts_stored``: total fact-type entries after all turns processed.
- ``llm_calls``: extract + reconcile calls (summary is constant
  across combos so it doesn't move the cost needle).
- ``p50_add_ms`` / ``p95_add_ms``: per-turn ``add_message`` latency.
- ``recall_at_5``: standard LoCoMo recall scored against evidence
  turn IDs.
- ``qa_accuracy`` (optional): LLM-as-judge QA score when
  ``--run-qa-eval`` is set.

Outputs a markdown table at
``benchmarks/reports/conv_threshold_sweep.md`` with a recommendation
banner highlighting the best combination by QA-accuracy-per-LLM-call
(cost-adjusted quality).

Run::

    python -m benchmarks.run_conv_threshold_sweep
    python -m benchmarks.run_conv_threshold_sweep --run-qa-eval

Requires an LLM backend reachable via :func:`soma.llm.backend_from_env`
(Ollama / OpenAI / Anthropic) for real numbers. Falls back to
:class:`DryRunBackend` with a warning so the harness smoke-runs
anywhere.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.datasets.locomo import (
    LoCoMoQuery,
    LoCoMoTurn,
    load_locomo,
)
from benchmarks.harness.adapters.base import BenchmarkHit
from benchmarks.harness.qa_eval import evaluate_qa
from soma.llm.backends import DryRunBackend, LLMBackend
from soma.memory import ConversationalMemory, MemoryLayer

# Thresholds to sweep. The grid is symmetric around the shipped
# defaults (0.92, 0.75) so we see data at the corners and can
# confidently say whether the defaults were well-chosen.
NEAR_DUP_GRID: tuple[float, ...] = (0.88, 0.90, 0.92, 0.94)
AMBIGUOUS_GRID: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80)


@dataclass
class SweepRow:
    """One row of the threshold-sweep table — one combo's metrics."""

    near_dup: float
    ambiguous: float
    facts_stored: int
    llm_calls: int
    turns_processed: int
    p50_add_ms: float
    p95_add_ms: float
    recall_at_5: float
    qa_accuracy: float | None = None
    qa_questions: int = 0


class _CountingBackend:
    """Wraps an LLMBackend, counts every ``generate`` call.

    Mirrors the protocol, so :class:`ConversationalMemory` sees a
    normal backend. Exposes ``call_count`` so the sweep row can
    record cost.
    """

    def __init__(self, inner: LLMBackend) -> None:
        self._inner = inner
        self.name = f"counting:{inner.name}"
        self.call_count = 0

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.call_count += 1
        return self._inner.generate(prompt, max_tokens=max_tokens)


def _build_memory(
    *,
    embed_fn: Callable[[str], torch.Tensor] | None,
    embed_dim: int | None,
    use_sbert: bool,
) -> MemoryLayer:
    """Build a fresh MemoryLayer for one sweep combo.

    Tests pass a stub embed_fn so the sweep runs offline and
    deterministically. Production runs (use_sbert=True) go through
    sentence-transformers to match the rest of the LoCoMo bench.
    """
    if embed_fn is not None:
        assert embed_dim is not None
        return MemoryLayer(embed_fn=embed_fn, embed_dim=embed_dim)
    if use_sbert:
        return MemoryLayer.with_sbert("all-MiniLM-L6-v2")
    # Tiny-BPE fallback — used if sbert is unavailable (offline dev).
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

    tok = train_bpe_tokenizer(["placeholder"], vocab_size=128)
    enc = TextEncoder(tok, embed_dim=32, max_seq_len=128)
    return MemoryLayer(tokenizer=tok, encoder=enc)


def _score_recall_at_5(hits: list[BenchmarkHit], evidence: list[str]) -> float:
    dia_ids = [h.metadata.get("dia_id", "") for h in hits[:5]]
    return 1.0 if any(e in dia_ids for e in evidence) else 0.0


def _run_one_combo(
    *,
    near_dup: float,
    ambiguous: float,
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
    llm_factory: Callable[[], LLMBackend],
    embed_fn: Callable[[str], torch.Tensor] | None,
    embed_dim: int | None,
    use_sbert: bool,
    capture_qa_triples: bool,
    qa_max_questions: int | None,
) -> tuple[SweepRow, list[tuple[str, list[BenchmarkHit], str]]]:
    """Process ``turns`` through a fresh ConversationalMemory + score."""
    memory = _build_memory(
        embed_fn=embed_fn, embed_dim=embed_dim, use_sbert=use_sbert,
    )
    llm = _CountingBackend(llm_factory())
    cm = ConversationalMemory(
        memory=memory,
        llm=llm,
        session_id="sweep",
        near_dup_threshold=near_dup,
        ambiguous_threshold=ambiguous,
        summary_every=1_000_000,  # avoid summary noise in the sweep
    )

    per_turn_ms: list[float] = []
    for turn in turns:
        t0 = time.perf_counter()
        cm.add_message(
            "user",
            turn.text,
            metadata={
                "sample_id": turn.sample_id,
                "dia_id": turn.dia_id,
                "speaker": turn.speaker,
            },
        )
        per_turn_ms.append((time.perf_counter() - t0) * 1000.0)

    facts_stored = sum(
        1 for m in memory._metadatas if m.get("type") == "fact"
    )

    # Retrieval + optional QA-triple capture.
    recall_scores: list[float] = []
    qa_triples: list[tuple[str, list[BenchmarkHit], str]] = []
    qa_slice_cap = qa_max_questions if qa_max_questions is not None else len(queries)
    for q in queries:
        hits = cm.retrieve(q.question, k=5)
        same_sample = [
            BenchmarkHit(
                text=h.text,
                score=h.score,
                metadata=h.metadata,
                node_id=h.node_id,
            )
            for h in hits
            if h.metadata.get("sample_id") == q.sample_id
        ]
        recall_scores.append(_score_recall_at_5(same_sample, q.evidence))
        if (
            capture_qa_triples
            and q.answer
            and len(qa_triples) < qa_slice_cap
        ):
            qa_triples.append((q.question, same_sample, q.answer))

    p50 = statistics.median(per_turn_ms) if per_turn_ms else 0.0
    p95 = (
        statistics.quantiles(per_turn_ms, n=20)[18]
        if len(per_turn_ms) >= 2
        else (per_turn_ms[0] if per_turn_ms else 0.0)
    )
    recall_at_5 = sum(recall_scores) / max(1, len(recall_scores))

    row = SweepRow(
        near_dup=near_dup,
        ambiguous=ambiguous,
        facts_stored=facts_stored,
        llm_calls=llm.call_count,
        turns_processed=len(turns),
        p50_add_ms=p50,
        p95_add_ms=p95,
        recall_at_5=recall_at_5,
    )
    return row, qa_triples


def run_sweep(
    *,
    turns: list[LoCoMoTurn],
    queries: list[LoCoMoQuery],
    llm_factory: Callable[[], LLMBackend],
    embed_fn: Callable[[str], torch.Tensor] | None = None,
    embed_dim: int | None = None,
    use_sbert: bool = False,
    run_qa_eval: bool = False,
    qa_responder_llm: LLMBackend | None = None,
    qa_judge_llm: LLMBackend | None = None,
    qa_max_questions: int | None = 100,
    progress: bool = False,
) -> list[SweepRow]:
    """Run the 4x4 threshold sweep. Returns one :class:`SweepRow` per combo.

    Uses the same fresh-memory-per-combo pattern so each combo is
    independent. ``llm_factory`` builds a new backend per combo so
    counters reset cleanly.
    """
    rows: list[SweepRow] = []
    total = len(NEAR_DUP_GRID) * len(AMBIGUOUS_GRID)
    idx = 0
    for near_dup in NEAR_DUP_GRID:
        for ambiguous in AMBIGUOUS_GRID:
            idx += 1
            if ambiguous > near_dup:
                # The ConversationalMemory constructor rejects
                # ambiguous > near_dup. Emit a degenerate row so the
                # grid stays 4x4 — makes the table uniform and keeps
                # the recommendation logic simple.
                rows.append(
                    SweepRow(
                        near_dup=near_dup,
                        ambiguous=ambiguous,
                        facts_stored=0,
                        llm_calls=0,
                        turns_processed=0,
                        p50_add_ms=0.0,
                        p95_add_ms=0.0,
                        recall_at_5=float("nan"),
                    )
                )
                continue
            if progress:
                print(
                    f"  [{idx}/{total}] near_dup={near_dup} "
                    f"ambiguous={ambiguous}"
                )
            row, qa_triples = _run_one_combo(
                near_dup=near_dup,
                ambiguous=ambiguous,
                turns=turns,
                queries=queries,
                llm_factory=llm_factory,
                embed_fn=embed_fn,
                embed_dim=embed_dim,
                use_sbert=use_sbert,
                capture_qa_triples=run_qa_eval,
                qa_max_questions=qa_max_questions,
            )
            if run_qa_eval and qa_triples and qa_responder_llm is not None:
                judge = qa_judge_llm or qa_responder_llm
                qa = evaluate_qa(
                    qa_triples,
                    responder_llm=qa_responder_llm,
                    judge_llm=judge,
                )
                row.qa_accuracy = qa.accuracy
                row.qa_questions = qa.n_questions
                # Count QA-eval LLM calls separately from build-time
                # calls so the cost-per-combo stays interpretable.
                row.llm_calls += qa.llm_calls
            rows.append(row)
    return rows


def _recommend(rows: list[SweepRow]) -> SweepRow | None:
    """Pick the row that maximizes quality-per-LLM-call.

    Metric: ``qa_accuracy / llm_calls`` when QA eval ran, else
    ``recall_at_5 / llm_calls``. NaN/0 combos excluded.
    """
    best: SweepRow | None = None
    best_score = -float("inf")
    for r in rows:
        if r.turns_processed == 0 or r.llm_calls == 0:
            continue
        if r.qa_accuracy is not None:
            quality = r.qa_accuracy
        elif r.recall_at_5 == r.recall_at_5:  # not NaN
            quality = r.recall_at_5
        else:
            continue
        score = quality / max(1, r.llm_calls)
        if score > best_score:
            best_score = score
            best = r
    return best


def render_report(
    rows: list[SweepRow],
    *,
    n_turns: int,
    n_queries: int,
    backend_name: str | None = None,
) -> str:
    """Render a markdown report with the sweep table + recommendation."""
    show_qa = any(r.qa_accuracy is not None for r in rows)
    header = ["near_dup", "ambiguous", "facts", "llm calls", "p50 add (ms)",
              "p95 add (ms)", "R@5"]
    if show_qa:
        header.append("QA acc")
    sep = [":---:"] * len(header)

    is_dry_run = backend_name is not None and "dry-run" in backend_name

    lines = [
        "# Conversational Memory — Threshold Calibration Sweep",
        "",
        f"**Corpus:** {n_turns} turns, {n_queries} queries.",
        "",
    ]
    if backend_name:
        lines.append(f"**LLM backend:** `{backend_name}`.")
        lines.append("")
    if is_dry_run:
        lines.append(
            "> **NOTE:** this is a smoke run with `DryRunBackend` — the "
            "LLM doesn't emit JSON so every extract is dropped and no "
            "facts land. The table proves the harness runs end-to-end; "
            "re-run with `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` set or "
            "Ollama running to get real numbers."
        )
        lines.append("")

    lines.extend([
        "**What's measured:** for every (near_dup_threshold, "
        "ambiguous_threshold) combination we build a fresh "
        "`ConversationalMemory`, process every turn, and record "
        "facts_stored, llm_calls (extract + reconcile, and QA "
        "responder/judge when --run-qa-eval is set), p50/p95 "
        "add_message latency, and Recall@5 against LoCoMo evidence "
        "turns. Cost-adjusted quality = accuracy / llm_calls; the "
        "recommended row is the highest such score.",
        "",
        "## Sweep table",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ])
    for r in rows:
        cells = [
            f"{r.near_dup:.2f}",
            f"{r.ambiguous:.2f}",
            str(r.facts_stored),
            str(r.llm_calls),
            f"{r.p50_add_ms:.2f}",
            f"{r.p95_add_ms:.2f}",
        ]
        if r.recall_at_5 != r.recall_at_5:  # NaN → skipped combo
            cells.append("-")
        else:
            cells.append(f"{r.recall_at_5:.3f}")
        if show_qa:
            cells.append(
                f"{r.qa_accuracy:.3f}" if r.qa_accuracy is not None else "-"
            )
        lines.append("| " + " | ".join(cells) + " |")

    rec = _recommend(rows)
    if rec is not None:
        lines.extend([
            "",
            "## Recommendation",
            "",
            f"**Recommended defaults:** `near_dup_threshold={rec.near_dup:.2f}`, "
            f"`ambiguous_threshold={rec.ambiguous:.2f}` — best "
            f"quality-per-LLM-call on this workload.",
            "",
            f"- Facts stored: {rec.facts_stored}",
            f"- LLM calls: {rec.llm_calls}",
            f"- p50 add_message: {rec.p50_add_ms:.2f} ms",
            f"- Recall@5: {rec.recall_at_5:.3f}",
        ])
        if rec.qa_accuracy is not None:
            lines.append(f"- QA accuracy: {rec.qa_accuracy:.3f}")
    else:
        lines.extend([
            "",
            "## Recommendation",
            "",
            "No recommendation emitted — every combo had zero LLM calls "
            "(DryRunBackend or empty corpus). Re-run with a live backend "
            "to produce a data-backed default.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "Generated by `benchmarks/run_conv_threshold_sweep.py`.",
    ])
    return "\n".join(lines) + "\n"


def _pick_qa_backend(prefer: str | None = None) -> LLMBackend:
    """QA responder/judge picker — Ollama > OpenAI > Anthropic > DryRun.

    Mirrors :func:`benchmarks.run_locomo._maybe_qa_backend` so the two
    benches share the same "don't accidentally blast the HF loader"
    fallback behaviour.
    """
    import os

    from soma.llm import backend_from_env
    from soma.llm.backends import _ollama_alive  # noqa: PLC2701

    if prefer:
        return backend_from_env(prefer=prefer)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(key):
            return backend_from_env()
    if _ollama_alive(
        os.environ.get("SOMA_LLM_BASE_URL") or "http://localhost:11434"
    ):
        return backend_from_env()
    return DryRunBackend()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/conv_threshold_sweep.md"),
    )
    p.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of LoCoMo conversations to include (default 20).",
    )
    p.add_argument(
        "--run-qa-eval",
        action="store_true",
        help=(
            "Score each combo with the LLM-as-judge QA eval. Adds ~2 "
            "LLM calls per question per combo — gated by cost."
        ),
    )
    p.add_argument(
        "--qa-eval-max-questions",
        type=int,
        default=50,
        help=(
            "Cap on QA questions per combo. 50 x 16 combos = 800 "
            "judge calls max on paid APIs."
        ),
    )
    p.add_argument(
        "--judge-llm-name",
        type=str,
        default=None,
        help=(
            "Override judge LLM backend (same values as SOMA_LLM_BACKEND)."
        ),
    )
    args = p.parse_args()

    print("Loading LoCoMo dataset...")
    all_turns, all_queries = load_locomo()
    sample_ids = sorted({t.sample_id for t in all_turns})[: args.samples]
    turns = [t for t in all_turns if t.sample_id in sample_ids]
    queries = [q for q in all_queries if q.sample_id in sample_ids]
    print(
        f"  {len(sample_ids)} conversations, {len(turns)} turns, "
        f"{len(queries)} queries with evidence."
    )

    # Build factory so each combo gets a fresh counting backend.
    import os

    from soma.llm import backend_from_env
    from soma.llm.backends import _ollama_alive  # noqa: PLC2701

    has_live_backend = bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or _ollama_alive(
            os.environ.get("SOMA_LLM_BASE_URL") or "http://localhost:11434"
        )
    )
    def _live_factory() -> LLMBackend:
        return backend_from_env()

    def _dry_factory() -> LLMBackend:
        return DryRunBackend()

    if has_live_backend:
        llm_factory: Callable[[], LLMBackend] = _live_factory
    else:
        print(
            "  WARNING: no live LLM backend reachable "
            "(OPENAI_API_KEY / ANTHROPIC_API_KEY / Ollama). "
            "Running with DryRunBackend — the sweep will emit a "
            "smoke report; re-run with a real backend for real data."
        )
        llm_factory = _dry_factory

    responder = _pick_qa_backend()
    judge = _pick_qa_backend(args.judge_llm_name) if args.judge_llm_name else responder

    rows = run_sweep(
        turns=turns,
        queries=queries,
        llm_factory=llm_factory,
        use_sbert=True,
        run_qa_eval=args.run_qa_eval,
        qa_responder_llm=responder if args.run_qa_eval else None,
        qa_judge_llm=judge if args.run_qa_eval else None,
        qa_max_questions=args.qa_eval_max_questions,
        progress=True,
    )

    # Probe the factory once for a representative name so the report
    # records which backend produced the numbers.
    probe_backend = llm_factory()
    backend_name = probe_backend.name

    report = render_report(
        rows,
        n_turns=len(turns),
        n_queries=len(queries),
        backend_name=backend_name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
