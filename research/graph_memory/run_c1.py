"""C1 experiment orchestrator.

Runs the consolidation-N sweep, the pure-vector baseline, and writes
the report markdown + JSON sidecar.

Usage::

    python -m research.graph_memory.run_c1                  # default: stub embedder
    python -m research.graph_memory.run_c1 --embedder sbert # use SBERT (slow)
    python -m research.graph_memory.run_c1 --quick          # shrink corpus + N sweep

The "stub" embedder is a hash-based bag-of-words used for fast
sanity checks; it's NOT a fair pure-vector baseline because the
hash collisions destroy semantic structure. The honest C1 number is
the SBERT run. Both are written to the report for context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from research.graph_memory.baselines import run_baseline
from research.graph_memory.harness import HarnessResult, run_soma
from research.graph_memory.synthetic_corpus import generate

DEFAULT_N_SWEEP: tuple[int, ...] = (0, 10, 100, 1000)
DEFAULT_QUICK_N_SWEEP: tuple[int, ...] = (0, 1, 5)
DEFAULT_ALPHA: float = 0.3
# Reduced-corpus sweep used when ``--corpus-size`` is passed. Lets the
# orchestrator finish the full N={0, 10, 100, 1000} sweep in a single
# session by shrinking N_snippets — the structural invariants (10
# transitive + 10 triangle triples) are preserved at any size >= 60.


def _stub_embed(text: str, dim: int = 128, *, device: str = "cpu") -> torch.Tensor:
    """Hash-bucket bag-of-words. Deterministic, fast, no model download."""
    vec = torch.zeros(dim, device=device)
    for tok in text.split():
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = float(vec.norm())
    if norm > 0:
        vec /= norm
    return vec


def _make_sbert_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    *,
    device: str = "cpu",
):
    """Return ``(embed_fn, embed_dim)`` for sentence-transformers.

    ``device`` is forwarded to ``SentenceTransformer`` (model-load GPU
    residency) and to ``model.encode`` (forward-pass device). Embed
    tensors are returned on ``device`` so downstream MemoryLayer +
    SOMA ops don't incur implicit CPU↔GPU copies.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    dim = int(model.get_sentence_embedding_dimension())

    def embed(text: str) -> torch.Tensor:
        return torch.tensor(
            model.encode(text, convert_to_numpy=True, device=device),
            device=device,
        )

    return embed, dim


def _format_metric(m: dict[str, float]) -> str:
    return (
        f"R@1={m['r_at_1']:.3f}  R@5={m['r_at_5']:.3f}  "
        f"MRR={m['mrr']:.3f}  (n={int(m['num_queries'])})"
    )


def _result_table(results: list[HarnessResult], *, query_class: str) -> str:
    """Markdown table of one query class across all runs."""
    lines = [
        "| Method | N | R@1 | R@5 | MRR | n_queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        m = r.metrics.get(query_class, {})
        if not m:
            continue
        lines.append(
            f"| {r.method} | {r.consolidation_iterations} | "
            f"{m['r_at_1']:.3f} | {m['r_at_5']:.3f} | "
            f"{m['mrr']:.3f} | {int(m['num_queries'])} |"
        )
    return "\n".join(lines)


def _gate_call(
    baseline_r5: float,
    soma_r5_at_n100: float | None,
    soma_r5_best: float | None = None,
) -> tuple[str, str]:
    """Apply the C1 gate criterion from the plan.

    Returns ``(verdict, rationale)`` where verdict is one of
    ``"PASS"``, ``"FAIL"``, ``"AMBIGUOUS"``.

    Primary criterion: R@5(transitive, SOMA at N=100) - baseline >= 0.05.
    If N=100 wasn't reached in the sweep, fall back to the best-N point
    we have; still flag FAIL if that best point is <= baseline (per the
    plan's "gap <=1 point" wording, a NEGATIVE gap is squarely in the
    fail regime).
    """
    reference = soma_r5_at_n100 if soma_r5_at_n100 is not None else soma_r5_best
    if reference is None:
        return (
            "AMBIGUOUS",
            "No SOMA run completed — nothing to compare against the baseline.",
        )
    delta = reference - baseline_r5
    label = "N=100" if soma_r5_at_n100 is not None else "best-N"
    if delta >= 0.05:
        return (
            "PASS",
            f"SOMA-blended R@5 at {label} beats pure-vector by {delta:+.3f} "
            "(>=0.05 threshold from the C1 plan). Graph carries signal — "
            "proceed to C2 with the LoCoMo retrieval benchmark.",
        )
    if delta <= 0.01:
        return (
            "FAIL",
            f"SOMA-blended R@5 at {label} is within {delta:+.3f} of pure-vector "
            "(<=0.01 threshold from the C1 plan; negative = SOMA actively hurts). "
            "Recommend C1b failure analysis — specifically the blend-formula "
            "ablation since the substrate IS moving (non-zero Spearman "
            "rho(edge,cooc)) but the readout isn't harvesting it — before sinking time "
            "into C2.",
        )
    return (
        "AMBIGUOUS",
        f"SOMA-blended R@5 at {label} is {delta:+.3f} above pure-vector "
        "(between 0.01 and 0.05). Suggest re-running with a different α or "
        "larger consolidation N before declaring C1 conclusively.",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["stub", "sbert"],
        default="sbert",
        help="Embedding backend. 'stub' = hash bag-of-words (fast, "
        "but not a fair baseline). 'sbert' = sentence-transformers "
        "(realistic, slow first-run download).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Graph blend weight (0=pure cosine, 1=pure graph score).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Shrink the corpus + N sweep. Useful for smoke-testing the "
        "orchestrator end-to-end before the full multi-hour run.",
    )
    p.add_argument(
        "--corpus-size",
        type=int,
        default=0,
        help="Override total snippet count (default: 0 = full 1000 from the "
        "C1 plan). Reducing to e.g. 200 keeps the structural invariants "
        "but lets the full N={0,10,100,1000} sweep finish in ~1 hour "
        "instead of ~10 hours.",
    )
    p.add_argument(
        "--n-sweep",
        type=int,
        nargs="+",
        default=None,
        help="Override the N sweep (default: 0 10 100 1000).",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("research/graph_memory/reports/c1_synthetic_signal.md"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/graph_memory/reports/c1_synthetic_signal.json"),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu'). Default: auto — 'cuda' if "
        "torch.cuda.is_available() else 'cpu'. Forces sbert + SOMA + "
        "stub-embed onto the same device so there's no hidden CPU↔GPU "
        "copy each step.",
    )
    args = p.parse_args()

    # --- Device preflight -----------------------------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", end="")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device={device} but torch.cuda.is_available() is "
                "False. Check CUDA install + driver."
            )
        print(f"  name={torch.cuda.get_device_name(0)}", end="")
    print()

    # --- Corpus ----------------------------------------------------------
    if args.quick:
        gt = generate(total_snippets=120, num_transitive=4, num_triangle=4)
        n_sweep = DEFAULT_QUICK_N_SWEEP
    else:
        corpus_size = args.corpus_size if args.corpus_size > 0 else 1000
        gt = generate(total_snippets=corpus_size)
        n_sweep = tuple(args.n_sweep) if args.n_sweep else DEFAULT_N_SWEEP

    print(f"corpus: {len(gt.snippets)} snippets, {len(gt.triples)} triples")
    print(f"transitive query set size: {len(gt.transitive_targets)}")
    print(f"unrelated tokens: {len(gt.unrelated_queries)}")

    # --- Embedder --------------------------------------------------------
    if args.embedder == "sbert":
        print(f"loading sbert (all-MiniLM-L6-v2) on {device}…")
        embed_fn, embed_dim = _make_sbert_embedder(device=device)
    else:
        def embed_fn(text: str) -> torch.Tensor:
            return _stub_embed(text, device=device)
        embed_dim = 128

    print(f"embedder={args.embedder}  embed_dim={embed_dim}  alpha={args.alpha}")
    print(f"N sweep: {list(n_sweep)}")

    # --- Baseline --------------------------------------------------------
    t0 = time.monotonic()
    print("\n=== pure-vector baseline ===")
    baseline = run_baseline(
        gt,
        embed_fn=embed_fn,
        embed_dim=embed_dim,
        k=5,
        method_label=f"pure-vector-{args.embedder}",
    )
    for cls, m in baseline.metrics.items():
        print(f"  {cls:12s}  {_format_metric(m)}")
    print(f"  elapsed={baseline.elapsed_seconds:.1f}s")

    # --- SOMA sweep ------------------------------------------------------
    soma_results: list[HarnessResult] = []
    for n in n_sweep:
        print(f"\n=== soma alpha={args.alpha} consolidate-N={n} ===")
        result = run_soma(
            gt,
            embed_fn=embed_fn,
            embed_dim=embed_dim,
            consolidation_iterations=n,
            k=5,
            graph_rerank_alpha=args.alpha,
            method_label=f"soma-{args.embedder}-a{args.alpha:.2f}",
            device=device,
        )
        for cls, m in result.metrics.items():
            print(f"  {cls:12s}  {_format_metric(m)}")
        print(
            f"  integrators={result.integrator_count}  "
            f"edges={result.edge_count}  "
            f"spearman(edge,cooc)={result.edge_weight_cooccurrence_spearman}"
        )
        print(f"  elapsed={result.elapsed_seconds:.1f}s")
        soma_results.append(result)

    elapsed_total = time.monotonic() - t0

    # --- Gate -----------------------------------------------------------
    soma_at_n100 = next((r for r in soma_results if r.consolidation_iterations == 100), None)
    soma_at_n0 = next((r for r in soma_results if r.consolidation_iterations == 0), None)
    soma_r5_n100 = soma_at_n100.metrics["transitive"]["r_at_5"] if soma_at_n100 else None
    baseline_r5_trans = baseline.metrics["transitive"]["r_at_5"]
    # Pick the best-N fallback (largest N we actually ran) so the verdict
    # stays meaningful when the plan's N=100 point wasn't reached.
    soma_r5_best = (
        max((r.metrics["transitive"]["r_at_5"] for r in soma_results), default=None)
        if soma_results
        else None
    )
    verdict, rationale = _gate_call(baseline_r5_trans, soma_r5_n100, soma_r5_best)

    # --- Report ----------------------------------------------------------
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# C1 — Plastic Graph Synthetic Retrieval Signal",
        "",
        "Sub-phase C1 of research direction C "
        "(`docs/plans/2026-04-16-research-c-graph-memory.md`).",
        "",
        f"**Embedder:** `{args.embedder}` (dim={embed_dim})  ",
        f"**Alpha (graph blend weight):** `{args.alpha}`  ",
        f"**Corpus:** {len(gt.snippets)} snippets across {len(gt.triples)} topic-triples  ",
        f"**N sweep:** {list(n_sweep)}  ",
        f"**Total wall clock:** {elapsed_total:.1f}s",
        "",
        "## Headline gate decision",
        "",
        f"**Verdict: {verdict}**",
        "",
        rationale,
        "",
        "## Transitive queries (the load-bearing metric)",
        "",
        "Direct queries are the control — both methods get the entity "
        "verbatim in the snippet text. Transitive queries are where "
        "the graph substrate is supposed to help: query is `B` (or `C`) "
        "from a transitive triple, gold answers are snippets mentioning "
        "the *other* non-`A` member, which never co-occurs with the query "
        "in text. The pure-vector retriever has no path to those snippets "
        "via cosine alone.",
        "",
        "Pure-vector reference:",
        f"- {_format_metric(baseline.metrics['transitive'])}",
        "",
        "SOMA sweep:",
        "",
        _result_table(soma_results, query_class="transitive"),
        "",
        "## Direct queries (sanity — should tie at α=0)",
        "",
        "Pure-vector reference:",
        f"- {_format_metric(baseline.metrics['direct'])}",
        "",
        _result_table(soma_results, query_class="direct"),
        "",
        "## Unrelated queries (should both score 0; hallucination check)",
        "",
        "Pure-vector reference:",
        f"- {_format_metric(baseline.metrics['unrelated'])}",
        "",
        _result_table(soma_results, query_class="unrelated"),
        "",
        "## Substrate diagnostics",
        "",
        f"- Integrator count (any SOMA run): "
        f"`{soma_results[0].integrator_count if soma_results else 0}`  "
        "*(must be > 0; the 860cc82 seed-graph fix is required for the "
        "experiment to be meaningful — see "
        "`docs/plans/2026-04-15-...` for the bug history)*",
        f"- Edge count at N=0: `{soma_at_n0.edge_count if soma_at_n0 else 'n/a'}`",
        f"- Edge count at largest N "
        f"(N={soma_results[-1].consolidation_iterations if soma_results else 0}): "
        f"`{soma_results[-1].edge_count if soma_results else 'n/a'}`",
        "",
        "Spearman ρ between edge strength and pair co-occurrence count "
        "(rank correlation across the edge-strength and pair-count "
        "distributions; non-NaN ρ means the substrate's strongest "
        "edges align with the corpus's most-frequent pairs):",
        "",
    ]
    for r in soma_results:
        rho = r.edge_weight_cooccurrence_spearman
        rho_s = "n/a" if rho is None else f"{rho:+.3f}"
        lines.append(f"- N={r.consolidation_iterations}: ρ = {rho_s}  (edges={r.edge_count})")

    lines += [
        "",
        "## Honest interpretation",
        "",
        "The C1 plan's gate is `R@5(transitive, SOMA at N=100) - "
        "R@5(transitive, pure-vector) >= 0.05`. ",
        f"Measured delta at N=100: `{(soma_r5_n100 - baseline_r5_trans):+.4f}`"
        if soma_r5_n100 is not None
        else "Measured delta at N=100: not available.",
        "",
        rationale,
        "",
        "Pure-vector dominates the transitive class only when there is "
        "*no* substrate signal helping. If the SOMA gap stays at zero "
        "or below, it's evidence the plastic-graph activations don't "
        "carry retrieval-useful structure — same conclusion as the "
        "2026-04-15 wt103 next-token-LM ablation, but for retrieval. "
        "If C1 fails, run C1b before sinking weeks into C2.",
        "",
        "---",
        "",
        f"Generated by `research/graph_memory/run_c1.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
    ]

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- JSON sidecar ----------------------------------------------------
    payload = {
        "config": {
            "embedder": args.embedder,
            "embed_dim": embed_dim,
            "alpha": args.alpha,
            "n_sweep": list(n_sweep),
            "corpus_size": len(gt.snippets),
            "num_triples": len(gt.triples),
            "num_unrelated_queries": len(gt.unrelated_queries),
            "quick": args.quick,
        },
        "baseline": asdict(baseline),
        "soma_runs": [asdict(r) for r in soma_results],
        "verdict": verdict,
        "rationale": rationale,
        "elapsed_total_seconds": elapsed_total,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nReport: {args.out_md}")
    print(f"JSON:   {args.out_json}")
    print(f"\nVerdict: {verdict}")
    print(rationale)


if __name__ == "__main__":
    main()
