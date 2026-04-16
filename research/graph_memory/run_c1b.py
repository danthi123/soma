"""C1b orchestrator: readout-formula ablation.

Follows the C1 FAIL verdict (commit 1db7bb6). C1 established that
the SOMA substrate IS capturing co-occurrence (Spearman ρ=0.854
between edge strength and pair counts after N=10 consolidation)
but the α=0.3 cosine-blend readout doesn't consume the signal —
transitive R@5 was -0.050 vs pure-vector.

C1b asks: does any OTHER readout extract the signal? Runs 9
readout modes on the same trained substrate:

* pure_vector                      (cosine only; baseline)
* alpha_blend_{0.1, 0.3, 0.5, 0.7, 0.9}  (existing α-blend at 5 α)
* pure_graph_rank                  (α=1.0; blend is 100% graph cos)
* graph_traversal_expand           (cosine-top-K + 1-hop expand)
* centrality_prior                 (cosine × 1+log1p(activation norm))

Gate (from the plan):
  PASS       — a mode beats pure-vector transitive R@5 by ≥ +0.05
  AMBIGUOUS  — the best delta is +0.01 to +0.05 (noise-dominated)
  FAIL       — no mode clears +0.05 (direction C is dead for this
               substrate; recommend folding to B or D).

Usage::

    python -m research.graph_memory.run_c1b                 # default
    python -m research.graph_memory.run_c1b --quick         # smoke test
    python -m research.graph_memory.run_c1b --device cuda   # GPU

The default is 120 snippets, consolidation N=10 (the point where
C1 observed ρ=0.854), sbert embedder. A secondary N=100 run is
included when ``--secondary-n`` is set.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from research.graph_memory.baselines import run_baseline
from research.graph_memory.harness import (
    HarnessResult,
    _build_soma,
    _retrieve_node_ids,
    evaluate_memory_with_retriever,
    retrieve_centrality_prior,
    retrieve_graph_traversal_expand,
)
from research.graph_memory.harness import (
    _degree_per_node as harness_degree_per_node,
)
from research.graph_memory.synthetic_corpus import GroundTruth, generate

# Pulled from run_c1.py so the two orchestrators agree on the stub /
# sbert embedder shapes — intentionally inlined (not imported) so
# run_c1.py stays the stable C1 entry point.
DEFAULT_N: int = 10
DEFAULT_CORPUS: int = 120
DEFAULT_ALPHAS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


# ---------------------------------------------------------------------------
# Per-mode result payload
# ---------------------------------------------------------------------------
@dataclass
class ReadoutResult:
    """Metrics for ONE readout mode at ONE consolidation-N.

    Separate from HarnessResult because HarnessResult is keyed on
    training config (N, α); here we're keyed on *readout* and the
    underlying model is shared across 9 modes per N.
    """

    mode: str
    consolidation_iterations: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    # Per-N substrate diagnostics (same value across every mode at
    # the same N, but we store per-result for JSON self-containment).
    integrator_count: int = 0
    edge_count: int = 0
    edge_weight_cooccurrence_spearman: float | None = None


# ---------------------------------------------------------------------------
# Embedders (mirror run_c1.py so results are apples-to-apples)
# ---------------------------------------------------------------------------
def _stub_embed(text: str, *, dim: int = 128, device: str = "cpu") -> torch.Tensor:
    """Hash bag-of-words — deterministic, no download. Fallback only."""
    import hashlib

    vec = torch.zeros(dim, device=device)
    for tok in text.split():
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = float(vec.norm())
    if norm > 0:
        vec /= norm
    return vec


def _make_sbert_embedder(
    model_name: str = "all-MiniLM-L6-v2", *, device: str = "cpu"
):
    """Load sbert + return (embed_fn, embed_dim). Required for a fair
    pure-vector baseline: the stub embedder destroys semantic
    structure and makes pure-vector near-zero."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    dim = int(model.get_sentence_embedding_dimension())

    def embed(text: str) -> torch.Tensor:
        return torch.tensor(
            model.encode(text, convert_to_numpy=True, device=device),
            device=device,
        )

    return embed, dim


# ---------------------------------------------------------------------------
# Build a trained MemoryLayer once, reuse for every readout
# ---------------------------------------------------------------------------
def _build_trained_mem(
    gt: GroundTruth,
    *,
    embed_fn,
    embed_dim: int,
    consolidation_iterations: int,
    device: str | None = None,
) -> tuple[Any, list[str], Any]:
    """Build MemoryLayer + SOMA + ingest + consolidate.

    Returns ``(mem, snippet_ids, soma)``. This is the common setup
    step shared by every readout mode — the point of C1b is to
    *re-score the same trained model* through different lenses, so
    we pay the SOMA build+ingest cost once.
    """
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.memory.api import MemoryLayer

    # α starts at 0; each mode that needs the blend will poke the
    # attribute before calling retrieve.
    mem = MemoryLayer(
        embed_fn=embed_fn,
        embed_dim=embed_dim,
        graph_rerank_alpha=0.0,
        graph_rerank_stable_capture=True,
    )
    soma = _build_soma(embed_dim, device=device)
    soma_tokenizer = train_bpe_tokenizer([s.text for s in gt.snippets], vocab_size=512)
    soma_encoder = TextEncoder(soma_tokenizer, embed_dim=embed_dim, max_seq_len=128)
    if device is not None:
        soma_encoder = soma_encoder.to(device)
    mem.attach_soma(soma, soma_tokenizer, soma_encoder)

    snippet_ids = [mem.store(s.text) for s in gt.snippets]
    for _ in range(consolidation_iterations):
        mem._consolidation_cursor = 0
        mem.consolidate()
    mem.stable_capture()
    return mem, snippet_ids, soma


# ---------------------------------------------------------------------------
# Readout runners
# ---------------------------------------------------------------------------
def _score_mode(
    mode: str,
    *,
    retriever_fn,
    gt: GroundTruth,
    snippet_ids: list[str],
    k: int,
    consolidation_iterations: int,
    soma_diagnostics: dict[str, Any],
) -> ReadoutResult:
    """Run one retriever end-to-end and wrap in a ReadoutResult."""
    started = time.monotonic()
    metrics = evaluate_memory_with_retriever(retriever_fn, gt, snippet_ids, k=k)
    return ReadoutResult(
        mode=mode,
        consolidation_iterations=consolidation_iterations,
        metrics=metrics,
        elapsed_seconds=time.monotonic() - started,
        **soma_diagnostics,
    )


def _run_readout_sweep(
    gt: GroundTruth,
    *,
    embed_fn,
    embed_dim: int,
    consolidation_iterations: int,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    k: int = 5,
    device: str | None = None,
) -> tuple[list[ReadoutResult], dict[str, Any]]:
    """Run every C1b readout on one trained MemoryLayer."""
    from research.graph_memory.harness import edge_weight_vs_cooccurrence_spearman
    from soma.core.node import NodeType

    mem, snippet_ids, soma = _build_trained_mem(
        gt,
        embed_fn=embed_fn,
        embed_dim=embed_dim,
        consolidation_iterations=consolidation_iterations,
        device=device,
    )
    # Substrate diagnostics — identical across every readout at this N.
    soma_diagnostics = {
        "integrator_count": len(soma.graph.nodes_by_type(NodeType.INTEGRATOR)),
        "edge_count": soma.graph.num_edges,
        "edge_weight_cooccurrence_spearman": edge_weight_vs_cooccurrence_spearman(soma, gt),
    }
    # Pre-compute centrality cache once so the centrality mode isn't
    # O(N * corpus) doing the same work per query.
    centrality_cache = harness_degree_per_node(mem)

    results: list[ReadoutResult] = []

    # ----- pure_vector via the trained mem (cos-only, α=0) -----
    def pure_vector(q: str, kk: int) -> list[str]:
        mem._graph_rerank_alpha = 0.0
        return _retrieve_node_ids(mem, q, k=kk)

    results.append(
        _score_mode(
            "pure_vector_trained",
            retriever_fn=pure_vector,
            gt=gt,
            snippet_ids=snippet_ids,
            k=k,
            consolidation_iterations=consolidation_iterations,
            soma_diagnostics=soma_diagnostics,
        )
    )

    # ----- α-blend sweep -----
    for alpha in alphas:
        def alpha_blend(q: str, kk: int, a: float = alpha) -> list[str]:
            mem._graph_rerank_alpha = a
            return _retrieve_node_ids(mem, q, k=kk)

        results.append(
            _score_mode(
                f"alpha_blend_{alpha:.2f}",
                retriever_fn=alpha_blend,
                gt=gt,
                snippet_ids=snippet_ids,
                k=k,
                consolidation_iterations=consolidation_iterations,
                soma_diagnostics=soma_diagnostics,
            )
        )

    # ----- pure_graph_rank (α=1.0) -----
    def pure_graph_rank(q: str, kk: int) -> list[str]:
        mem._graph_rerank_alpha = 1.0
        return _retrieve_node_ids(mem, q, k=kk)

    results.append(
        _score_mode(
            "pure_graph_rank",
            retriever_fn=pure_graph_rank,
            gt=gt,
            snippet_ids=snippet_ids,
            k=k,
            consolidation_iterations=consolidation_iterations,
            soma_diagnostics=soma_diagnostics,
        )
    )

    # ----- graph_traversal_expand -----
    def expand(q: str, kk: int) -> list[str]:
        return retrieve_graph_traversal_expand(mem, q, k=kk, seed_k=5, expand_k=3)

    results.append(
        _score_mode(
            "graph_traversal_expand",
            retriever_fn=expand,
            gt=gt,
            snippet_ids=snippet_ids,
            k=k,
            consolidation_iterations=consolidation_iterations,
            soma_diagnostics=soma_diagnostics,
        )
    )

    # ----- centrality_prior -----
    def centrality(q: str, kk: int) -> list[str]:
        return retrieve_centrality_prior(
            mem, q, k=kk, centrality_cache=centrality_cache
        )

    results.append(
        _score_mode(
            "centrality_prior",
            retriever_fn=centrality,
            gt=gt,
            snippet_ids=snippet_ids,
            k=k,
            consolidation_iterations=consolidation_iterations,
            soma_diagnostics=soma_diagnostics,
        )
    )

    return results, soma_diagnostics


# ---------------------------------------------------------------------------
# Gate + report helpers
# ---------------------------------------------------------------------------
def _gate_call(
    baseline_r5: float,
    readout_r5_by_mode: dict[str, float],
) -> tuple[str, str, str, float]:
    """Apply the C1b gate from the plan.

    Returns ``(verdict, rationale, best_mode, best_delta)``.

    PASS: max delta ≥ +0.05.
    AMBIGUOUS: max delta between +0.01 and +0.05.
    FAIL: max delta < +0.01 (plan's "no readout clears +0.05 and at
          least one actively hurts" — we use max-delta as the decisive
          term; the orchestrator's rationale flags any active-harm
          modes alongside).
    """
    if not readout_r5_by_mode:
        return ("AMBIGUOUS", "No readout modes ran — nothing to compare.", "(none)", 0.0)
    best_mode, best_r5 = max(readout_r5_by_mode.items(), key=lambda kv: kv[1])
    best_delta = best_r5 - baseline_r5

    # Identify actively-harming modes (delta ≤ -0.01) for the rationale.
    hurting = [
        (m, r5 - baseline_r5)
        for m, r5 in readout_r5_by_mode.items()
        if (r5 - baseline_r5) <= -0.01
    ]
    hurting_line = ""
    if hurting:
        hurting_sorted = sorted(hurting, key=lambda x: x[1])[:3]
        hurting_str = ", ".join(f"{m} ({d:+.3f})" for m, d in hurting_sorted)
        hurting_line = f" Actively-harming modes: {hurting_str}."

    if best_delta >= 0.05:
        return (
            "PASS",
            f"Readout `{best_mode}` beats pure-vector transitive R@5 by "
            f"{best_delta:+.3f} (≥+0.05 threshold from the C1b plan). The "
            "graph substrate carries retrieval signal — the C1 blend was "
            "the broken piece. Recommend C2 with this readout plumbed into "
            f"MemoryLayer.{hurting_line}",
            best_mode,
            best_delta,
        )
    if best_delta >= 0.01:
        return (
            "AMBIGUOUS",
            f"Best readout `{best_mode}` beats pure-vector transitive R@5 "
            f"by {best_delta:+.3f} (between +0.01 and +0.05 — noisy, not "
            "conclusive). Recommend a larger corpus (300-500 snippets) "
            f"re-run before committing to C2.{hurting_line}",
            best_mode,
            best_delta,
        )
    return (
        "FAIL",
        f"No readout clears the +0.05 gate. Best was `{best_mode}` at "
        f"{best_delta:+.3f} vs pure-vector transitive R@5. Declare research "
        "direction C dead for retrieval objectives; recommend folding to "
        f"Research B (CL benchmark) or D (associative memory).{hurting_line}",
        best_mode,
        best_delta,
    )


def _transitive_table(results: list[ReadoutResult], baseline: HarnessResult) -> str:
    """Markdown table: readout mode × transitive R@1/R@5/MRR.

    First row is the pure-vector baseline (reference), then every
    C1b mode sorted by decreasing R@5 so the winner is at the top.
    """
    lines = [
        "| Mode | N | R@1 | R@5 | MRR | Δ R@5 vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    bm = baseline.metrics["transitive"]
    lines.append(
        f"| pure_vector (baseline) | - | {bm['r_at_1']:.3f} | "
        f"{bm['r_at_5']:.3f} | {bm['mrr']:.3f} | (ref) |"
    )
    # Sort so the best R@5 is on top.
    rows = sorted(
        results,
        key=lambda r: -r.metrics.get("transitive", {}).get("r_at_5", 0.0),
    )
    for r in rows:
        m = r.metrics.get("transitive", {})
        if not m:
            continue
        delta = m["r_at_5"] - bm["r_at_5"]
        lines.append(
            f"| {r.mode} | {r.consolidation_iterations} | "
            f"{m['r_at_1']:.3f} | {m['r_at_5']:.3f} | "
            f"{m['mrr']:.3f} | {delta:+.3f} |"
        )
    return "\n".join(lines)


def _class_table(results: list[ReadoutResult], baseline: HarnessResult, query_class: str) -> str:
    """Markdown table for a non-transitive query class (direct, unrelated)."""
    lines = [
        "| Mode | N | R@1 | R@5 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    bm = baseline.metrics.get(query_class, {})
    if bm:
        lines.append(
            f"| pure_vector (baseline) | - | {bm['r_at_1']:.3f} | "
            f"{bm['r_at_5']:.3f} | {bm['mrr']:.3f} |"
        )
    for r in results:
        m = r.metrics.get(query_class, {})
        if not m:
            continue
        lines.append(
            f"| {r.mode} | {r.consolidation_iterations} | "
            f"{m['r_at_1']:.3f} | {m['r_at_5']:.3f} | "
            f"{m['mrr']:.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["stub", "sbert"],
        default="sbert",
        help="Embedding backend. 'stub' = hash bag-of-words (fast, "
        "unfair baseline — NOT recommended for the headline number). "
        "'sbert' = sentence-transformers (slow first-run download).",
    )
    p.add_argument(
        "--corpus-size",
        type=int,
        default=DEFAULT_CORPUS,
        help=f"Snippets in the synthetic corpus (default {DEFAULT_CORPUS}). "
        "C1 used 60; 120 gives more transitive query instances while "
        "staying within the C1b wall-clock budget.",
    )
    p.add_argument(
        "--consolidation-n",
        type=int,
        default=DEFAULT_N,
        help=f"Consolidation iterations (default {DEFAULT_N}, where C1 "
        "observed ρ=0.854 between edge strength and co-occurrence).",
    )
    p.add_argument(
        "--secondary-n",
        type=int,
        default=0,
        help="Optional second N point (default 0 = skip). 100 is a "
        "useful check if the primary point is ambiguous.",
    )
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help=f"Alpha values for the blend sweep (default {list(DEFAULT_ALPHAS)}).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: shrink corpus to 40 snippets, consolidation "
        "N=1, skip secondary N. Used for end-to-end validation.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda / cpu). Default: cpu (GPU is 2x "
        "slower on this tiny substrate per the 2026-04-16 smoke).",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("research/graph_memory/reports/c1b_blend_ablation.md"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("research/graph_memory/reports/c1b_blend_ablation.json"),
    )
    args = p.parse_args()

    # --- Device preflight ----------------------------------------------
    device = args.device or "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={device} but torch.cuda.is_available() is False."
        )
    print(f"device={device}")

    # --- Corpus --------------------------------------------------------
    if args.quick:
        corpus_size = 40
        primary_n = 1
        secondary_n = 0
        alphas = (0.3, 0.7)  # just enough to exercise the sweep
    else:
        corpus_size = args.corpus_size
        primary_n = args.consolidation_n
        secondary_n = args.secondary_n
        alphas = tuple(args.alphas)

    gt = generate(total_snippets=corpus_size)
    print(f"corpus: {len(gt.snippets)} snippets, {len(gt.triples)} triples")
    print(f"transitive query set size: {len(gt.transitive_targets)}")
    print(f"alphas: {list(alphas)}")
    print(f"primary consolidation N: {primary_n}")
    if secondary_n > 0:
        print(f"secondary consolidation N: {secondary_n}")

    # --- Embedder ------------------------------------------------------
    if args.embedder == "sbert":
        print(f"loading sbert (all-MiniLM-L6-v2) on {device}…")
        embed_fn, embed_dim = _make_sbert_embedder(device=device)
    else:
        def embed_fn(text: str) -> torch.Tensor:
            return _stub_embed(text, device=device)

        embed_dim = 128

    # --- Pure-vector baseline (reference for the gate) -----------------
    print("\n=== pure-vector baseline ===")
    baseline = run_baseline(
        gt,
        embed_fn=embed_fn,
        embed_dim=embed_dim,
        k=5,
        method_label=f"pure-vector-{args.embedder}",
    )
    for cls, m in baseline.metrics.items():
        print(
            f"  {cls:12s}  R@1={m['r_at_1']:.3f}  R@5={m['r_at_5']:.3f}  "
            f"MRR={m['mrr']:.3f}  (n={int(m['num_queries'])})"
        )
    print(f"  elapsed={baseline.elapsed_seconds:.1f}s")

    # --- Primary N readout sweep --------------------------------------
    t_start = time.monotonic()
    print(f"\n=== readout sweep @ N={primary_n} ===")
    primary_results, primary_diag = _run_readout_sweep(
        gt,
        embed_fn=embed_fn,
        embed_dim=embed_dim,
        consolidation_iterations=primary_n,
        alphas=alphas,
        k=5,
        device=device,
    )
    for r in primary_results:
        m = r.metrics["transitive"]
        print(
            f"  {r.mode:32s}  R@1={m['r_at_1']:.3f}  R@5={m['r_at_5']:.3f}  "
            f"MRR={m['mrr']:.3f}  elapsed={r.elapsed_seconds:.1f}s"
        )
    print(
        f"  substrate: integrators={primary_diag['integrator_count']}  "
        f"edges={primary_diag['edge_count']}  "
        f"spearman={primary_diag['edge_weight_cooccurrence_spearman']}"
    )

    # --- Optional secondary N ------------------------------------------
    secondary_results: list[ReadoutResult] = []
    secondary_diag: dict[str, Any] = {}
    if secondary_n > 0:
        print(f"\n=== readout sweep @ N={secondary_n} ===")
        secondary_results, secondary_diag = _run_readout_sweep(
            gt,
            embed_fn=embed_fn,
            embed_dim=embed_dim,
            consolidation_iterations=secondary_n,
            alphas=alphas,
            k=5,
            device=device,
        )
        for r in secondary_results:
            m = r.metrics["transitive"]
            print(
                f"  {r.mode:32s}  R@1={m['r_at_1']:.3f}  "
                f"R@5={m['r_at_5']:.3f}  MRR={m['mrr']:.3f}"
            )

    elapsed_total = time.monotonic() - t_start

    # --- Gate ----------------------------------------------------------
    baseline_r5 = baseline.metrics["transitive"]["r_at_5"]
    # Gate applies to the PRIMARY N. Secondary is informational.
    primary_r5_by_mode = {
        r.mode: r.metrics["transitive"]["r_at_5"] for r in primary_results
    }
    verdict, rationale, best_mode, best_delta = _gate_call(baseline_r5, primary_r5_by_mode)

    # --- Report --------------------------------------------------------
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# C1b — Blend-Formula Ablation",
        "",
        "Sub-phase C1b of research direction C "
        "(`docs/plans/2026-04-16-research-c-graph-memory.md`). "
        "Follow-up to the C1 FAIL verdict (see "
        "`research/graph_memory/reports/c1_synthetic_signal.md`).",
        "",
        f"**Run date:** {time.strftime('%Y-%m-%d')}",
        f"**Embedder:** `{args.embedder}` (dim={embed_dim})  ",
        f"**Device:** `{device}`  ",
        f"**Corpus:** {len(gt.snippets)} snippets, "
        f"{len(gt.triples)} triples ({len(gt.transitive_targets)} "
        "transitive queries + 60 direct + 10 unrelated)  ",
        f"**Primary consolidation N:** `{primary_n}`  ",
        f"**Secondary consolidation N:** "
        f"`{secondary_n if secondary_n > 0 else 'skipped'}`  ",
        f"**α sweep:** `{list(alphas)}`  ",
        f"**Total wall clock:** {elapsed_total:.1f}s",
        "",
        "## Headline gate decision",
        "",
        f"**Verdict: {verdict}**",
        "",
        rationale,
        "",
        f"**Best mode:** `{best_mode}`  ",
        f"**Best Δ R@5 vs pure-vector:** `{best_delta:+.3f}`",
        "",
        "## Transitive queries (load-bearing)",
        "",
        "This is the class C1b is measuring. Direct queries should "
        "tie (sanity), unrelated should both score 0 (hallucination "
        "check), but transitive is where ONLY the graph substrate "
        "has a path to the correct answer — a winning readout must "
        "beat pure-vector here by ≥ +0.05.",
        "",
        _transitive_table(primary_results, baseline),
        "",
    ]
    if secondary_results:
        lines += [
            f"### Secondary N={secondary_n} (informational)",
            "",
            _transitive_table(secondary_results, baseline),
            "",
        ]
    lines += [
        "## Direct queries (sanity — should tie)",
        "",
        _class_table(primary_results, baseline, "direct"),
        "",
        "## Unrelated queries (hallucination check — should be ≈0)",
        "",
        _class_table(primary_results, baseline, "unrelated"),
        "",
        "## Substrate diagnostics",
        "",
        f"- Integrator count: `{primary_diag['integrator_count']}` "
        "*(must be > 0 — post-860cc82 seed-graph fix)*",
        f"- Edge count after N={primary_n}: `{primary_diag['edge_count']}`",
        f"- Spearman ρ(edge strength vs pair co-occurrence): "
        f"`{primary_diag['edge_weight_cooccurrence_spearman']}`",
        "",
        "For comparison, C1 saw ρ=+0.854 at N=10 on a 60-snippet "
        "corpus. If the number above is meaningfully different, the "
        "substrate's behaviour changed between experiments and that "
        "should be investigated separately.",
        "",
        "## Honest interpretation",
        "",
        "",
    ]
    # Append a honest-interpretation paragraph tied to the verdict.
    if verdict == "PASS":
        lines += [
            f"The `{best_mode}` readout extracted retrieval signal the C1 "
            f"α=0.3 blend was leaving on the table ({best_delta:+.3f} R@5 "
            "on transitive). This is the first evidence the plastic graph "
            "lifts retrieval on this substrate. Next step: plumb this "
            "readout into `MemoryLayer.retrieve` as an opt-in "
            "`retrieval_mode=` argument and run C2's LoCoMo benchmark.",
        ]
    elif verdict == "AMBIGUOUS":
        lines += [
            f"The `{best_mode}` readout moves the needle ({best_delta:+.3f} "
            "R@5) but the delta is in the noise band. The substrate-level "
            "signal (ρ≈0.85 between edge strength and co-occurrence) is "
            "real; the readout is partially harvesting it. Recommend a "
            "300-500 snippet re-run before deciding C2 vs. fold.",
        ]
    else:
        lines += [
            "Across 9 readout modes — spanning pure-graph-rank, α-blend "
            "from 0.1 to 0.9, graph-traversal expansion, and centrality "
            "prior — NONE cleared the +0.05 gate. The substrate does "
            "capture co-occurrence (ρ≈0.85) but the captured signal does "
            "not translate into retrieval lift under any of the readouts "
            "tested.",
            "",
            "Interpretation candidates (from most to least likely):",
            "",
            "1. **The edge-strength distribution captures a STATISTICAL "
            "property (pair-count rank) but not the node-pair IDENTITY**. "
            "ρ is computed over *sorted* edge-strength and pair-count "
            "distributions — it tells us the top-strength edges align "
            "with the top-frequency pairs IN AGGREGATE, but not that "
            "*specific* edges correspond to *specific* pairs. A readout "
            "that relies on \"edge (A,B) is strong because A and B "
            "co-occurred\" has no edge-to-pair lookup table to exploit. "
            "This is a structural limit of the current substrate, not "
            "a blend-formula bug.",
            "2. **The query-to-activation path (TextEncoder → SOMA "
            "graph → stored output) is too lossy to preserve "
            "pair-identity through retrieval**. The stored activations "
            "are the output of a graph-wide propagation, not a direct "
            "edge-weight readout; by the time the signal reaches the "
            "output layer, the pair-specific information is averaged "
            "across many nodes.",
            "3. **The readouts tested are the wrong family**. We have "
            "NOT tested: learned blend (C3), query-time consolidation "
            "(C4), or direct edge-list retrieval (bypass activations "
            "entirely — query the graph edges by entity tokens, if we "
            "can index SOMA nodes by the text they respond to). That "
            "last one is the most promising untested option; it would "
            "require extending the substrate to expose a text-token → "
            "graph-node index, which is a sizeable engineering effort.",
            "",
            "Recommendation: fold C to C3 with \"direct edge-list "
            "retrieval\" as the only remaining candidate readout, or "
            "declare C dead and redirect engineering to Research B or "
            "D. Which depends on strategic priority, not further "
            "evidence from C1 or C1b.",
        ]

    lines += [
        "",
        "---",
        "",
        f"Generated by `research/graph_memory/run_c1b.py` on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
    ]

    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    # --- JSON sidecar --------------------------------------------------
    from dataclasses import asdict

    payload = {
        "config": {
            "embedder": args.embedder,
            "embed_dim": embed_dim,
            "device": device,
            "corpus_size": len(gt.snippets),
            "primary_n": primary_n,
            "secondary_n": secondary_n,
            "alphas": list(alphas),
            "num_transitive_queries": len(gt.transitive_targets),
            "num_unrelated_queries": len(gt.unrelated_queries),
            "quick": args.quick,
        },
        "baseline": asdict(baseline),
        "primary": {
            "n": primary_n,
            "diagnostics": primary_diag,
            "results": [asdict(r) for r in primary_results],
        },
        "secondary": {
            "n": secondary_n,
            "diagnostics": secondary_diag,
            "results": [asdict(r) for r in secondary_results],
        }
        if secondary_results
        else None,
        "verdict": verdict,
        "rationale": rationale,
        "best_mode": best_mode,
        "best_delta_r5_vs_baseline": best_delta,
        "elapsed_total_seconds": elapsed_total,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"\nReport: {args.out_md}")
    print(f"JSON:   {args.out_json}")
    print(f"\nVerdict: {verdict}")
    print(rationale)


if __name__ == "__main__":
    main()
