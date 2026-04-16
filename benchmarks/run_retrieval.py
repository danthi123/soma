"""Retrieval benchmark: SOMA vs Chroma plus two graph-ablation comparisons.

Runs the full 50-fact / 26-query synthetic dataset through:
- SOMA MemoryLayer (sentence-transformers) — apples-to-apples vs Chroma
- Chroma (default embedder) — plain RAG baseline
- SOMA MemoryLayer (TextEncoder, no graph) — same-embedder internal control
- SOMA MemoryLayer (TextEncoder, with graph consolidation) — graph ablation
- SOMA MemoryLayer (sbert, no graph) — graph-off control on quality embeddings
- SOMA MemoryLayer (sbert, with graph consolidation) — the key question:
  does graph consolidation help once embeddings are strong?

The report layers three comparisons: SOMA-vs-Chroma (apples-to-apples on
sbert, no graph on either side) for the pure store-efficiency claim,
then graph-vs-no-graph on both TextEncoder (graph-forced-to-work-with-
random-base) and sbert (graph-on-quality-base) so the graph value
proposition is isolated from embedding quality.

    python -m benchmarks.run_retrieval --out benchmarks/reports/retrieval.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.datasets.synthetic import generate_dataset
from benchmarks.harness.adapters.chroma import ChromaAdapter
from benchmarks.harness.adapters.soma import SomaAdapter
from benchmarks.harness.runner import format_results_table, run_retrieval_benchmark


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/retrieval.md"),
    )
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    facts, topic_facts, queries = generate_dataset()
    print(
        f"Dataset: {len(facts)} facts across "
        f"{len({t.topic for t in topic_facts})} topics, "
        f"{len(queries)} labeled queries.\n"
    )

    # Comparison 1: sbert apples-to-apples
    print("=== Comparison 1: sbert embeddings ===")
    sbert_systems = [
        SomaAdapter(use_sbert=True),
        ChromaAdapter(),
    ]
    sbert_results = run_retrieval_benchmark(
        systems=sbert_systems, dataset=facts, queries=queries, k=args.k,
    )

    # Comparison 2: TextEncoder with/without graph re-rank.
    # Graph re-rank is off by default (alpha=0); enable explicitly in the
    # "graph" arm so the comparison actually exercises the blend path.
    print("\n=== Comparison 2: SOMA graph ablation (TextEncoder embeddings) ===")
    te_flat = SomaAdapter(use_sbert=False, attach_soma=False)
    te_flat.name = "soma-te-flat"
    te_graph = SomaAdapter(
        use_sbert=False,
        attach_soma=True,
        graph_rerank_alpha=0.3,
        graph_rerank_stable_capture=True,
    )
    te_graph.name = "soma-te-graph"
    te_graph_results = run_retrieval_benchmark(
        systems=[te_flat, te_graph],
        dataset=facts,
        queries=queries,
        k=args.k,
        consolidate_after_store=True,
    )

    # Comparison 3: sbert with/without graph re-rank — the ablation that
    # matters for the research claim. See graph_ablation.md for the full
    # alpha sweep + stable-capture investigation.
    print("\n=== Comparison 3: SOMA graph ablation (sbert embeddings) ===")
    sb_flat = SomaAdapter(use_sbert=True, attach_soma=False)
    sb_flat.name = "soma-sbert-flat"
    sb_graph = SomaAdapter(
        use_sbert=True,
        attach_soma=True,
        graph_rerank_alpha=0.3,
        graph_rerank_stable_capture=True,
    )
    sb_graph.name = "soma-sbert-graph"
    sb_graph_results = run_retrieval_benchmark(
        systems=[sb_flat, sb_graph],
        dataset=facts,
        queries=queries,
        k=args.k,
        consolidate_after_store=True,
    )

    # Report
    lines = [
        "# Retrieval Benchmark — Synthetic Topic-Clustered Dataset",
        "",
        f"**Dataset:** {len(facts)} facts across "
        f"{len({t.topic for t in topic_facts})} topic clusters, "
        f"{len(queries)} labeled queries.",
        f"**k:** {args.k}",
        "",
        "## Comparison 1: SOMA vs Chroma (sentence-transformers)",
        "",
        "Apples-to-apples with identical embeddings — isolates "
        "indexing/storage mechanics.",
        "",
        format_results_table(sbert_results, k=args.k),
        "",
        "## Comparison 2: SOMA Graph Ablation (TextEncoder embeddings)",
        "",
        "Same TextEncoder embeddings in both rows — isolates the "
        "contribution of SOMA's graph consolidation + re-ranking on top of "
        "the project's default random-init encoder.",
        "",
        format_results_table(te_graph_results, k=args.k),
        "",
        "## Comparison 3: SOMA Graph Ablation (sbert embeddings)",
        "",
        "Same sbert embeddings in both rows — asks whether SOMA's graph "
        "consolidation adds value once the base embeddings are already "
        "strong. This is the ablation that matters for the research "
        "claim.",
        "",
        format_results_table(sb_graph_results, k=args.k),
        "",
        "## Headline",
        "",
    ]
    soma_r = next(r for r in sbert_results if r.system == "soma")
    chroma_r = next(r for r in sbert_results if r.system == "chroma")
    lines.append(
        f"- **Quality:** SOMA Recall@{args.k}={soma_r.recall_at_k:.3f} "
        f"vs Chroma {chroma_r.recall_at_k:.3f} "
        f"(Δ {soma_r.recall_at_k - chroma_r.recall_at_k:+.3f})"
    )
    lines.append(
        f"- **Retrieve latency:** SOMA {soma_r.retrieve_avg_ms:.2f}ms "
        f"vs Chroma {chroma_r.retrieve_avg_ms:.2f}ms "
        f"({chroma_r.retrieve_avg_ms / max(soma_r.retrieve_avg_ms, 1e-3):.1f}x faster)"
    )
    lines.append(
        f"- **Disk:** SOMA {soma_r.disk_bytes / 1024:.1f}KB "
        f"vs Chroma {chroma_r.disk_bytes / 1024:.1f}KB "
        f"({chroma_r.disk_bytes / max(soma_r.disk_bytes, 1):.1f}x smaller)"
    )
    te_flat_r = next(r for r in te_graph_results if r.system == "soma-te-flat")
    te_graph_r = next(r for r in te_graph_results if r.system == "soma-te-graph")
    lines.append(
        f"- **Graph ablation (TE):** flat Recall@{args.k}="
        f"{te_flat_r.recall_at_k:.3f} vs graph {te_graph_r.recall_at_k:.3f} "
        f"(Δ {te_graph_r.recall_at_k - te_flat_r.recall_at_k:+.3f})"
    )
    sb_flat_r = next(r for r in sb_graph_results if r.system == "soma-sbert-flat")
    sb_graph_r = next(r for r in sb_graph_results if r.system == "soma-sbert-graph")
    lines.append(
        f"- **Graph ablation (sbert):** flat Recall@{args.k}="
        f"{sb_flat_r.recall_at_k:.3f} vs graph {sb_graph_r.recall_at_k:.3f} "
        f"(Δ {sb_graph_r.recall_at_k - sb_flat_r.recall_at_k:+.3f})"
    )

    lines += [
        "",
        "---",
        "",
        "Generated by `benchmarks/run_retrieval.py`.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
