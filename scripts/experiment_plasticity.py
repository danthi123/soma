"""Experiment: does SOMA's graph restructure meaningfully after consolidation?

Stores facts via MemoryLayer, simulates skewed usage (some facts
retrieved far more often), consolidates through a SOMA graph, then
measures whether the graph structure reflects the usage patterns:
edge count, edge strength distribution, node activation variance.

    python scripts/experiment_plasticity.py --out reports/plasticity-experiment.md

This does NOT require sentence-transformers or a GPU — it uses SOMA's
built-in TextEncoder with random embeddings. The experiment measures
graph-level structural changes, not retrieval quality (which would need
a real embedder). Retrieval quality improvement is the Stage 5 follow-up
once graph-aware re-ranking lands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory import MemoryLayer
from soma.system import SOMA

FACTS = [
    "Alex lives in Portland, Oregon",
    "Alex is 32 years old",
    "Alex is vegetarian",
    "Alex is allergic to shellfish",
    "Alex's dog is named Luna, a 3-year-old border collie",
    "Alex works as a senior engineer at ArcMotion, a robotics startup",
    "Alex graduated from MIT with a CS degree in 2016",
    "Alex speaks fluent French and English",
    "Alex prefers dark mode in all applications",
    "Alex's favorite programming language is Rust",
    "Alex enjoys trail running on weekends",
    "Alex is currently reading Godel Escher Bach",
    "Alex's partner is named Jordan",
    "Jordan works as a veterinarian",
    "Alex and Jordan adopted Luna from a rescue shelter",
    "Alex's favorite restaurant is Pok Pok in Portland",
    "Alex drives a 2022 Rivian R1T",
    "Alex's birthday is March 15",
    "Alex has a standing desk at home",
    "Alex uses Neovim as a primary editor",
    "The project deadline at ArcMotion is June 15",
    "Alex is team lead on the perception module",
    "ArcMotion's main product is an autonomous warehouse robot",
    "Alex commutes by bicycle",
    "Alex has been at ArcMotion for 3 years",
]

HIGH_USAGE_QUERIES = [
    "Where does Alex live?",
    "What does Alex do for work?",
    "Tell me about Alex's dog",
    "What are Alex's dietary restrictions?",
    "When is the project deadline?",
]

LOW_USAGE_QUERIES = [
    "What car does Alex drive?",
    "What editor does Alex use?",
    "What is Alex reading?",
]

HIGH_USAGE_REPEATS = 20
LOW_USAGE_REPEATS = 1


def _graph_stats(soma: SOMA) -> dict[str, float]:
    """Snapshot graph-level statistics."""
    graph = soma.graph
    edges = list(graph.edges.values())
    nodes = list(graph.nodes.values())

    strengths = [float(e.strength) for e in edges]
    step = soma.global_step
    ages = [max(0, step - e.creation_step) for e in edges]

    activations = []
    for n in nodes:
        if n.last_activation is not None:
            activations.append(float(n.last_activation.abs().mean().item()))

    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "mean_edge_strength": sum(strengths) / max(len(strengths), 1),
        "max_edge_strength": max(strengths) if strengths else 0.0,
        "min_edge_strength": min(strengths) if strengths else 0.0,
        "std_edge_strength": _std(strengths),
        "mean_edge_age": sum(ages) / max(len(ages), 1),
        "mean_node_activation": sum(activations) / max(len(activations), 1),
        "std_node_activation": _std(activations),
        "wm_occupancy": float(soma.working_memory.occupancy()),
        "episodic_count": int(soma.episodic_memory.num_valid),
    }


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _format_stats(stats: dict[str, float]) -> str:
    lines = []
    for key, val in stats.items():
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.4f} |")
        else:
            lines.append(f"| {key} | {val} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/plasticity-experiment.md"),
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # Build embedder + SOMA
    all_texts = FACTS + HIGH_USAGE_QUERIES + LOW_USAGE_QUERIES
    tokenizer = train_bpe_tokenizer(all_texts, vocab_size=512)
    embed_dim = 32
    encoder = TextEncoder(tokenizer, embed_dim=embed_dim, max_seq_len=128)
    config = SOMAConfig(
        vocab_size=512,
        text_embed_dim=embed_dim,
        sensor_output_dim=embed_dim,
        max_input_tokens=128,
        synaptogenesis_interval=5,
        neurogenesis_interval=50,
        pruning_interval=25,
        consolidation_interval=50,
    )
    soma = SOMA(config)

    # Phase 1: store facts
    print("Phase 1: Storing facts...")
    mem = MemoryLayer(tokenizer=tokenizer, encoder=encoder)
    for fact in FACTS:
        mem.store(fact)
    print(f"  Stored {len(mem)} facts.\n")

    # Snapshot: pre-usage graph
    pre_stats = _graph_stats(soma)
    print("Pre-usage graph stats:")
    for k, v in pre_stats.items():
        print(f"  {k}: {v}")
    print()

    # Phase 2: simulate usage (retrieve-heavy for some queries)
    print("Phase 2: Simulating skewed usage...")
    for query in HIGH_USAGE_QUERIES:
        for _ in range(HIGH_USAGE_REPEATS):
            mem.retrieve(query, k=3)
    for query in LOW_USAGE_QUERIES:
        for _ in range(LOW_USAGE_REPEATS):
            mem.retrieve(query, k=3)
    total_retrieves = (
        len(HIGH_USAGE_QUERIES) * HIGH_USAGE_REPEATS
        + len(LOW_USAGE_QUERIES) * LOW_USAGE_REPEATS
    )
    print(f"  {total_retrieves} retrieves simulated.\n")

    # Phase 3: consolidate through SOMA
    print("Phase 3: Consolidating through SOMA graph...")
    mem.attach_soma(soma, tokenizer, encoder)
    processed = mem.consolidate()
    print(f"  {processed} entries pushed through SOMA.\n")

    # Snapshot: post-consolidation graph
    post_stats = _graph_stats(soma)
    print("Post-consolidation graph stats:")
    for k, v in post_stats.items():
        print(f"  {k}: {v}")
    print()

    # Phase 4: second consolidation pass (reinforcement)
    print("Phase 4: Second consolidation pass (reinforcement)...")
    processed2 = mem.consolidate()
    print(f"  {processed2} entries pushed through SOMA.\n")

    post2_stats = _graph_stats(soma)
    print("Post-second-consolidation graph stats:")
    for k, v in post2_stats.items():
        print(f"  {k}: {v}")
    print()

    # Write report
    report_lines = [
        "# Plasticity Experiment — Graph Structure After Consolidation",
        "",
        f"**Dataset:** {len(FACTS)} personal-profile facts.",
        f"**Usage simulation:** {len(HIGH_USAGE_QUERIES)} high-usage queries "
        f"x{HIGH_USAGE_REPEATS} + {len(LOW_USAGE_QUERIES)} low-usage queries "
        f"x{LOW_USAGE_REPEATS} = {total_retrieves} retrieves.",
        f"**Consolidation:** 2 passes, {processed + processed2} total entries "
        "through SOMA graph.",
        "",
        "## Graph Statistics",
        "",
        "| Metric | Pre-usage | Post-1st consolidation | Post-2nd consolidation |",
        "| --- | --- | --- | --- |",
    ]
    all_keys = list(pre_stats.keys())
    for key in all_keys:
        pre = pre_stats[key]
        post = post_stats[key]
        post2 = post2_stats[key]
        fmt = ".4f" if isinstance(pre, float) else ""
        report_lines.append(
            f"| {key} | {pre:{fmt}} | {post:{fmt}} | {post2:{fmt}} |"
        )

    # Compute deltas
    report_lines += [
        "",
        "## Key Deltas",
        "",
        f"- Edge count: {pre_stats['num_edges']} -> "
        f"{post_stats['num_edges']} -> {post2_stats['num_edges']}",
        f"- Node count: {pre_stats['num_nodes']} -> "
        f"{post_stats['num_nodes']} -> {post2_stats['num_nodes']}",
        f"- Mean edge strength: {pre_stats['mean_edge_strength']:.4f} -> "
        f"{post_stats['mean_edge_strength']:.4f} -> "
        f"{post2_stats['mean_edge_strength']:.4f}",
        f"- Edge strength std: {pre_stats['std_edge_strength']:.4f} -> "
        f"{post_stats['std_edge_strength']:.4f} -> "
        f"{post2_stats['std_edge_strength']:.4f}",
        f"- Episodic memory: {pre_stats['episodic_count']} -> "
        f"{post_stats['episodic_count']} -> {post2_stats['episodic_count']}",
        f"- WM occupancy: {pre_stats['wm_occupancy']} -> "
        f"{post_stats['wm_occupancy']} -> {post2_stats['wm_occupancy']}",
        "",
        "## Interpretation",
        "",
        "If consolidation is working, we expect:",
        "- **Edge count to change** (synaptogenesis adds, pruning removes).",
        "- **Edge strength distribution to widen** (frequently-coactivated "
        "edges strengthen; others weaken).",
        "- **Episodic memory to fill** (each SOMA.step encodes an episode).",
        "- **Node activations to differentiate** (higher std = more "
        "specialized nodes).",
        "",
        "If all numbers stay flat, the consolidation path is not reaching "
        "the growth engine (bug).",
        "",
        "---",
        "",
        "Generated by `scripts/experiment_plasticity.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
