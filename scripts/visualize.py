"""CLI graph visualization — produce a textual + optional graphviz summary.

Usage::

    python scripts/visualize.py --checkpoint checkpoints/soma_final.pt \
        --output reports/graph.txt

The script loads a SOMA checkpoint and emits:
- node counts by type
- edge count + average strength / weight
- the top-K most-active nodes
- (optional) a graphviz ``.dot`` file for off-line rendering

No heavy dependencies — we generate the ``.dot`` by hand to keep the
install footprint small.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from soma.core.config import SOMAConfig
from soma.core.graph import Graph
from soma.core.node import Node, NodeType
from soma.system import SOMA

LOGGER = logging.getLogger("soma.visualize")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a SOMA checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a SOMA checkpoint file (torch.save format).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the text summary here (defaults to stdout).",
    )
    parser.add_argument(
        "--dot",
        type=Path,
        default=None,
        help="Optional path for a graphviz DOT file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of most-active nodes to show.",
    )
    parser.add_argument(
        "--max-dot-nodes",
        type=int,
        default=500,
        help="Truncate the DOT file to the most-active K nodes (graphviz scales poorly).",
    )
    return parser


# ----------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------
def load_soma(checkpoint_path: Path) -> SOMA:
    """Load a SOMA instance from a checkpoint, using the config it embeds."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    state = torch.load(str(checkpoint_path), weights_only=False)
    config = SOMAConfig.from_dict(state["config"])
    soma = SOMA(config)
    soma.load_state(checkpoint_path)
    return soma


# ----------------------------------------------------------------------
# Summarization
# ----------------------------------------------------------------------
def node_type_histogram(graph: Graph) -> dict[str, int]:
    """Return a histogram of node_type.value -> count."""
    counter: Counter[str] = Counter()
    for node in graph.all_nodes():
        counter[node.node_type.value] += 1
    return dict(counter)


def edge_weight_statistics(graph: Graph) -> dict[str, float]:
    """Compute mean/min/max/abs-mean weight statistics across all edges."""
    if graph.num_edges == 0:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0}
    weights = torch.tensor([float(e.weight.item()) for e in graph.all_edges()])
    return {
        "count": float(graph.num_edges),
        "mean": float(weights.mean().item()),
        "min": float(weights.min().item()),
        "max": float(weights.max().item()),
        "abs_mean": float(weights.abs().mean().item()),
    }


def most_active_nodes(graph: Graph, k: int) -> list[Node]:
    k = max(1, min(k, graph.num_nodes))
    return graph.most_active_nodes(k=k)


def format_summary(soma: SOMA, *, top_k: int) -> str:
    """Build a human-readable textual summary of the current SOMA state."""
    lines: list[str] = []
    lines.append("=== SOMA Checkpoint Summary ===")
    lines.append(f"global_step: {soma.global_step}")
    lines.append(f"num_nodes: {soma.graph.num_nodes}")
    lines.append(f"num_edges: {soma.graph.num_edges}")

    lines.append("")
    lines.append("-- Node counts by type --")
    for node_type, count in sorted(node_type_histogram(soma.graph).items()):
        lines.append(f"  {node_type:<12}: {count}")

    stats = edge_weight_statistics(soma.graph)
    lines.append("")
    lines.append("-- Edge weight statistics --")
    for name, value in stats.items():
        lines.append(f"  {name:<10}: {value:.4f}")

    lines.append("")
    lines.append(f"-- Top {top_k} most-active nodes (by activation_ema) --")
    for rank, node in enumerate(most_active_nodes(soma.graph, top_k), start=1):
        modality = _modality_for_node(soma.graph, node)
        lines.append(
            f"  {rank:>2}. id={node.id[:8]} type={node.node_type.value:<11} "
            f"ema={node.activation_ema:.4f} maturity={node.maturity:.4f}"
            f"{f' modality={modality}' if modality else ''}"
        )

    lines.append("")
    lines.append("-- Memory --")
    lines.append(f"  working_memory occupancy: {soma.working_memory.occupancy():.3f}")
    lines.append(f"  episodic_memory entries:  {soma.episodic_memory.num_valid}")
    lines.append("")
    return "\n".join(lines)


def _modality_for_node(graph: Graph, node: Node) -> str | None:
    """Return the modality a SENSOR/OUTPUT node owns, or ``None`` otherwise."""
    if node.node_type is NodeType.SENSOR:
        for modality, sensor in graph.sensor_nodes.items():
            if sensor.id == node.id:
                return modality
    elif node.node_type is NodeType.OUTPUT:
        for modality, out in graph.output_nodes.items():
            if out.id == node.id:
                return modality
    return None


# ----------------------------------------------------------------------
# DOT export
# ----------------------------------------------------------------------
_NODE_COLORS: dict[NodeType, str] = {
    NodeType.SENSOR: "skyblue",
    NodeType.ASSOCIATOR: "lightyellow",
    NodeType.INTEGRATOR: "lightgreen",
    NodeType.OUTPUT: "salmon",
}


def to_dot(soma: SOMA, *, max_nodes: int) -> str:
    """Produce a graphviz DOT description, truncating to ``max_nodes`` most-active."""
    graph = soma.graph
    keep_ids: set[str]
    if graph.num_nodes <= max_nodes:
        keep_ids = {n.id for n in graph.all_nodes()}
    else:
        # Always include boundary nodes regardless of activation.
        boundary_ids = {n.id for n in graph.all_nodes() if n.node_type.is_boundary}
        active = most_active_nodes(graph, max_nodes)
        keep_ids = boundary_ids | {n.id for n in active}

    lines: list[str] = ["digraph SOMA {", '  rankdir="LR";', '  node [style="filled"];']
    for node in graph.all_nodes():
        if node.id not in keep_ids:
            continue
        color = _NODE_COLORS.get(node.node_type, "white")
        modality = _modality_for_node(graph, node)
        label = f"{node.node_type.value}\\n{node.id[:6]}"
        if modality:
            label += f"\\n[{modality}]"
        lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}"];')

    for edge in graph.all_edges():
        if edge.source_id not in keep_ids or edge.target_id not in keep_ids:
            continue
        weight = float(edge.weight.item())
        penwidth = max(0.5, min(6.0, abs(weight) * 1.5))
        lines.append(
            f'  "{edge.source_id}" -> "{edge.target_id}" '
            f'[label="{weight:.2f}", penwidth={penwidth:.2f}];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    """Programmatic entry — returns a dict with the summary + paths written."""
    soma = load_soma(args.checkpoint)
    summary = format_summary(soma, top_k=args.top_k)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")
        LOGGER.info("summary written to %s", args.output)
    else:  # pragma: no cover - stdout path
        print(summary)

    written_dot: Path | None = None
    if args.dot is not None:
        dot_text = to_dot(soma, max_nodes=args.max_dot_nodes)
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        args.dot.write_text(dot_text, encoding="utf-8")
        written_dot = args.dot
        LOGGER.info("DOT graph written to %s", args.dot)

    return {
        "summary": summary,
        "output_path": str(args.output) if args.output is not None else None,
        "dot_path": str(written_dot) if written_dot is not None else None,
        "num_nodes": soma.graph.num_nodes,
        "num_edges": soma.graph.num_edges,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    sys.exit(main())
