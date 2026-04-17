"""Serialize SOMA's internal state to structured text for LLM consumption."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soma.core.node import NodeType

if TYPE_CHECKING:
    from soma.system import SOMA


def _developmental_stage(step: int) -> str:
    """Map global step to a human-readable developmental stage name."""
    if step < 100:
        return "blank-slate"
    if step < 500:
        return "early-plasticity"
    if step < 2000:
        return "pattern-recognition"
    if step < 5000:
        return "association-formation"
    return "mature"


def _format_active_nodes(soma: SOMA, top_k: int = 5) -> str:
    """Format the top-k most active nodes by activation EMA."""
    nodes = soma.graph.most_active_nodes(top_k)
    if not nodes:
        return "  (no nodes)"
    lines: list[str] = []
    for i, node in enumerate(nodes, 1):
        lines.append(
            f"  {i}. {node.node_type.value.upper()} node "
            f"(ema={node.activation_ema:.3f}, maturity={node.maturity:.2f})"
        )
    return "\n".join(lines)


def _format_working_memory(soma: SOMA) -> str:
    """Format working-memory occupancy and top slots by usage."""
    usage: torch.Tensor = soma.working_memory.usage
    age: torch.Tensor = soma.working_memory.age
    num_slots = soma.working_memory.num_slots

    occupied = int((usage > 0).sum().item())
    lines: list[str] = [f"  {occupied}/{num_slots} slots occupied"]

    # Top 3 occupied slots by usage.
    if occupied > 0:
        top_k = min(3, occupied)
        values, indices = torch.topk(usage, top_k)
        for v, idx in zip(values.tolist(), indices.tolist(), strict=True):
            if v > 0:
                lines.append(f"  - slot {idx}: usage={v:.2f}, age={int(age[idx].item())} steps")

    return "\n".join(lines)


def _format_graph_stats(soma: SOMA) -> str:
    """Format graph size and per-type node breakdown."""
    all_nodes = soma.graph.all_nodes()
    num_nodes = len(all_nodes)
    num_edges = len(soma.graph.edges)

    counts: dict[str, int] = {}
    for nt in NodeType:
        c = len(soma.graph.nodes_by_type(nt))
        if c > 0:
            counts[nt.value.upper()] = c

    breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
    return f"  {num_nodes} nodes ({breakdown}), {num_edges} edges"


def _format_growth_log(soma: SOMA, recent: int = 5) -> str:
    """Format the most recent growth events."""
    entries = list(soma.growth_log)[-recent:]
    if not entries:
        return "  (none)"
    lines: list[str] = []
    for entry in entries:
        lines.append(f"  - step {entry['step']}: {entry['event']}")
    return "\n".join(lines)


def verbalize_state(soma: SOMA) -> str:
    """Assemble a structured text snapshot of SOMA's internal state.

    Returns a multi-line string suitable for injection into an LLM prompt.
    """
    step = soma.global_step
    stage = _developmental_stage(step)
    novelty = soma.last_curiosity
    lr_mult = soma.homeostasis.global_lr_multiplier
    loss_ema = soma.homeostasis.loss_ema

    valid_tensor: torch.Tensor = soma.episodic_memory.valid
    ep_stored = int(valid_tensor.sum().item())
    ep_capacity = soma.episodic_memory.capacity

    sections: list[str] = [
        "[SOMA Internal State]",
        f"Developmental Stage: {stage} (step {step})",
        f"Novelty Score: {novelty:.3f}",
        f"Stability: lr_mult={lr_mult:.3f}, loss_ema={loss_ema:.4f}",
        "",
        "Graph Structure:",
        _format_graph_stats(soma),
        "",
        "Active Nodes (top 5):",
        _format_active_nodes(soma),
        "",
        "Working Memory:",
        _format_working_memory(soma),
        "",
        "Recent Growth Events:",
        _format_growth_log(soma),
        "",
        f"Episodic Memory: {ep_stored}/{ep_capacity} experiences stored",
    ]

    return "\n".join(sections)
