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
        event_name = entry.get("event", entry.get("event_type", "unknown"))
    lines.append(f"  - step {entry.get('step', '?')}: {event_name}")
    return "\n".join(lines)


_SBERT = None
_EMBED_CACHE: dict[str, torch.Tensor] = {}


def _get_sbert():
    """Lazy-load sentence-transformers model."""
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer

        _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT


def _embed(text: str) -> torch.Tensor:
    """Embed text with caching to avoid redundant encoding."""
    if text not in _EMBED_CACHE:
        _EMBED_CACHE[text] = _get_sbert().encode(
            text, convert_to_tensor=True,
        )
    return _EMBED_CACHE[text]


def _format_recalled_memories(
    text_store: dict[int, str] | None,
    query: str | None = None,
    top_k: int = 5,
) -> str:
    """Format recalled memories from the text store.

    If *query* is given, rank stored texts by semantic similarity
    using sentence-transformers embeddings.  Otherwise show the
    most recent entries.
    """
    if not text_store:
        return "  (no memories)"

    if query is not None:
        query_emb = _embed(query)
        scored: list[tuple[float, int, str]] = []
        for step, text in text_store.items():
            text_emb = _embed(text)
            sim = float(torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0), text_emb.unsqueeze(0),
            ).item())
            scored.append((sim, step, text))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        selected = scored[:top_k]
    else:
        # Most recent
        items = sorted(text_store.items(), key=lambda t: -t[0])
        selected = [(0.0, step, text) for step, text in items[:top_k]]

    lines: list[str] = []
    for _score, step, text in selected:
        # Truncate long entries
        display = text[:150] + "..." if len(text) > 150 else text
        lines.append(f"  - [step {step}] {display}")
    return "\n".join(lines)


def verbalize_state(
    soma: SOMA,
    *,
    query: str | None = None,
    text_store: dict[int, str] | None = None,
) -> str:
    """Assemble a structured text snapshot of SOMA's internal state.

    Parameters
    ----------
    soma:
        The SOMA system to introspect.
    query:
        Optional query string — if provided, recalled memories are
        ranked by keyword relevance to this query.
    text_store:
        Mapping of step → original text, used to populate the
        "Recalled Memories" section.  Typically comes from
        ``PredictiveSOMA.text_store``.

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

    # Include recalled memories if text_store is available
    if text_store:
        sections.extend([
            "",
            "Recalled Memories (most relevant):"
            if query
            else "Recalled Memories (most recent):",
            _format_recalled_memories(text_store, query=query),
        ])

    return "\n".join(sections)
