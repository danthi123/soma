"""Context packer -- assemble a prompt-ready context string from memory.

Replaces the Phase 42 stub with priority-aware mixing, per-slot
token budgeting, type filtering, and deduplication.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from soma.memory.api import MemoryLayer

__all__ = ["pack_context"]

_DEFAULT_MIX: dict[str, float] = {
    "recency": 0.15,
    "relevant": 0.50,
    "task_state": 0.10,
    "decisions": 0.10,
    "preferences": 0.15,
}


def _format_entry(type_name: str, text: str) -> str:
    """Format a single entry as ``[type] text``."""
    return f"[{type_name}] {text}"


def _type_matches(type_name: str, patterns: list[str]) -> bool:
    """Check if *type_name* matches any of the glob *patterns*."""
    return any(fnmatch.fnmatch(type_name, p) for p in patterns)


def _truncate_lines(lines: list[str], budget: int) -> list[str]:
    """Return as many full lines as fit within *budget* characters."""
    result: list[str] = []
    used = 0
    for line in lines:
        # Account for the newline separator between lines.
        cost = len(line) + (1 if result else 0)
        if used + cost > budget:
            break
        result.append(line)
        used += cost
    return result


def pack_context(
    mem: MemoryLayer,
    query: str,
    *,
    max_tokens: int = 3800,
    mix: dict[str, float] | None = None,
    types: list[str] | None = None,
    chars_per_token: float = 4.0,
) -> str:
    """Assemble a prompt-ready context string from memory.

    Parameters
    ----------
    mem:
        A :class:`MemoryLayer` instance to pull entries from.
    query:
        The current user query, used for semantic retrieval.
    max_tokens:
        Approximate token budget for the returned string.
    mix:
        Weight distribution across slots. Keys are slot names
        (``recency``, ``relevant``, ``task_state``, ``decisions``,
        ``preferences``). Values are floats that will be normalised
        to sum to 1.0.  ``None`` uses the built-in defaults.
    types:
        Restrict retrieval to schema type prefixes matching these
        glob patterns (e.g. ``["agent.*", "conv.*"]``).
        ``None`` means no restriction.
    chars_per_token:
        Approximate characters per token for budget calculation.

    Returns
    -------
    str
        A single string ready to prepend to the LLM prompt, with
        entries formatted as ``[type_name] text`` lines.
    """
    effective_mix = dict(_DEFAULT_MIX) if mix is None else dict(mix)

    # Normalise weights to sum to 1.0.
    total_weight = sum(effective_mix.values())
    if total_weight <= 0:
        return ""
    for k in effective_mix:
        effective_mix[k] /= total_weight

    max_chars = int(max_tokens * chars_per_token)
    seen_ids: set[str] = set()
    sections: list[str] = []

    for slot_name, weight in effective_mix.items():
        budget = int(weight * max_chars)
        if budget <= 0:
            continue
        lines = _fill_slot(mem, slot_name, query, budget, types, seen_ids)
        sections.extend(lines)

    return "\n".join(sections)


def _fill_slot(
    mem: Any,
    slot_name: str,
    query: str,
    budget: int,
    types: list[str] | None,
    seen_ids: set[str],
) -> list[str]:
    """Fill a single slot, returning formatted lines within *budget*."""
    raw_lines: list[str] = []

    if slot_name == "recency":
        hits = mem.get_recent(20)
        for hit in hits:
            if hit.node_id in seen_ids:
                continue
            type_name = hit.metadata.get("type", "unknown")
            if types and not _type_matches(type_name, types):
                continue
            seen_ids.add(hit.node_id)
            raw_lines.append(_format_entry(type_name, hit.text))

    elif slot_name == "relevant":
        hits = mem.retrieve(query, k=20)
        for hit in hits:
            if hit.node_id in seen_ids:
                continue
            type_name = hit.metadata.get("type", "unknown")
            if types and not _type_matches(type_name, types):
                continue
            seen_ids.add(hit.node_id)
            raw_lines.append(_format_entry(type_name, hit.text))

    elif slot_name == "task_state":
        raw_lines = _fill_typed_slot(
            mem, "agent.task_state", query, types, seen_ids, status="active"
        )

    elif slot_name == "decisions":
        raw_lines = _fill_typed_slot(
            mem, "agent.decision", query, types, seen_ids
        )

    elif slot_name == "preferences":
        raw_lines = _fill_typed_slot(
            mem, "conv.preference", query, types, seen_ids
        )

    else:
        # Unknown slot -- try as a schema type name for extensibility.
        raw_lines = _fill_typed_slot(
            mem, slot_name, query, types, seen_ids
        )

    return _truncate_lines(raw_lines, budget)


def _fill_typed_slot(
    mem: Any,
    type_name: str,
    query: str,
    types: list[str] | None,
    seen_ids: set[str],
    **filter_kwargs: Any,
) -> list[str]:
    """Attempt to fill a slot via ``retrieve_typed`` for *type_name*.

    Falls back gracefully if the schema is not registered or there
    are no matching entries.
    """
    if types and not _type_matches(type_name, types):
        return []

    from soma.schemas.registry import get_schema

    try:
        schema_cls = get_schema(type_name)
    except KeyError:
        return []

    try:
        instances = mem.retrieve_typed(schema_cls, query, k=10, **filter_kwargs)
    except Exception:
        return []

    lines: list[str] = []
    for inst in instances:
        # Deduplicate: use the metadata round-trip to get an id if possible.
        text = inst._search_text() if hasattr(inst, "_search_text") else str(inst)
        if not text:
            continue
        lines.append(_format_entry(type_name, text))
    return lines
