"""Context packer — stub for Phase 42; real mixing ships in Phase 43.

This module provides a minimal ``pack_context()`` that concatenates
retrieved results into a single string. Phase 43 adds priority-aware
mixing, token budgeting, and deduplication.
"""

from __future__ import annotations

from typing import Any

__all__ = ["pack_context"]


def pack_context(
    results: list[Any],
    *,
    max_tokens: int | None = None,
    separator: str = "\n---\n",
) -> str:
    """Concatenate retrieved schema instances (or raw strings) into context.

    Parameters
    ----------
    results:
        A list of schema instances (with ``_search_text()``) or plain
        strings. Schema instances are converted via ``_search_text()``.
    max_tokens:
        Ignored in this stub. Phase 43 adds token-aware truncation.
    separator:
        String inserted between entries.

    Returns
    -------
    str
        Concatenated context string.
    """
    parts: list[str] = []
    for item in results:
        if isinstance(item, str):
            parts.append(item)
        elif hasattr(item, "_search_text"):
            parts.append(item._search_text())
        else:
            parts.append(str(item))
    return separator.join(parts)
