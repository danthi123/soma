"""Tests for soma.memory.conversational_prompts — the three prompt templates.

These are string templates with ``.format(...)`` interpolation. The tests
pin the anchors small models need (empty-list example, all four op
names, "do not invent") so a refactor can't silently degrade extraction
or reconcile quality.
"""

from __future__ import annotations

from soma.memory.conversational_prompts import (
    EXTRACT_PROMPT,
    RECONCILE_PROMPT,
    SUMMARY_PROMPT,
)


def test_extract_prompt_contains_empty_list_example() -> None:
    """Small models need an anchor showing it's OK to return ``[]``.

    Without an empty-list example, 3B-range models hallucinate a fact
    for greetings like "Hey" or "thanks". The example is the single
    biggest quality lever for small-model extraction.
    """
    assert "[]" in EXTRACT_PROMPT


def test_extract_prompt_is_model_agnostic_length() -> None:
    """Under 1500 chars so tiny context windows don't truncate it."""
    assert len(EXTRACT_PROMPT) < 1500, (
        f"EXTRACT_PROMPT is {len(EXTRACT_PROMPT)} chars; keep it under 1500"
    )


def test_extract_prompt_has_message_placeholder() -> None:
    """The caller uses ``.format(message=...)`` — placeholder must exist."""
    formatted = EXTRACT_PROMPT.format(message="hello world")
    assert "hello world" in formatted
    assert "{message}" not in formatted


def test_reconcile_prompt_lists_all_four_ops() -> None:
    """ADD / UPDATE / SUPERSEDE / NOOP must all be named explicitly."""
    for op in ("ADD", "UPDATE", "SUPERSEDE", "NOOP"):
        assert op in RECONCILE_PROMPT, f"reconcile prompt missing op name {op!r}"


def test_reconcile_prompt_has_new_fact_and_candidates_placeholders() -> None:
    formatted = RECONCILE_PROMPT.format(
        new_fact="user lives in Boston",
        candidates="[1] user lives in Portland",
    )
    assert "user lives in Boston" in formatted
    assert "user lives in Portland" in formatted
    assert "{new_fact}" not in formatted
    assert "{candidates}" not in formatted


def test_summary_prompt_preserves_entities_clause() -> None:
    """Explicit 'do not invent details' clause (prevents summary drift)."""
    assert "do not invent" in SUMMARY_PROMPT.lower(), (
        "SUMMARY_PROMPT must carry an explicit 'do not invent' clause"
    )


def test_summary_prompt_has_turns_placeholder() -> None:
    formatted = SUMMARY_PROMPT.format(turns="user: hi\nassistant: hello")
    assert "user: hi" in formatted
    assert "{turns}" not in formatted
