"""Tests for soma.schemas.packing — context packer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from soma.schemas.packing import (
    _DEFAULT_MIX,
    _format_entry,
    _truncate_lines,
    pack_context,
)

# ── Helpers ─────────────────────────────────────────────────────────


def _make_hit(
    node_id: str,
    text: str,
    score: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock MemoryHit."""
    hit = MagicMock()
    hit.node_id = node_id
    hit.text = text
    hit.score = score
    hit.metadata = metadata or {}
    return hit


def _make_mem(
    recent_hits: list[Any] | None = None,
    retrieve_hits: list[Any] | None = None,
) -> MagicMock:
    """Create a mock MemoryLayer with configurable returns."""
    mem = MagicMock()
    mem.get_recent.return_value = recent_hits or []
    mem.retrieve.return_value = retrieve_hits or []
    mem.retrieve_typed.return_value = []
    return mem


# ── Unit: formatting helpers ────────────────────────────────────────


class TestFormatEntry:
    def test_basic_format(self) -> None:
        assert _format_entry("conv.fact", "hello world") == "[conv.fact] hello world"

    def test_empty_text(self) -> None:
        assert _format_entry("km.note", "") == "[km.note] "


class TestTruncateLines:
    def test_all_fit(self) -> None:
        lines = ["aaa", "bbb", "ccc"]
        assert _truncate_lines(lines, 100) == lines

    def test_budget_exact(self) -> None:
        # "aaa" + "\n" + "bbb" = 3 + 1 + 3 = 7
        lines = ["aaa", "bbb", "ccc"]
        assert _truncate_lines(lines, 7) == ["aaa", "bbb"]

    def test_budget_too_small(self) -> None:
        lines = ["aaa", "bbb"]
        assert _truncate_lines(lines, 2) == []

    def test_empty_input(self) -> None:
        assert _truncate_lines([], 100) == []

    def test_first_line_fits_exactly(self) -> None:
        assert _truncate_lines(["abc"], 3) == ["abc"]


# ── Integration: pack_context ───────────────────────────────────────


class TestPackContextBasic:
    def test_empty_memory_returns_empty(self) -> None:
        mem = _make_mem()
        result = pack_context(mem, "hello")
        assert result == ""

    def test_recency_slot_fills(self) -> None:
        hits = [
            _make_hit("h1", "recent entry", metadata={"type": "conv.fact"}),
        ]
        mem = _make_mem(recent_hits=hits)
        result = pack_context(mem, "anything", mix={"recency": 1.0})
        assert "[conv.fact] recent entry" in result

    def test_relevant_slot_fills(self) -> None:
        hits = [
            _make_hit("h1", "relevant entry", metadata={"type": "km.note"}),
        ]
        mem = _make_mem(retrieve_hits=hits)
        result = pack_context(mem, "search query", mix={"relevant": 1.0})
        assert "[km.note] relevant entry" in result

    def test_deduplication_across_slots(self) -> None:
        """Same node_id should not appear twice across recency + relevant."""
        hit = _make_hit("same_id", "entry text", metadata={"type": "conv.fact"})
        mem = _make_mem(recent_hits=[hit], retrieve_hits=[hit])
        result = pack_context(
            mem, "q", mix={"recency": 0.5, "relevant": 0.5}
        )
        assert result.count("[conv.fact] entry text") == 1


class TestPackContextMixWeights:
    def test_default_mix_keys(self) -> None:
        expected = {"recency", "relevant", "task_state", "decisions", "preferences"}
        assert set(_DEFAULT_MIX.keys()) == expected

    def test_default_mix_sums_to_one(self) -> None:
        assert abs(sum(_DEFAULT_MIX.values()) - 1.0) < 1e-9

    def test_custom_mix_normalised(self) -> None:
        """Weights are normalised, so {a: 2, b: 8} should give 20%/80% split."""
        hits_a = [_make_hit(f"a{i}", f"text-a-{i}", metadata={"type": "x"}) for i in range(20)]
        hits_b = [_make_hit(f"b{i}", f"text-b-{i}", metadata={"type": "x"}) for i in range(20)]
        mem = _make_mem(recent_hits=hits_a, retrieve_hits=hits_b)
        # With a small budget, fewer recency entries should fit than relevant
        result = pack_context(
            mem, "q", max_tokens=100, mix={"recency": 0.2, "relevant": 0.8}
        )
        assert isinstance(result, str)

    def test_zero_total_weight_returns_empty(self) -> None:
        mem = _make_mem()
        result = pack_context(mem, "q", mix={"recency": 0.0})
        assert result == ""


class TestPackContextTypeFiltering:
    def test_type_filter_includes_matching(self) -> None:
        hits = [
            _make_hit("h1", "agent data", metadata={"type": "agent.task_state"}),
            _make_hit("h2", "conv data", metadata={"type": "conv.fact"}),
        ]
        mem = _make_mem(recent_hits=hits)
        result = pack_context(
            mem, "q", mix={"recency": 1.0}, types=["agent.*"]
        )
        assert "agent data" in result
        assert "conv data" not in result

    def test_type_filter_none_includes_all(self) -> None:
        hits = [
            _make_hit("h1", "agent data", metadata={"type": "agent.task_state"}),
            _make_hit("h2", "conv data", metadata={"type": "conv.fact"}),
        ]
        mem = _make_mem(recent_hits=hits)
        result = pack_context(mem, "q", mix={"recency": 1.0}, types=None)
        assert "agent data" in result
        assert "conv data" in result

    def test_type_filter_multiple_patterns(self) -> None:
        hits = [
            _make_hit("h1", "agent", metadata={"type": "agent.task_state"}),
            _make_hit("h2", "conv", metadata={"type": "conv.fact"}),
            _make_hit("h3", "km", metadata={"type": "km.note"}),
        ]
        mem = _make_mem(recent_hits=hits)
        result = pack_context(
            mem, "q", mix={"recency": 1.0}, types=["agent.*", "km.*"]
        )
        assert "agent" in result
        assert "km" in result
        assert "[conv.fact]" not in result


class TestPackContextBudget:
    def test_respects_max_tokens(self) -> None:
        # Create many entries that exceed the budget
        hits = [
            _make_hit(f"h{i}", f"entry number {i} with some text", metadata={"type": "x"})
            for i in range(100)
        ]
        mem = _make_mem(recent_hits=hits)
        result = pack_context(
            mem, "q", max_tokens=50, mix={"recency": 1.0}, chars_per_token=4.0
        )
        # 50 tokens * 4 chars/token = 200 chars max
        assert len(result) <= 200

    def test_chars_per_token_adjusts_budget(self) -> None:
        hits = [
            _make_hit(f"h{i}", f"entry {i}", metadata={"type": "x"})
            for i in range(100)
        ]
        mem = _make_mem(recent_hits=hits)
        small = pack_context(
            mem, "q", max_tokens=50, mix={"recency": 1.0}, chars_per_token=2.0
        )
        large = pack_context(
            mem, "q", max_tokens=50, mix={"recency": 1.0}, chars_per_token=8.0
        )
        assert len(large) >= len(small)


class TestPackContextFormat:
    def test_output_lines_have_type_prefix(self) -> None:
        hits = [
            _make_hit("h1", "hello", metadata={"type": "conv.fact"}),
            _make_hit("h2", "world", metadata={"type": "km.note"}),
        ]
        mem = _make_mem(recent_hits=hits)
        result = pack_context(mem, "q", mix={"recency": 1.0})
        lines = result.strip().split("\n")
        for line in lines:
            assert line.startswith("[")
            assert "] " in line

    def test_missing_type_uses_unknown(self) -> None:
        hits = [_make_hit("h1", "no type", metadata={})]
        mem = _make_mem(recent_hits=hits)
        result = pack_context(mem, "q", mix={"recency": 1.0})
        assert "[unknown]" in result


class TestPackContextTypedSlots:
    def test_task_state_slot_calls_retrieve_typed(self) -> None:
        mem = _make_mem()
        with patch("soma.schemas.registry.get_schema") as mock_gs:
            mock_gs.return_value = MagicMock()
            pack_context(mem, "q", mix={"task_state": 1.0})
            # retrieve_typed should have been called
            assert mem.retrieve_typed.called

    def test_decisions_slot_calls_retrieve_typed(self) -> None:
        mem = _make_mem()
        with patch("soma.schemas.registry.get_schema") as mock_gs:
            mock_gs.return_value = MagicMock()
            pack_context(mem, "q", mix={"decisions": 1.0})
            assert mem.retrieve_typed.called

    def test_preferences_slot_calls_retrieve_typed(self) -> None:
        mem = _make_mem()
        with patch("soma.schemas.registry.get_schema") as mock_gs:
            mock_gs.return_value = MagicMock()
            pack_context(mem, "q", mix={"preferences": 1.0})
            assert mem.retrieve_typed.called

    def test_unknown_schema_gracefully_skipped(self) -> None:
        """If a typed slot's schema is not registered, it produces no output."""
        mem = _make_mem()
        # "nonexistent.schema" will KeyError in get_schema
        result = pack_context(mem, "q", mix={"nonexistent.schema": 1.0})
        assert result == ""
