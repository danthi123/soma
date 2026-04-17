"""Tests for soma.schemas.builtin.creative — creative domain schemas."""

from __future__ import annotations

import pytest

from soma.schemas import get_schema, list_schemas
from soma.schemas.builtin.creative import (
    Character,
    Continuity,
    PlotThread,
    WorldDetail,
)

# ── Registration ────────────────────────────────────────────────────


class TestRegistration:
    def test_character_registered(self) -> None:
        assert get_schema("creative.character") is Character

    def test_world_detail_registered(self) -> None:
        assert get_schema("creative.world_detail") is WorldDetail

    def test_continuity_registered(self) -> None:
        assert get_schema("creative.continuity") is Continuity

    def test_plot_thread_registered(self) -> None:
        assert get_schema("creative.plot_thread") is PlotThread

    def test_all_creative_schemas_in_list(self) -> None:
        names = list_schemas()
        for name in [
            "creative.character",
            "creative.world_detail",
            "creative.continuity",
            "creative.plot_thread",
        ]:
            assert name in names


# ── Round-trip: to_metadata / from_metadata ─────────────────────────


class TestRoundTrip:
    def test_character_round_trip(self) -> None:
        c = Character(
            name="Elara",
            project="Starfall",
            traits="brave,curious,stubborn",
            relationships="sister of Kael, rival of Mira",
            arc="reluctant hero to leader",
            last_appearance="chapter 12",
        )
        meta = c.to_metadata()
        assert meta["type"] == "creative.character"
        assert Character.from_metadata(meta) == c

    def test_character_minimal(self) -> None:
        c = Character(name="NPC Guard")
        assert Character.from_metadata(c.to_metadata()) == c

    def test_world_detail_round_trip(self) -> None:
        wd = WorldDetail(
            project="Starfall",
            category="magic_system",
            detail="magic draws from starlight; stronger at night",
            canonical=True,
            introduced_in="chapter 1",
        )
        assert WorldDetail.from_metadata(wd.to_metadata()) == wd

    def test_continuity_round_trip(self) -> None:
        ct = Continuity(
            project="Starfall",
            assertion="The capital has three gates",
            source="chapter 4",
            contradicted_by="chapter 9 mentions four gates",
        )
        assert Continuity.from_metadata(ct.to_metadata()) == ct

    def test_continuity_none_fields_omitted(self) -> None:
        ct = Continuity(project="Starfall", assertion="Dragons are extinct")
        meta = ct.to_metadata()
        assert "contradicted_by" not in meta

    def test_plot_thread_round_trip(self) -> None:
        pt = PlotThread(
            project="Starfall",
            thread="the lost heir revelation",
            status="developing",
            introduced_in="chapter 3",
        )
        assert PlotThread.from_metadata(pt.to_metadata()) == pt

    def test_plot_thread_resolved(self) -> None:
        pt = PlotThread(
            project="Starfall",
            thread="who stole the gem",
            status="resolved",
            introduced_in="chapter 1",
            resolved_in="chapter 8",
        )
        assert PlotThread.from_metadata(pt.to_metadata()) == pt


# ── Validation: choices ─────────────────────────────────────────────


class TestChoicesValidation:
    def test_world_detail_bad_category(self) -> None:
        with pytest.raises(ValueError, match="category"):
            WorldDetail(project="X", category="weather", detail="it rains")

    def test_world_detail_good_categories(self) -> None:
        for cat in [
            "geography",
            "politics",
            "magic_system",
            "technology",
            "culture",
            "history",
        ]:
            WorldDetail(project="X", category=cat, detail="d")

    def test_plot_thread_bad_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            PlotThread(project="X", thread="t", status="paused")

    def test_plot_thread_good_statuses(self) -> None:
        for s in ["planted", "developing", "resolved", "abandoned"]:
            PlotThread(project="X", thread="t", status=s)


# ── Searchable text extraction ──────────────────────────────────────


class TestSearchableText:
    def test_character_search_text(self) -> None:
        c = Character(
            name="Elara",
            traits="brave",
            relationships="sister of Kael",
            arc="hero arc",
        )
        assert c._search_text() == "Elara brave sister of Kael hero arc"

    def test_world_detail_search_text(self) -> None:
        wd = WorldDetail(
            project="X", category="geography", detail="mountain range in the north"
        )
        assert wd._search_text() == "mountain range in the north"

    def test_continuity_search_text(self) -> None:
        ct = Continuity(project="X", assertion="The king is old")
        assert ct._search_text() == "The king is old"

    def test_plot_thread_search_text(self) -> None:
        pt = PlotThread(project="X", thread="the prophecy")
        assert pt._search_text() == "the prophecy"


# ── Filterable fields ───────────────────────────────────────────────


class TestFilterableFields:
    def test_character_filterable(self) -> None:
        expected = {"name", "project"}
        assert Character._filterable_fields() == expected

    def test_world_detail_filterable(self) -> None:
        expected = {"project", "category", "canonical"}
        assert WorldDetail._filterable_fields() == expected

    def test_continuity_filterable(self) -> None:
        expected = {"project", "source", "contradicted_by"}
        assert Continuity._filterable_fields() == expected

    def test_plot_thread_filterable(self) -> None:
        expected = {"project", "status"}
        assert PlotThread._filterable_fields() == expected
