"""Tests for soma.schemas.base — @schema decorator and field() descriptor."""

from __future__ import annotations

import dataclasses

import pytest

from soma.schemas import field, schema

# ── Decorated classes used across tests ──────────────────────────────


@schema("test_base.widget")
class Widget:
    name: str = field(filterable=True)
    color: str = field(default="red")


@schema("test_base.doc")
class Doc:
    title: str = field(searchable=True)
    body: str = field(searchable=True)
    tag: str = field(filterable=True)


# ── Tests ────────────────────────────────────────────────────────────


def test_schema_decorator_creates_dataclass() -> None:
    w = Widget(name="gear")
    assert w.name == "gear"
    assert w.color == "red"
    assert dataclasses.is_dataclass(w)


def test_to_metadata_includes_type() -> None:
    w = Widget(name="gear")
    meta = w.to_metadata()
    assert meta["type"] == "test_base.widget"
    assert meta["name"] == "gear"
    # color is "red" (non-None), should be present
    assert meta["color"] == "red"


def test_to_metadata_omits_none() -> None:
    @schema("test_base.nullable")
    class Nullable:
        x: str = field()
        y: str | None = field(default=None)

    n = Nullable(x="hello")
    meta = n.to_metadata()
    assert "y" not in meta


def test_from_metadata_round_trips() -> None:
    w = Widget(name="gear", color="blue")
    w2 = Widget.from_metadata(w.to_metadata())
    assert w == w2


def test_from_metadata_ignores_extra_keys() -> None:
    meta = {"type": "test_base.widget", "name": "sprocket", "color": "red", "extra_junk": 42}
    w = Widget.from_metadata(meta)
    assert w.name == "sprocket"
    assert w.color == "red"


def test_from_metadata_missing_required_raises() -> None:
    with pytest.raises((ValueError, TypeError)):
        Widget.from_metadata({"type": "test_base.widget", "color": "blue"})


def test_field_choices_validated_on_construction() -> None:
    @schema("test_base.priority")
    class PrioItem:
        level: str = field(choices=["low", "med", "high"])

    PrioItem(level="low")  # ok
    with pytest.raises(ValueError, match="level"):
        PrioItem(level="ultra")


def test_searchable_fields_extracted() -> None:
    d = Doc(title="hello", body="world", tag="test")
    assert d._search_text() == "hello world"


def test_filterable_fields_listed() -> None:
    assert Widget._filterable_fields() == {"name"}


def test_schema_on_pre_existing_dataclass() -> None:
    """@schema must work on a class already decorated with @dataclass."""

    @schema("test_base.predc")
    @dataclasses.dataclass
    class PreDC:
        x: int = field()
        y: str = field(default="y")

    obj = PreDC(x=1)
    assert obj.x == 1
    assert obj.y == "y"
    meta = obj.to_metadata()
    assert meta["type"] == "test_base.predc"
    assert PreDC.from_metadata(meta) == obj


def test_meta_inner_class_defaults() -> None:
    """Classes without Meta get sane defaults."""
    assert Widget._schema_meta.get("consolidation") is None
    assert Widget._schema_meta.get("ttl_seconds") is None
    assert Widget._schema_meta.get("context_priority") == 0.5


def test_meta_inner_class_custom() -> None:
    @schema("test_base.with_meta")
    class WithMeta:
        x: str = field()

        class Meta:
            consolidation = "supersede"
            ttl_seconds = 3600
            context_priority = 0.9

    assert WithMeta._schema_meta["consolidation"] == "supersede"
    assert WithMeta._schema_meta["ttl_seconds"] == 3600
    assert WithMeta._schema_meta["context_priority"] == 0.9


def test_field_both_filterable_and_searchable() -> None:
    @schema("test_base.dual")
    class Dual:
        name: str = field(filterable=True, searchable=True)

    d = Dual(name="hello")
    assert d._search_text() == "hello"
    assert Dual._filterable_fields() == {"name"}
