"""Tests for soma.schemas.validation — instance + filter validation."""

from __future__ import annotations

import pytest

from soma.schemas import field, schema
from soma.schemas.validation import validate_filter_kwargs, validate_instance


@schema("test_val.item")
class Item:
    name: str = field(filterable=True, searchable=True)
    priority: str = field(filterable=True, choices=["p0", "p1", "p2", "p3"])
    description: str = field(searchable=True)
    status: str = field(filterable=True, default="open")


# ── validate_instance ────────────────────────────────────────────────


def test_validate_instance_ok() -> None:
    item = Item(name="bug", priority="p1", description="broken")
    validate_instance(item)  # should not raise


def test_validate_instance_bad_choice_raises() -> None:
    # Bypass __post_init__ by using object.__setattr__ after construction
    item = Item(name="bug", priority="p1", description="broken")
    object.__setattr__(item, "priority", "p5")
    with pytest.raises(ValueError, match="priority"):
        validate_instance(item)


def test_validate_instance_not_a_schema_raises() -> None:
    with pytest.raises(TypeError, match="not a SOMA schema"):
        validate_instance("not a schema instance")


# ── validate_filter_kwargs ───────────────────────────────────────────


def test_validate_filter_kwargs_ok() -> None:
    result = validate_filter_kwargs(Item, priority="p1", status="open")
    assert "priority" in result
    assert "status" in result


def test_validate_filter_kwargs_non_filterable_raises() -> None:
    with pytest.raises(ValueError, match="not filterable"):
        validate_filter_kwargs(Item, description="oops")


def test_validate_filter_kwargs_bad_choice_raises() -> None:
    with pytest.raises(ValueError, match="priority"):
        validate_filter_kwargs(Item, priority="p5")


def test_validate_filter_kwargs_converts_to_filter_spec() -> None:
    result = validate_filter_kwargs(Item, priority="p1")
    # Should be {"priority": {"$eq": "p1"}}
    assert result == {"priority": {"$eq": "p1"}}


def test_validate_filter_kwargs_unknown_field_raises() -> None:
    with pytest.raises(ValueError, match="not filterable"):
        validate_filter_kwargs(Item, nonexistent="x")
