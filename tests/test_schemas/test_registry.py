"""Tests for soma.schemas.registry — global registration + discovery."""

from __future__ import annotations

import pytest

from soma.schemas import field, get_schema, list_schemas, schema


def test_schema_registered_in_registry() -> None:
    @schema("test_reg.alpha")
    class Alpha:
        x: str = field()

    assert get_schema("test_reg.alpha") is Alpha
    assert "test_reg.alpha" in list_schemas()


def test_get_schema_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_schema("no.such.schema")


def test_schema_double_registration_raises() -> None:
    @schema("test_reg.once")
    class Once:
        x: int = field()

    with pytest.raises(ValueError, match="already registered"):

        @schema("test_reg.once")
        class Once2:
            x: int = field()
