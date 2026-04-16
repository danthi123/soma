"""Unit tests for :mod:`soma.memory.backends.lancedb_filter`.

Pins the translation from Chroma-style ``where`` dicts to LanceDB's
SQL-like predicate string. These tests are the canonical reference
for which operators are supported; adding a new operator must come
with a new test here.
"""

from __future__ import annotations

import pytest

from soma.memory.backend import FilterPushdownUnsupported
from soma.memory.backends.lancedb_filter import to_lancedb_where


def test_exact_match_shorthand_becomes_eq() -> None:
    assert to_lancedb_where({"tag": "fiction"}) == "tag = 'fiction'"


def test_eq_is_equivalent_to_shorthand() -> None:
    assert to_lancedb_where({"tag": {"$eq": "fiction"}}) == "tag = 'fiction'"


def test_ne_becomes_bang_eq() -> None:
    assert to_lancedb_where({"tag": {"$ne": "spam"}}) == "tag != 'spam'"


def test_in_becomes_sql_in() -> None:
    assert to_lancedb_where({"tag": {"$in": ["a", "b", "c"]}}) == "tag IN ('a', 'b', 'c')"


def test_nin_becomes_sql_not_in() -> None:
    assert to_lancedb_where({"tag": {"$nin": ["x", "y"]}}) == "tag NOT IN ('x', 'y')"


def test_range_operators_gt_gte_lt_lte() -> None:
    for op, sql_op, value in [
        ("$gt", ">", 5),
        ("$gte", ">=", 6),
        ("$lt", "<", 10),
        ("$lte", "<=", 11),
    ]:
        out = to_lancedb_where({"year": {op: value}})
        assert out == f"year {sql_op} {value}"


def test_multiple_fields_joined_with_and() -> None:
    out = to_lancedb_where({"tag": "a", "year": {"$gte": 2020}})
    assert out == "tag = 'a' AND year >= 2020"


def test_unsupported_op_raises_filter_pushdown_unsupported() -> None:
    with pytest.raises(FilterPushdownUnsupported) as exc_info:
        to_lancedb_where({"tag": {"$regex": "^a"}})
    assert exc_info.value.op == "$regex"
    assert exc_info.value.field == "tag"


def test_in_with_non_list_raises_pushdown_unsupported() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_lancedb_where({"tag": {"$in": "oops"}})


def test_nin_with_non_list_raises_pushdown_unsupported() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_lancedb_where({"tag": {"$nin": 42}})


def test_empty_where_produces_empty_string() -> None:
    assert to_lancedb_where({}) == ""


def test_boolean_literal_renders_as_lowercase() -> None:
    assert to_lancedb_where({"active": True}) == "active = true"
    assert to_lancedb_where({"active": {"$eq": False}}) == "active = false"


def test_none_literal_renders_as_null() -> None:
    assert to_lancedb_where({"field": {"$eq": None}}) == "field = NULL"


def test_string_with_apostrophe_is_escaped() -> None:
    # Single quote must be doubled per ANSI SQL so it round-trips
    # intact without injection risk.
    assert to_lancedb_where({"name": "O'Brien"}) == "name = 'O''Brien'"


def test_float_literal_renders_numeric() -> None:
    out = to_lancedb_where({"score": {"$gte": 0.5}})
    assert out == "score >= 0.5"


def test_unsupported_literal_type_raises() -> None:
    # Lists are not valid $eq values; LanceDB can't take one here.
    with pytest.raises(FilterPushdownUnsupported):
        to_lancedb_where({"field": {"$eq": [1, 2, 3]}})


def test_empty_in_list_is_unsatisfiable() -> None:
    # LanceDB rejects `IN ()` — we emit `false` so the search returns
    # zero rows without a SQL parse error.
    assert to_lancedb_where({"tag": {"$in": []}}) == "false"


def test_empty_nin_list_is_trivially_true() -> None:
    assert to_lancedb_where({"tag": {"$nin": []}}) == "true"
