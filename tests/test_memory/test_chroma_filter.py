"""Unit tests for :mod:`soma.memory.backends.chroma_filter`.

Pins the translation from SOMA's flat ``where`` dict into Chroma's
native ``where`` dialect. Chroma already uses a MongoDB-style
operator vocabulary (``$eq``/``$ne``/``$gt``/``$gte``/``$lt``/``$lte``/
``$in``/``$nin``), so the translator is closer to a normalizer than
a compiler: bare ``{"field": value}`` becomes ``{"field": {"$eq":
value}}``, and multiple top-level fields are wrapped in ``$and``
because Chroma rejects bare multi-field dicts. Unsupported operators
raise :class:`FilterPushdownUnsupported` so MemoryLayer can fall back
to its Python pre-filter + ``search_subset`` path.
"""

from __future__ import annotations

import pytest

from soma.memory.backend import FilterPushdownUnsupported
from soma.memory.backends.chroma_filter import to_chroma_where


def test_none_passes_through() -> None:
    assert to_chroma_where(None) is None


def test_empty_dict_returns_none() -> None:
    # Chroma rejects ``where={}`` at query time; returning None lets
    # the caller skip the kwarg entirely.
    assert to_chroma_where({}) is None


def test_bare_value_becomes_eq() -> None:
    assert to_chroma_where({"user_id": "alice"}) == {"user_id": {"$eq": "alice"}}


def test_explicit_eq_round_trips() -> None:
    assert to_chroma_where({"tag": {"$eq": "fiction"}}) == {"tag": {"$eq": "fiction"}}


def test_in_passes_through() -> None:
    assert to_chroma_where({"tag": {"$in": ["a", "b"]}}) == {"tag": {"$in": ["a", "b"]}}


def test_nin_passes_through() -> None:
    assert to_chroma_where({"tag": {"$nin": ["x", "y"]}}) == {"tag": {"$nin": ["x", "y"]}}


def test_range_ops_pass_through() -> None:
    for op in ("$gt", "$gte", "$lt", "$lte", "$ne"):
        assert to_chroma_where({"year": {op: 2020}}) == {"year": {op: 2020}}


def test_multiple_fields_wrapped_in_and() -> None:
    # Chroma rejects `{"tag": "a", "year": {"$gte": 2020}}` directly
    # because it wants exactly one top-level operator. Wrap in $and.
    out = to_chroma_where({"tag": "a", "year": {"$gte": 2020}})
    assert out == {
        "$and": [
            {"tag": {"$eq": "a"}},
            {"year": {"$gte": 2020}},
        ]
    }


def test_unsupported_op_raises() -> None:
    with pytest.raises(FilterPushdownUnsupported) as exc_info:
        to_chroma_where({"x": {"$regex": "^a"}})
    assert exc_info.value.op == "$regex"
    assert exc_info.value.field == "x"


def test_in_with_non_list_raises() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_chroma_where({"tag": {"$in": "oops"}})


def test_nin_with_non_list_raises() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_chroma_where({"tag": {"$nin": 42}})


def test_empty_in_list_raises_unsupported() -> None:
    # Chroma rejects `$in: []` at query time ("non-empty list"). Raise
    # so MemoryLayer falls back to the Python pre-filter path, which
    # correctly interprets "in empty set" as "no matches".
    with pytest.raises(FilterPushdownUnsupported):
        to_chroma_where({"tag": {"$in": []}})


def test_empty_nin_list_raises_unsupported() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_chroma_where({"tag": {"$nin": []}})


def test_multiple_ops_on_same_field_wrapped_in_and() -> None:
    # `{"year": {"$gte": 2020, "$lt": 2025}}` — Chroma also only
    # accepts a single operator under a field, so combined ranges must
    # be split into $and.
    out = to_chroma_where({"year": {"$gte": 2020, "$lt": 2025}})
    assert out == {
        "$and": [
            {"year": {"$gte": 2020}},
            {"year": {"$lt": 2025}},
        ]
    }
