"""Unit tests for :mod:`soma.memory.backends.qdrant_filter`.

Pins the translation from Chroma-style ``where`` dicts to Qdrant's
``Filter`` model. These tests are the canonical reference for which
operators are supported; adding a new operator must come with a new
test here.
"""

from __future__ import annotations

import pytest

qm = pytest.importorskip("qdrant_client.http.models")

from soma.memory.backend import FilterPushdownUnsupported  # noqa: E402
from soma.memory.backends.qdrant_filter import to_qdrant_filter  # noqa: E402


def _conds(f, attr: str):
    """Return the list of conditions on ``f.must`` or ``f.must_not``."""
    return list(getattr(f, attr) or [])


def test_exact_match_shorthand_becomes_match_value() -> None:
    f = to_qdrant_filter({"tag": "fiction"})
    must = _conds(f, "must")
    assert len(must) == 1
    assert must[0].key == "tag"
    assert isinstance(must[0].match, qm.MatchValue)
    assert must[0].match.value == "fiction"


def test_eq_is_equivalent_to_shorthand() -> None:
    f = to_qdrant_filter({"tag": {"$eq": "fiction"}})
    must = _conds(f, "must")
    assert must[0].match.value == "fiction"


def test_ne_goes_to_must_not() -> None:
    f = to_qdrant_filter({"tag": {"$ne": "spam"}})
    must_not = _conds(f, "must_not")
    assert len(must_not) == 1
    assert must_not[0].key == "tag"
    assert must_not[0].match.value == "spam"


def test_in_translates_to_match_any() -> None:
    f = to_qdrant_filter({"tag": {"$in": ["a", "b", "c"]}})
    must = _conds(f, "must")
    assert isinstance(must[0].match, qm.MatchAny)
    assert list(must[0].match.any) == ["a", "b", "c"]


def test_nin_translates_to_must_not_match_any() -> None:
    f = to_qdrant_filter({"tag": {"$nin": ["x", "y"]}})
    must_not = _conds(f, "must_not")
    assert isinstance(must_not[0].match, qm.MatchAny)
    assert list(must_not[0].match.any) == ["x", "y"]


def test_range_operators_gt_gte_lt_lte() -> None:
    for op, attr, value in [
        ("$gt", "gt", 5),
        ("$gte", "gte", 6),
        ("$lt", "lt", 10),
        ("$lte", "lte", 11),
    ]:
        f = to_qdrant_filter({"year": {op: value}})
        must = _conds(f, "must")
        assert must[0].key == "year"
        assert getattr(must[0].range, attr) == value


def test_multiple_fields_populate_must_with_AND_semantics() -> None:
    f = to_qdrant_filter({"tag": "a", "year": {"$gte": 2020}})
    must = _conds(f, "must")
    keys = {c.key for c in must}
    assert keys == {"tag", "year"}


def test_unsupported_op_raises_filter_pushdown_unsupported() -> None:
    with pytest.raises(FilterPushdownUnsupported) as exc_info:
        to_qdrant_filter({"tag": {"$regex": "^a"}})
    assert exc_info.value.op == "$regex"
    assert exc_info.value.field == "tag"


def test_in_with_non_list_raises_pushdown_unsupported() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_qdrant_filter({"tag": {"$in": "oops"}})


def test_nin_with_non_list_raises_pushdown_unsupported() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_qdrant_filter({"tag": {"$nin": 42}})


def test_empty_where_produces_empty_filter() -> None:
    f = to_qdrant_filter({})
    # Both clauses should be None so Qdrant sees a no-op filter.
    assert getattr(f, "must", None) in (None, [])
    assert getattr(f, "must_not", None) in (None, [])
