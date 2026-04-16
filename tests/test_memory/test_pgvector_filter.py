"""Unit tests for :mod:`soma.memory.backends.pgvector_filter`.

Pins the translation from SOMA's flat ``where`` dict into a pair of
``(WHERE-clause fragment, params list)`` suitable for ``psycopg``
parameter binding. pgvector itself has no opinion on filters — they're
just Postgres predicates over a JSONB ``metadata`` column — so the
translator bridges SOMA's Mongo-ish dialect to Postgres's JSONB
containment (``@>``) + typed extraction (``(metadata->>'field')::T``)
patterns.

Every op emits ``%s`` placeholders so psycopg owns the escaping. SQL
injection is prevented by construction: literal data never goes into
the clause string, only into the ``params`` list.
"""

from __future__ import annotations

import json

import pytest

from soma.memory.backend import FilterPushdownUnsupported
from soma.memory.backends.pgvector_filter import to_pgvector_where


def test_none_returns_none() -> None:
    assert to_pgvector_where(None) is None


def test_empty_dict_returns_none() -> None:
    assert to_pgvector_where({}) is None


def test_eq_uses_jsonb_containment() -> None:
    clause, params = to_pgvector_where({"user_id": "alice"})
    assert clause == "metadata @> %s::jsonb"
    assert params == [json.dumps({"user_id": "alice"})]


def test_explicit_eq_matches_bare_shorthand() -> None:
    c1, p1 = to_pgvector_where({"tag": "fiction"})
    c2, p2 = to_pgvector_where({"tag": {"$eq": "fiction"}})
    assert c1 == c2
    assert p1 == p2


def test_ne_casts_and_compares() -> None:
    clause, params = to_pgvector_where({"tag": {"$ne": "fiction"}})
    assert "metadata->>'tag'" in clause
    assert "!=" in clause or "<>" in clause
    assert params == ["fiction"]


def test_gte_casts_and_compares() -> None:
    clause, params = to_pgvector_where({"score": {"$gte": 0.5}})
    assert "(metadata->>'score')::float" in clause
    assert ">=" in clause
    assert params == [0.5]


def test_gt_lt_lte_all_numeric_cast() -> None:
    for op, sql_op in [("$gt", ">"), ("$lt", "<"), ("$lte", "<=")]:
        clause, params = to_pgvector_where({"year": {op: 2020}})
        assert "(metadata->>'year')::float" in clause
        assert f" {sql_op} " in clause
        assert params == [2020]


def test_in_uses_any_or_in() -> None:
    clause, params = to_pgvector_where({"tag": {"$in": ["a", "b"]}})
    # Implementation may pick ANY(%s) or IN (%s, %s); both are fine.
    assert "ANY" in clause or "IN " in clause
    # params: either [["a", "b"]] (single array) or ["a", "b"] (expanded).
    flat: list[str] = []
    for p in params:
        if isinstance(p, list):
            flat.extend(p)
        else:
            flat.append(p)
    assert set(flat) == {"a", "b"}


def test_nin_uses_not_any_or_not_in() -> None:
    clause, params = to_pgvector_where({"tag": {"$nin": ["x", "y"]}})
    assert "NOT" in clause and ("ANY" in clause or "IN " in clause)
    flat: list[str] = []
    for p in params:
        if isinstance(p, list):
            flat.extend(p)
        else:
            flat.append(p)
    assert set(flat) == {"x", "y"}


def test_in_empty_list_raises() -> None:
    # Semantically "no matches"; rather than emit `IN ()` (invalid SQL)
    # or a constant `false`, we raise so MemoryLayer's fallback path
    # can correctly interpret the empty set.
    with pytest.raises(FilterPushdownUnsupported):
        to_pgvector_where({"tag": {"$in": []}})


def test_nin_empty_list_raises() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_pgvector_where({"tag": {"$nin": []}})


def test_in_non_list_raises() -> None:
    with pytest.raises(FilterPushdownUnsupported):
        to_pgvector_where({"tag": {"$in": "oops"}})


def test_multiple_fields_joined_by_and() -> None:
    clause, params = to_pgvector_where({"user_id": "alice", "year": 2026})
    assert " AND " in clause
    assert len(params) == 2


def test_multiple_ops_on_same_field_joined_by_and() -> None:
    # `{"year": {"$gte": 2020, "$lt": 2025}}` should split into two
    # predicates combined with AND.
    clause, params = to_pgvector_where({"year": {"$gte": 2020, "$lt": 2025}})
    assert clause.count(" AND ") >= 1
    assert set(params) == {2020, 2025}


def test_unsupported_op_raises() -> None:
    with pytest.raises(FilterPushdownUnsupported) as exc_info:
        to_pgvector_where({"x": {"$regex": "^a"}})
    assert exc_info.value.op == "$regex"
    assert exc_info.value.field == "x"


def test_string_escaping_safe() -> None:
    # SQL injection safety: the literal goes into params, never into
    # the clause text. JSONB-containment shape is still valid JSON.
    clause, params = to_pgvector_where({"name": "'; DROP TABLE x; --"})
    assert "DROP" not in clause
    assert params == [json.dumps({"name": "'; DROP TABLE x; --"})]


def test_clause_uses_placeholders_not_interpolation() -> None:
    # Every emitted clause must use ``%s`` placeholders; no raw values.
    clause, _ = to_pgvector_where({"year": {"$gte": 2020}, "tag": "a"})
    assert "%s" in clause
    # And no values should be inlined as literals (heuristic: no ``2020``
    # substring in the clause text).
    assert "2020" not in clause
    assert "'a'" not in clause
