"""Translate Chroma-style ``where`` dicts into LanceDB SQL-like strings.

MemoryLayer passes a filter dict like::

    {"tag": "fiction", "year": {"$gte": 2022}}

into ``VectorBackend.search(where=...)``. LanceDBBackend translates
that into ``"tag = 'fiction' AND year >= 2022"`` and hands it to
``table.search(q).where(...)`` for server-side execution. Any operator
we can't translate raises
:class:`soma.memory.backend.FilterPushdownUnsupported` so MemoryLayer
can fall back to its Python pre-filter path.

Supported operators mirror :data:`soma.memory.api._COMPARE_OPS`
(``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``) plus
``$in`` / ``$nin``. Bare ``{"field": value}`` is the Chroma shorthand
for ``$eq``. Field names pass through untransformed; values go through
:func:`_literal` which quotes strings, renders booleans/ints/floats
inline, and rejects types LanceDB's SQL dialect can't express.
"""

from __future__ import annotations

from typing import Any

from soma.memory.backend import FilterPushdownUnsupported


def _literal(value: Any, *, field: str, op: str) -> str:
    """Render a Python value as a LanceDB SQL literal.

    LanceDB accepts standard SQL scalar literals: ``'single-quoted
    string'``, ``42`` / ``3.14`` for numbers, ``true`` / ``false`` for
    booleans, ``NULL`` for None. Anything else (list, dict, bytes,
    datetime without a stable repr) is rejected with
    :class:`FilterPushdownUnsupported` so callers drop back to the
    Python pre-filter path rather than shipping a bogus predicate that
    LanceDB would reject at query time.

    Strings get their internal single-quotes doubled (SQL standard) so
    values like ``O'Brien`` stay intact without injection risk.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        # bool is a subclass of int; keep this branch above int/float.
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise FilterPushdownUnsupported(
        op=op,
        field=field,
        message=(
            f"cannot translate literal of type {type(value).__name__} "
            f"for field {field!r} operator {op!r}"
        ),
    )


def _compare_clause(
    field: str, op: str, sql_op: str, value: Any
) -> str:
    """Render ``field <sql_op> literal`` (e.g. ``year >= 2022``)."""
    return f"{field} {sql_op} {_literal(value, field=field, op=op)}"


def _list_clause(
    field: str,
    op: str,
    expected: Any,
    *,
    negate: bool,
) -> str:
    """Render ``field IN (...)`` or ``field NOT IN (...)``.

    ``$in`` / ``$nin`` expect a list/tuple/set; anything else is a
    caller error that we flag as unsupported so the caller falls
    back rather than emitting malformed SQL.
    """
    if not isinstance(expected, (list, tuple, set)):
        raise FilterPushdownUnsupported(op=op, field=field)
    rendered = [_literal(v, field=field, op=op) for v in expected]
    if not rendered:
        # Empty list: $in is unsatisfiable, $nin is trivially satisfied.
        # LanceDB rejects `IN ()` so we emit a constant predicate
        # instead.
        return "false" if not negate else "true"
    joined = ", ".join(rendered)
    keyword = "NOT IN" if negate else "IN"
    return f"{field} {keyword} ({joined})"


def to_lancedb_where(where: dict[str, Any]) -> str:
    """Compile ``where`` into a LanceDB SQL-like predicate string.

    Multiple top-level fields are joined with ``AND`` — same as the
    Chroma default. An empty ``where`` returns an empty string; the
    caller is expected to check for that and skip the ``.where(...)``
    call in that case.
    """
    if not where:
        return ""
    clauses: list[str] = []
    for field_name, spec in where.items():
        if isinstance(spec, dict):
            for op, expected in spec.items():
                if op == "$eq":
                    clauses.append(
                        _compare_clause(field_name, op, "=", expected)
                    )
                elif op == "$ne":
                    clauses.append(
                        _compare_clause(field_name, op, "!=", expected)
                    )
                elif op == "$gt":
                    clauses.append(
                        _compare_clause(field_name, op, ">", expected)
                    )
                elif op == "$gte":
                    clauses.append(
                        _compare_clause(field_name, op, ">=", expected)
                    )
                elif op == "$lt":
                    clauses.append(
                        _compare_clause(field_name, op, "<", expected)
                    )
                elif op == "$lte":
                    clauses.append(
                        _compare_clause(field_name, op, "<=", expected)
                    )
                elif op == "$in":
                    clauses.append(
                        _list_clause(
                            field_name, op, expected, negate=False
                        )
                    )
                elif op == "$nin":
                    clauses.append(
                        _list_clause(
                            field_name, op, expected, negate=True
                        )
                    )
                else:
                    raise FilterPushdownUnsupported(
                        op=op, field=field_name
                    )
        else:
            # Bare value = exact match (Chroma shorthand for $eq).
            clauses.append(
                _compare_clause(field_name, "$eq", "=", spec)
            )
    return " AND ".join(clauses)


__all__ = ["to_lancedb_where"]
