"""Translate SOMA's ``where`` dict into a pgvector-compatible SQL
``WHERE`` fragment plus a parameter list.

pgvector itself has no opinion on filters — it just provides the
``vector`` column type and distance operators. Filtering happens via
standard Postgres predicates, and SOMA stores per-vector metadata in a
single ``JSONB`` column (``metadata``). The translator bridges SOMA's
Mongo-ish operator vocabulary to two Postgres idioms:

- ``$eq`` on a flat field → ``metadata @> %s::jsonb`` (JSONB
  containment, GIN-indexable, O(1)-ish).
- ``$ne`` / ``$gt`` / ``$gte`` / ``$lt`` / ``$lte`` → typed extraction
  from the JSONB blob: ``(metadata->>'field')::float op %s`` for
  numerics, ``metadata->>'field' op %s`` for strings.
- ``$in`` / ``$nin`` → ``metadata->>'field' = ANY(%s)`` /
  ``... != ALL(%s)``, parameterised with a Python list (psycopg maps
  it to a PostgreSQL text array).

Every literal enters the query via a ``%s`` placeholder — never
interpolated into the clause string — so psycopg owns the escaping
story end-to-end. SQL injection is prevented by construction.

Returns ``None`` when the input is empty so the caller can skip the
``WHERE`` clause entirely (``WHERE`` with no predicates would be a
syntax error; omitting it is the right default).

Unsupported operators raise :class:`FilterPushdownUnsupported`; the
MemoryLayer catches that and falls back to its Python pre-filter +
``search_subset`` path.
"""

from __future__ import annotations

import json
from typing import Any

from soma.memory.backend import FilterPushdownUnsupported

_COMPARE_OPS: dict[str, str] = {
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$ne": "!=",
}


def _eq_clause(field: str, value: Any) -> tuple[str, Any]:
    """Render an equality predicate as JSONB containment.

    JSONB containment (``metadata @> %s::jsonb``) is the only form
    that uses the GIN index we build in :mod:`pgvector`, so
    equality-shaped predicates go through here unconditionally. The
    parameter is a JSON-encoded single-field object; psycopg passes it
    through as a string and the ``::jsonb`` cast converts it
    server-side.
    """
    payload = json.dumps({field: value})
    return "metadata @> %s::jsonb", payload


def _compare_clause(field: str, op: str, value: Any) -> tuple[str, Any]:
    """Render a comparison predicate (``$ne``, ``$gt``, ``$gte``,
    ``$lt``, ``$lte``) as typed extraction from the JSONB blob.

    Numerics (int, float, bool treated separately) get
    ``(metadata->>'field')::float`` so Postgres does a numeric compare
    rather than a lexicographic string compare. Strings stay text —
    ``metadata->>'field'`` is already ``text``.
    """
    sql_op = _COMPARE_OPS[op]
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int`` — keep this branch above
        # the numeric check so ``True`` doesn't slip through as 1.
        cast = f"(metadata->>'{field}')::boolean"
        return f"{cast} {sql_op} %s", value
    if isinstance(value, (int, float)):
        cast = f"(metadata->>'{field}')::float"
        return f"{cast} {sql_op} %s", value
    # Default: text compare.
    return f"metadata->>'{field}' {sql_op} %s", value


def _list_clause(
    field: str,
    op: str,
    expected: Any,
    *,
    negate: bool,
) -> tuple[str, list[Any]]:
    """Render ``$in`` / ``$nin`` as a Postgres ``ANY`` / ``ALL`` check.

    Using ``ANY(%s)`` with a Python list keeps the predicate a single
    placeholder regardless of list length — psycopg converts the list
    to a PostgreSQL array at bind time. Empty lists raise
    :class:`FilterPushdownUnsupported` because the semantics of
    "in-empty" are unambiguous in SOMA's Python fallback but cannot
    be expressed as a non-trivial Postgres predicate without
    introducing a dialect footgun (``ANY('{}')`` is always false;
    ``ALL('{}')`` is always true — easy to get wrong).
    """
    if not isinstance(expected, (list, tuple, set)):
        raise FilterPushdownUnsupported(
            op=op,
            field=field,
            message=f"{op} expects a list, got {type(expected).__name__}",
        )
    items = list(expected)
    if len(items) == 0:
        raise FilterPushdownUnsupported(
            op=op,
            field=field,
            message=f"empty {op} list cannot be translated to pgvector",
        )
    if negate:
        # NOT IN: every element must differ → ``<> ALL(%s)``. The
        # wrapping ``NOT`` also works, but ``!= ALL`` is the more
        # idiomatic Postgres form.
        return f"NOT (metadata->>'{field}' = ANY(%s))", [items]
    return f"metadata->>'{field}' = ANY(%s)", [items]


def _clauses_for_field(field: str, spec: Any) -> tuple[list[str], list[Any]]:
    """Translate one top-level ``field`` entry into one or more SQL
    fragments + their parameters.

    A bare value is ``$eq`` shorthand. A dict with multiple ops
    (``{"year": {"$gte": 2020, "$lt": 2025}}``) expands into multiple
    clauses that the caller joins with ``AND``.
    """
    fragments: list[str] = []
    params: list[Any] = []
    if not isinstance(spec, dict):
        frag, payload = _eq_clause(field, spec)
        fragments.append(frag)
        params.append(payload)
        return fragments, params
    for op, expected in spec.items():
        if op == "$eq":
            frag, payload = _eq_clause(field, expected)
            fragments.append(frag)
            params.append(payload)
        elif op in _COMPARE_OPS:
            frag, payload = _compare_clause(field, op, expected)
            fragments.append(frag)
            params.append(payload)
        elif op == "$in":
            frag, plist = _list_clause(field, op, expected, negate=False)
            fragments.append(frag)
            params.extend(plist)
        elif op == "$nin":
            frag, plist = _list_clause(field, op, expected, negate=True)
            fragments.append(frag)
            params.extend(plist)
        else:
            raise FilterPushdownUnsupported(op=op, field=field)
    return fragments, params


def to_pgvector_where(
    spec: dict[str, Any] | None,
) -> tuple[str, list[Any]] | None:
    """Compile ``spec`` into ``(WHERE fragment, params)``.

    Returns ``None`` for an absent or empty spec — the caller skips
    the ``WHERE`` clause entirely in that case. Otherwise the first
    element is a predicate string using ``%s`` placeholders (safe for
    psycopg parameter binding) and the second is the ordered
    parameter list.

    Examples::

        >>> to_pgvector_where({"user_id": "alice"})
        ('metadata @> %s::jsonb', ['{"user_id": "alice"}'])

        >>> to_pgvector_where({"score": {"$gte": 0.5}})
        ("(metadata->>'score')::float >= %s", [0.5])

        >>> to_pgvector_where({"tag": {"$in": ["a", "b"]}})
        ("metadata->>'tag' = ANY(%s)", [["a", "b"]])
    """
    if spec is None or len(spec) == 0:
        return None
    all_fragments: list[str] = []
    all_params: list[Any] = []
    for field_name, field_spec in spec.items():
        fragments, params = _clauses_for_field(field_name, field_spec)
        all_fragments.extend(fragments)
        all_params.extend(params)
    clause = " AND ".join(all_fragments)
    return clause, all_params


__all__ = ["to_pgvector_where"]
