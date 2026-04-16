"""Translate Chroma-style ``where`` dicts into Qdrant ``Filter`` models.

MemoryLayer passes a filter dict like::

    {"tag": "fiction", "year": {"$gte": 2022}}

into ``VectorBackend.search(where=...)``. QdrantBackend translates
that into ``qm.Filter(must=[...])`` and lets Qdrant execute it
server-side. Any operator we can't translate raises
:class:`soma.memory.backend.FilterPushdownUnsupported` so the caller
can fall back to the Python pre-filter path.

Supported operators mirror
:data:`soma.memory.api._COMPARE_OPS` (``$eq``, ``$ne``, ``$gt``,
``$gte``, ``$lt``, ``$lte``) plus ``$in`` / ``$nin``. Bare
``{"field": value}`` is the Chroma shorthand for ``$eq``.
"""

from __future__ import annotations

from typing import Any

from soma.memory.backend import FilterPushdownUnsupported


def to_qdrant_filter(where: dict[str, Any]) -> Any:
    """Compile ``where`` into a ``qm.Filter`` ready to pass as
    ``query_filter=...`` to Qdrant's ``search`` / ``query_points``.

    Multiple top-level fields become ``must`` clauses (AND) — same as
    Chroma's default.
    """
    from qdrant_client.http import models as qm

    must: list[Any] = []
    must_not: list[Any] = []
    for field_name, spec in where.items():
        if isinstance(spec, dict):
            for op, expected in spec.items():
                if op == "$eq":
                    must.append(
                        qm.FieldCondition(
                            key=field_name,
                            match=qm.MatchValue(value=expected),
                        )
                    )
                elif op == "$ne":
                    must_not.append(
                        qm.FieldCondition(
                            key=field_name,
                            match=qm.MatchValue(value=expected),
                        )
                    )
                elif op == "$in":
                    if not isinstance(expected, (list, tuple, set)):
                        raise FilterPushdownUnsupported(
                            op=op, field=field_name
                        )
                    must.append(
                        qm.FieldCondition(
                            key=field_name,
                            match=qm.MatchAny(any=list(expected)),
                        )
                    )
                elif op == "$nin":
                    if not isinstance(expected, (list, tuple, set)):
                        raise FilterPushdownUnsupported(
                            op=op, field=field_name
                        )
                    must_not.append(
                        qm.FieldCondition(
                            key=field_name,
                            match=qm.MatchAny(any=list(expected)),
                        )
                    )
                elif op in ("$gt", "$gte", "$lt", "$lte"):
                    kwargs: dict[str, Any] = {}
                    if op == "$gt":
                        kwargs["gt"] = expected
                    elif op == "$gte":
                        kwargs["gte"] = expected
                    elif op == "$lt":
                        kwargs["lt"] = expected
                    else:
                        kwargs["lte"] = expected
                    must.append(
                        qm.FieldCondition(
                            key=field_name, range=qm.Range(**kwargs)
                        )
                    )
                else:
                    raise FilterPushdownUnsupported(op=op, field=field_name)
        else:
            # Bare value = exact match (Chroma shorthand for $eq).
            must.append(
                qm.FieldCondition(
                    key=field_name,
                    match=qm.MatchValue(value=spec),
                )
            )
    return qm.Filter(must=must or None, must_not=must_not or None)


__all__ = ["to_qdrant_filter"]
