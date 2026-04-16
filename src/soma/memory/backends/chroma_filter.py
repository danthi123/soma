"""Translate SOMA's ``where`` dict into Chroma's native ``where`` dialect.

Chroma already uses a MongoDB-style operator vocabulary
(``$eq``/``$ne``/``$gt``/``$gte``/``$lt``/``$lte``/``$in``/``$nin``),
so the translation is closer to a normaliser than a compiler:

- Bare ``{"field": value}`` becomes ``{"field": {"$eq": value}}``.
- Multiple top-level fields are wrapped in ``$and`` because Chroma's
  query validator rejects bare multi-field dicts ("Expected where to
  have exactly one operator").
- Multiple operators on the same field (``{"year": {"$gte": 2020,
  "$lt": 2025}}``) are likewise split into ``$and`` clauses.
- Anything outside SOMA's supported op set raises
  :class:`FilterPushdownUnsupported` so MemoryLayer falls back to its
  Python pre-filter + ``search_subset`` path.

Empty ``$in`` / ``$nin`` lists are NOT passed through (Chroma rejects
them with "non-empty list"); we raise
:class:`FilterPushdownUnsupported` so the Python fallback can
correctly interpret "in empty set" = no matches.

The translator returns ``None`` for an empty or absent filter so the
caller can skip the ``where=`` kwarg on ``Collection.query`` — Chroma
also rejects ``where={}`` at query time.
"""

from __future__ import annotations

from typing import Any

from soma.memory.backend import FilterPushdownUnsupported

# SOMA's supported op vocabulary. Matches ``_COMPARE_OPS`` plus
# ``$in`` / ``$nin`` (see ``soma.memory.api._matches_where``). Every
# op in this set maps 1:1 onto Chroma's native dialect.
_PASSTHROUGH_OPS: frozenset[str] = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
)


def _clause_for_field(field: str, spec: Any) -> list[dict[str, Any]]:
    """Render one top-level ``field`` entry as a list of Chroma
    single-op clauses.

    Chroma only accepts one operator per field-dict, so
    ``{"year": {"$gte": 2020, "$lt": 2025}}`` splits into two clauses
    that the caller joins with ``$and``.
    """
    if not isinstance(spec, dict):
        # Bare value shorthand → explicit $eq (Chroma accepts both,
        # but normalising removes ambiguity for downstream callers).
        return [{field: {"$eq": spec}}]
    clauses: list[dict[str, Any]] = []
    for op, expected in spec.items():
        if op not in _PASSTHROUGH_OPS:
            raise FilterPushdownUnsupported(op=op, field=field)
        if op in ("$in", "$nin"):
            if not isinstance(expected, (list, tuple, set)):
                raise FilterPushdownUnsupported(
                    op=op,
                    field=field,
                    message=(f"{op} expects a list, got {type(expected).__name__}"),
                )
            if len(expected) == 0:
                # Chroma rejects `$in: []` / `$nin: []` with
                # "non-empty list". We defer to the Python pre-filter
                # path instead of trying to emit a constant predicate
                # because Chroma has no SQL-literal equivalent.
                raise FilterPushdownUnsupported(
                    op=op,
                    field=field,
                    message=f"Chroma rejects empty {op} lists",
                )
            clauses.append({field: {op: list(expected)}})
        else:
            clauses.append({field: {op: expected}})
    return clauses


def to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compile ``where`` into a Chroma-native ``where`` dict.

    Returns ``None`` for an absent or empty filter so the caller can
    skip passing ``where=`` to :meth:`chromadb.Collection.query`.
    Chroma would raise on ``where={}`` otherwise.
    """
    if where is None or len(where) == 0:
        return None
    all_clauses: list[dict[str, Any]] = []
    for field_name, spec in where.items():
        all_clauses.extend(_clause_for_field(field_name, spec))
    if len(all_clauses) == 1:
        return all_clauses[0]
    return {"$and": all_clauses}


__all__ = ["to_chroma_where"]
