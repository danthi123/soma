"""Validation for schema instances and filter kwargs."""

from __future__ import annotations

import dataclasses
from typing import Any

from soma.schemas.base import _get_field_spec

__all__ = ["validate_instance", "validate_filter_kwargs"]


def validate_instance(instance: Any) -> None:
    """Validate a schema instance.

    Checks: required fields present, choice fields in range.
    Raises ``ValueError`` with a clear message on failure, or
    ``TypeError`` if *instance* is not a SOMA schema instance.
    """
    if not dataclasses.is_dataclass(instance) or not hasattr(instance, "_type_name"):
        raise TypeError(
            f"{type(instance).__qualname__} is not a SOMA schema instance"
        )
    for f in dataclasses.fields(instance):
        val = getattr(instance, f.name)
        spec = _get_field_spec(f)
        if spec is None:
            continue
        # Required field check: MISSING default + None value.
        no_default = f.default is dataclasses.MISSING
        no_factory = f.default_factory is dataclasses.MISSING  # type: ignore[arg-type]
        if val is None and no_default and no_factory:
            raise ValueError(f"Required field {f.name!r} is None")
        # Choices check.
        if spec.choices is not None and val is not None and val not in spec.choices:
            raise ValueError(
                f"{f.name}={val!r} not in allowed choices {spec.choices}"
            )


def validate_filter_kwargs(schema_cls: type, **kwargs: Any) -> dict[str, Any]:
    """Validate filter kwargs against a schema.

    Returns a Chroma-compatible filter dict suitable for
    ``MemoryLayer.retrieve(where=...)``.

    Raises ``ValueError`` if a kwarg doesn't correspond to a filterable
    field, or if the value violates the field's choices constraint.
    """
    filterable = schema_cls._filterable_fields()  # type: ignore[attr-defined]
    choices_map: dict[str, list[str]] = getattr(schema_cls, "_choices_map", {})
    result: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key not in filterable:
            raise ValueError(
                f"{key!r} is not filterable on {schema_cls.__name__}; "
                f"filterable fields: {sorted(filterable)}"
            )
        if key in choices_map and value not in choices_map[key]:
            raise ValueError(
                f"{key}={value!r} not in allowed choices {choices_map[key]}"
            )
        result[key] = {"$eq": value}
    return result
