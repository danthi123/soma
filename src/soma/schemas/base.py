"""@schema decorator, Field descriptor, generated methods."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from soma.schemas.registry import register_schema

__all__ = ["schema", "field", "MISSING"]

# Sentinel for "no default" — distinct from None.
MISSING: Any = dataclasses.MISSING


class _FieldSpec:
    """Metadata carrier attached to dataclass field via ``metadata=``."""

    __slots__ = ("filterable", "searchable", "choices", "description")

    def __init__(
        self,
        *,
        filterable: bool = False,
        searchable: bool = False,
        choices: list[str] | None = None,
        description: str = "",
    ) -> None:
        self.filterable = filterable
        self.searchable = searchable
        self.choices = list(choices) if choices is not None else None
        self.description = description


def field(
    *,
    filterable: bool = False,
    searchable: bool = False,
    choices: list[str] | None = None,
    default: Any = MISSING,
    description: str = "",
) -> Any:
    """Declare a schema field.

    filterable: this field can be used in retrieve filter kwargs.
    searchable: this field's value is included in the embedded text.
    choices: if set, validate value is in this set on store.
    default: default value (MISSING = required).
    """
    spec = _FieldSpec(
        filterable=filterable,
        searchable=searchable,
        choices=choices,
        description=description,
    )
    kwargs: dict[str, Any] = {"metadata": {"_soma_field": spec}}
    if default is not MISSING:
        kwargs["default"] = default
    return dc_field(**kwargs)


def _get_field_spec(f: dataclasses.Field[Any]) -> _FieldSpec | None:
    """Extract the _FieldSpec from a dataclass field, if present."""
    meta = f.metadata
    if meta and "_soma_field" in meta:
        return meta["_soma_field"]
    return None


def schema(type_name: str) -> Any:
    """Register a class as a typed SOMA schema.

    Wraps the class as a dataclass (if not already one), generates
    ``to_metadata()`` and ``from_metadata()`` methods, and registers it
    in the global schema registry under *type_name*.
    """

    def decorator(cls: type) -> type:
        # Pre-scan field annotations for choices so we can inject
        # __post_init__ BEFORE wrapping as a dataclass. The dataclass
        # decorator's generated __init__ only calls __post_init__ when
        # it exists at wrapping time.
        already_dc = dataclasses.is_dataclass(cls)

        # Peek at the field specs from annotations (pre-dataclass, the
        # class attributes are the dc_field() return values).
        _choices_map_pre: dict[str, list[str]] = {}
        if already_dc:
            for f in dataclasses.fields(cls):
                spec = _get_field_spec(f)
                if spec is not None and spec.choices is not None:
                    _choices_map_pre[f.name] = spec.choices
        else:
            for attr_name in list(getattr(cls, "__annotations__", {})):
                attr_val = getattr(cls, attr_name, dataclasses.MISSING)
                if isinstance(attr_val, dataclasses.Field):
                    spec = _get_field_spec(attr_val)
                    if spec is not None and spec.choices is not None:
                        _choices_map_pre[attr_name] = spec.choices

        # Inject __post_init__ for choices validation BEFORE dataclass().
        _original_post_init = getattr(cls, "__post_init__", None)

        def __post_init__(self: Any) -> None:
            if _original_post_init is not None:
                _original_post_init(self)
            for fname, allowed in _choices_map_pre.items():
                val = getattr(self, fname)
                if val is not None and val not in allowed:
                    raise ValueError(f"{fname}={val!r} not in allowed choices {allowed}")

        if _choices_map_pre:
            cls.__post_init__ = __post_init__  # type: ignore[attr-defined]

        # Wrap as dataclass if needed.
        if not already_dc:
            cls = dataclass(cls)

        # Stash the type name on the class.
        cls._type_name = type_name  # type: ignore[attr-defined]

        # Parse Meta inner class.
        raw_meta = getattr(cls, "Meta", None)
        meta_dict: dict[str, Any] = {}
        if raw_meta is not None:
            for key in ("consolidation", "ttl_seconds", "context_priority"):
                if hasattr(raw_meta, key):
                    meta_dict[key] = getattr(raw_meta, key)
        meta_dict.setdefault("consolidation", None)
        meta_dict.setdefault("ttl_seconds", None)
        meta_dict.setdefault("context_priority", 0.5)
        cls._schema_meta = meta_dict  # type: ignore[attr-defined]

        # Build field-spec index for fast access.
        fields = dataclasses.fields(cls)
        _searchable: list[str] = []
        _filterable: set[str] = set()
        _choices_map: dict[str, list[str]] = {}
        for f in fields:
            spec = _get_field_spec(f)
            if spec is None:
                continue
            if spec.searchable:
                _searchable.append(f.name)
            if spec.filterable:
                _filterable.add(f.name)
            if spec.choices is not None:
                _choices_map[f.name] = spec.choices

        cls._searchable_fields = _searchable  # type: ignore[attr-defined]
        cls._filterable_field_names = _filterable  # type: ignore[attr-defined]
        cls._choices_map = _choices_map  # type: ignore[attr-defined]

        # to_metadata() — returns dict with "type" key.
        def to_metadata(self: Any) -> dict[str, Any]:
            result: dict[str, Any] = {"type": type_name}
            for f in dataclasses.fields(self):
                val = getattr(self, f.name)
                if val is not None:
                    result[f.name] = val
            return result

        cls.to_metadata = to_metadata  # type: ignore[attr-defined]

        # from_metadata(meta) — classmethod reconstructing instance.
        @classmethod  # type: ignore[misc]
        def from_metadata(klass: type, meta: dict[str, Any]) -> Any:
            field_names = {f.name for f in dataclasses.fields(klass)}
            kwargs: dict[str, Any] = {}
            for k, v in meta.items():
                if k in field_names:
                    kwargs[k] = v
            return klass(**kwargs)

        cls.from_metadata = from_metadata  # type: ignore[attr-defined]

        # _search_text() — concatenate searchable fields.
        def _search_text(self: Any) -> str:
            parts: list[str] = []
            for fname in _searchable:
                val = getattr(self, fname)
                if val is not None:
                    parts.append(str(val))
            return " ".join(parts)

        cls._search_text = _search_text  # type: ignore[attr-defined]

        # _filterable_fields() — classmethod returning set of names.
        @classmethod  # type: ignore[misc]
        def _filterable_fields_cm(klass: type) -> set[str]:
            return set(klass._filterable_field_names)  # type: ignore[attr-defined]

        cls._filterable_fields = _filterable_fields_cm  # type: ignore[attr-defined]

        # Register.
        register_schema(type_name, cls)

        return cls

    return decorator
