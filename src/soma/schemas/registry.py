"""Global schema registry — import-time registration, runtime discovery."""

from __future__ import annotations

__all__ = ["register_schema", "get_schema", "list_schemas"]

_REGISTRY: dict[str, type] = {}


def register_schema(type_name: str, cls: type) -> None:
    """Register *cls* under *type_name*. Raises on duplicates."""
    if type_name in _REGISTRY:
        raise ValueError(
            f"Schema {type_name!r} already registered (by {_REGISTRY[type_name].__qualname__})"
        )
    _REGISTRY[type_name] = cls


def get_schema(type_name: str) -> type:
    """Return the class registered under *type_name*, or raise KeyError."""
    return _REGISTRY[type_name]


def list_schemas() -> list[str]:
    """Return all registered type names."""
    return list(_REGISTRY.keys())


def _clear_registry() -> None:
    """Test-only helper — wipe all registrations."""
    _REGISTRY.clear()
