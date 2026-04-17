"""SOMA Typed Schema Framework.

Public API::

    from soma.schemas import schema, field, get_schema, list_schemas

Built-in domain schemas (auto-registered on import)::

    from soma.schemas.builtin import TaskState, Fact, Note  # etc.
    # or import the whole package:
    import soma.schemas.builtin
"""

from soma.schemas.base import MISSING, field, schema
from soma.schemas.registry import get_schema, list_schemas

__all__ = ["schema", "field", "get_schema", "list_schemas", "MISSING"]
