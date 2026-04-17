# SOMA Typed Schemas -- Developer Guide

> **Status:** Phase 42 (framework). Built-in domains ship in Phase 43.

## Defining a Schema

```python
from soma.schemas import schema, field

@schema("myapp.ticket")
class Ticket:
    customer_id: str = field(filterable=True)
    priority: str = field(filterable=True, choices=["p0", "p1", "p2", "p3"])
    status: str = field(filterable=True, default="open")
    description: str = field(searchable=True)
    resolution: str | None = field(default=None)

    class Meta:
        consolidation = "supersede"
        ttl_seconds = 86400 * 90
        context_priority = 0.8
```

### `field()` options

| Option       | Type             | Default   | Description                                  |
|------------- |------------------|-----------|----------------------------------------------|
| `filterable` | `bool`           | `False`   | Can be used in `retrieve_typed` filter kwargs |
| `searchable` | `bool`           | `False`   | Value included in embedded text               |
| `choices`    | `list[str]|None` | `None`    | Validate value on construction                |
| `default`    | `Any`            | (required)| Default value; omit for required fields       |
| `description`| `str`            | `""`      | Human-readable field description              |

### `Meta` inner class (optional)

| Attribute          | Default | Description                        |
|--------------------|---------|------------------------------------|
| `consolidation`    | `None`  | Consolidation hint (Phase 43)      |
| `ttl_seconds`      | `None`  | Expiry hint in seconds (Phase 43)  |
| `context_priority` | `0.5`   | Priority in context packing [0, 1] |

## Storing Typed Instances

```python
from soma.memory.api import MemoryLayer

mem = MemoryLayer(embed_fn=my_embed, embed_dim=768)

mem.store_typed(Ticket(
    customer_id="cust-42",
    priority="p1",
    description="Login page returns 500",
))
```

`store_typed()` validates the instance, extracts searchable fields for
embedding, and stores all fields as metadata.

## Retrieving Typed Instances

```python
tickets = mem.retrieve_typed(
    Ticket,
    query="login error",
    k=5,
    priority="p1",
    status="open",
)
# Returns list[Ticket]
```

Filter kwargs are validated against the schema at call time. Passing an
unknown or non-filterable field raises `ValueError`.

## Registry

```python
from soma.schemas import get_schema, list_schemas

cls = get_schema("myapp.ticket")  # returns Ticket class
names = list_schemas()            # ["myapp.ticket", ...]
```

Schemas register at import time via `@schema()`. Third-party packages
register by importing their schema module.

## Backward Compatibility

`store(text, metadata={})` and `retrieve(query, k)` are unchanged.
The typed API is additive.

## Context Packing (stub)

```python
from soma.schemas.packing import pack_context

context = pack_context(tickets, separator="\n---\n")
```

Phase 43 adds priority-aware mixing and token budgeting.

## Extending with Custom Domains

Define schemas in your package, import them at startup, and they
auto-register. Example:

```python
# myagent/schemas.py
from soma.schemas import schema, field

@schema("myagent.task")
class Task:
    goal: str = field(searchable=True)
    status: str = field(filterable=True, choices=["todo", "done"])
```

Then in your agent entrypoint:

```python
import myagent.schemas  # registers Task
```

## Known Limitations

- Choices validation runs on construction (`__post_init__`), not on
  field assignment. Mutating an instance after construction can violate
  constraints.
- Filter pushdown depends on the backend. InProc falls back to
  Python-side filtering (works, but no server-side optimization).
- `from_metadata()` ignores extra keys in the metadata dict.
