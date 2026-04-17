# SOMA Typed Schemas -- Developer Guide

Typed schemas give structure to what you store in a `MemoryLayer`.
Instead of free-form `metadata={"role": "user", ...}` dicts, you
define a Python class, decorate it with `@schema`, and SOMA handles
validation, searchable-text extraction, filter enforcement, and
context packing automatically.

## Why schemas?

Without schemas every memory entry is a bag of keys.  That works for
prototypes, but once you have multiple entry types (facts, tasks,
decisions, customer tickets) you lose track of which keys exist,
which are queryable, and which values are legal.  Schemas fix this:

- **Validation at write time** -- required fields, choice
  constraints, and type annotations are checked before anything hits
  the store.
- **Filter safety at read time** -- `retrieve_typed` rejects unknown
  or non-filterable fields instead of silently returning zero hits.
- **Searchable-text extraction** -- fields marked `searchable=True`
  are concatenated into the embedded text automatically so you don't
  have to format them by hand.
- **Context packing** -- `pack_context` knows which schema types to
  pull and how much budget each deserves.
- **Discovery** -- `list_schemas()` shows everything registered;
  third-party packages auto-register on import.

The untyped `store(text, metadata={})` / `retrieve(query, k)` API is
unchanged.  Schemas are additive -- you adopt them per entry type, at
your own pace.

---

## Quick start

```python
from soma.schemas import schema, field
from soma.memory  import MemoryLayer

# 1. Define a schema
@schema("myapp.ticket")
class Ticket:
    customer_id: str = field(filterable=True)
    priority: str = field(filterable=True, choices=["p0", "p1", "p2", "p3"])
    status: str = field(filterable=True, default="open")
    description: str = field(searchable=True)
    resolution: str | None = field(default=None)

# 2. Store a typed instance
mem = MemoryLayer.with_sbert()
mem.store_typed(Ticket(
    customer_id="cust-42",
    priority="p1",
    description="Login page returns 500",
))

# 3. Retrieve with type-safe filters
tickets = mem.retrieve_typed(
    Ticket,
    query="login error",
    k=5,
    priority="p1",
    status="open",
)
for t in tickets:
    print(t.description, t.priority)
```

---

## Defining a schema

### The `@schema` decorator

```python
from soma.schemas import schema, field

@schema("domain.type_name")
class MyType:
    required_field: str = field(filterable=True)
    optional_field: str = field(default="", searchable=True)
```

`@schema(type_name)` does three things:

1. Wraps the class as a `dataclass` (if it isn't one already).
2. Generates `to_metadata()`, `from_metadata()`, `_search_text()`,
   and `_filterable_fields()` methods.
3. Registers the class in the global schema registry under
   `type_name`.

The `type_name` string must be globally unique.  By convention it
uses a `domain.noun` format (e.g. `agent.task_state`,
`customer.issue`, `myapp.ticket`).

### The `field()` function

Every annotated attribute on a schema class should use `field()`.

| Option        | Type               | Default      | Description |
|---------------|--------------------|--------------|-------------|
| `filterable`  | `bool`             | `False`      | Can be used as a keyword filter in `retrieve_typed`. |
| `searchable`  | `bool`             | `False`      | Value is included in the embedded text for semantic search. |
| `choices`     | `list[str] | None` | `None`       | If set, the value is validated against this list on construction. |
| `default`     | `Any`              | *(required)* | Default value.  Omit to make the field required. |
| `description` | `str`              | `""`         | Human-readable description (for tooling / introspection). |

A field can be both `filterable` and `searchable` (e.g.
`customer_id` that you want to filter on AND include in the embedded
text).

**Supported field types:** `str`, `int`, `float`, `bool`,
`str | None` (optional).  Complex types (`list`, `dict`, nested
dataclasses) are not supported -- flatten them into comma-separated
strings or JSON-encoded strings.

### The `Meta` inner class (optional)

```python
@schema("myapp.ticket")
class Ticket:
    ...

    class Meta:
        consolidation = "supersede"   # hint for memory consolidation
        ttl_seconds = 86400 * 90      # expiry hint (90 days)
        context_priority = 0.8        # priority in context packing [0, 1]
```

| Attribute          | Default | Description |
|--------------------|---------|-------------|
| `consolidation`    | `None`  | Consolidation strategy hint (`"supersede"`, `"merge"`, or `None`). |
| `ttl_seconds`      | `None`  | Suggested time-to-live in seconds. |
| `context_priority` | `0.5`   | Weight when `pack_context` assembles the prompt. |

---

## Storing typed instances

```python
mem.store_typed(Ticket(
    customer_id="cust-42",
    priority="p1",
    description="Login page returns 500",
))
```

`store_typed()`:
1. Validates the instance (required fields, choices).
2. Calls `_search_text()` to build the string that gets embedded.
3. Calls `to_metadata()` to build the metadata dict (includes a
   `"type"` key with the schema's type name).
4. Delegates to the underlying `store(text, metadata)`.

If validation fails, a `ValueError` is raised and nothing is stored.

### Validation rules

- **Required fields** -- any field without a `default` must be passed
  a non-None value.
- **Choices** -- if `choices=["a", "b"]` is set, the value must be
  one of those strings.  `None` is allowed only if the field has
  `default=None`.
- Validation runs at construction time (`__post_init__`), not on
  later attribute mutation.  If you mutate an instance after
  construction, constraints are not re-checked.

---

## Retrieving typed instances

```python
tickets = mem.retrieve_typed(
    Ticket,
    query="login error",
    k=5,
    priority="p1",       # filter kwarg
    status="open",       # filter kwarg
)
# Returns list[Ticket]
```

Filter kwargs are validated against the schema:

- Unknown field names raise `ValueError`.
- Non-filterable field names raise `ValueError`.
- Values that violate the field's `choices` constraint raise
  `ValueError`.

Under the hood, `retrieve_typed` builds a Chroma-compatible `where`
dict and calls `mem.retrieve(query, k, where=...)`.  The returned
metadata dicts are reconstructed into schema instances via
`from_metadata()`.

---

## Registry

```python
from soma.schemas import get_schema, list_schemas

cls = get_schema("myapp.ticket")   # returns the Ticket class
names = list_schemas()             # ["agent.task_state", ..., "myapp.ticket"]
```

Schemas register at import time via `@schema()`.  The registry is a
plain dict keyed by type name.  Duplicate type names raise
`ValueError` at import time.

### Inspecting a schema at runtime

```python
import dataclasses

for f in dataclasses.fields(Ticket):
    print(f.name, f.type, f.default)

print(Ticket._filterable_fields())   # {"customer_id", "priority", "status"}
print(Ticket._choices_map)           # {"priority": ["p0","p1","p2","p3"]}
print(Ticket._searchable_fields)     # ["description"]
```

---

## Built-in schemas (31 types across 8 domains)

SOMA ships 31 schemas organised into eight domains.  All are
auto-registered when you `import soma.schemas.builtin` (or any of its
sub-modules).

### Agent (`soma.schemas.builtin.agent`)

| Type name            | Class         | Purpose |
|----------------------|---------------|---------|
| `agent.task_state`   | `TaskState`   | Task lifecycle tracking (pending/active/blocked/done/failed). |
| `agent.tool_call`    | `ToolCall`    | Records a tool invocation + outcome. |
| `agent.observation`  | `Observation` | An observation from tool, user, env, or internal source. |
| `agent.decision`     | `Decision`    | A decision with rationale and alternatives. |

### Conversation (`soma.schemas.builtin.conv`)

| Type name              | Class           | Purpose |
|------------------------|-----------------|---------|
| `conv.fact`            | `Fact`          | Subject-predicate-object triple from conversation. |
| `conv.preference`      | `Preference`    | A learned user preference. |
| `conv.contradiction`   | `Contradiction` | Two contradicting facts + optional resolution. |

### Knowledge management (`soma.schemas.builtin.knowledge`)

| Type name        | Class        | Purpose |
|------------------|--------------|---------|
| `km.note`        | `Note`       | A knowledge note (manual, extracted, imported). |
| `km.connection`  | `Connection` | A typed link between two entries. |
| `km.question`    | `Question`   | An open question with optional answer link. |
| `km.insight`     | `Insight`    | A synthesised claim with evidence and confidence. |

### Code intelligence (`soma.schemas.builtin.code`)

| Type name              | Class            | Purpose |
|------------------------|------------------|---------|
| `code.decision`        | `CodeDecision`   | Architectural/implementation decision with rationale. |
| `code.pattern`         | `Pattern`        | A reusable code pattern with usage guidance. |
| `code.incident`        | `Incident`       | Post-mortem: severity, root cause, fix, prevention. |
| `code.dependency_note` | `DependencyNote` | Dependency risk, version pin, upgrade blocker. |

### Research (`soma.schemas.builtin.research`)

| Type name              | Class        | Purpose |
|------------------------|--------------|---------|
| `research.hypothesis`  | `Hypothesis` | A claim with lifecycle (proposed/testing/confirmed/refuted). |
| `research.experiment`  | `Experiment` | Planned or running experiment tied to a hypothesis. |
| `research.result`      | `Result`     | Experiment outcome (pass/fail/ambiguous) + metrics. |
| `research.literature`  | `Literature` | Literature reference with relevance annotation. |

### Collaboration (`soma.schemas.builtin.collab`)

| Type name                      | Class                 | Purpose |
|--------------------------------|-----------------------|---------|
| `collab.action_item`           | `ActionItem`          | Task assigned in a meeting, with status and due date. |
| `collab.decision`              | `CollabDecision`      | Group decision with participants and dissent. |
| `collab.follow_up`             | `FollowUp`            | Links two meetings via a carry-forward topic. |
| `collab.stakeholder_position`  | `StakeholderPosition` | A person's stance on a topic. |

### Customer (`soma.schemas.builtin.customer`)

| Type name             | Class               | Purpose |
|-----------------------|---------------------|---------|
| `customer.profile`    | `Profile`           | Customer profile with tier and contact history. |
| `customer.issue`      | `CustomerIssue`     | Issue with resolution tracking. |
| `customer.sentiment`  | `Sentiment`         | Sentiment signal from a customer interaction. |
| `customer.preference` | `CustomerPreference` | Preference (explicit or inferred). |

### Creative (`soma.schemas.builtin.creative`)

| Type name              | Class        | Purpose |
|------------------------|--------------|---------|
| `creative.character`   | `Character`  | Fictional character with traits, arc, relationships. |
| `creative.world_detail`| `WorldDetail`| World-building detail (geography, politics, magic, etc.). |
| `creative.continuity`  | `Continuity` | Continuity assertion, possibly contradicted later. |
| `creative.plot_thread` | `PlotThread` | Narrative thread lifecycle (planted/developing/resolved). |

---

## Context packing

`pack_context` assembles a prompt-ready string from memory, mixing
entries by slot with per-slot token budgets.

```python
from soma.schemas.packing import pack_context

context = pack_context(
    mem,
    query="what is the deploy plan?",
    max_tokens=3800,
)
# Returns a string like:
# [agent.task_state] deploy migration -- status=active step=3
# [conv.preference] user prefers blue-green deploys
# [agent.decision] chose Kubernetes over ECS -- simpler networking
```

### Parameters

| Parameter        | Type               | Default | Description |
|------------------|--------------------|---------|-------------|
| `mem`            | `MemoryLayer`      | --      | Memory layer to pull entries from. |
| `query`          | `str`              | --      | Current user query for semantic retrieval. |
| `max_tokens`     | `int`              | `3800`  | Approximate token budget for the output. |
| `mix`            | `dict | None`      | `None`  | Slot weights (see below).  `None` = built-in defaults. |
| `types`          | `list[str] | None` | `None`  | Restrict to schema types matching these globs. |
| `chars_per_token`| `float`            | `4.0`   | Characters per token for budget calculation. |

### Default mix

```python
{
    "recency":     0.15,   # recent entries regardless of type
    "relevant":    0.50,   # semantically relevant to query
    "task_state":  0.10,   # active agent.task_state entries
    "decisions":   0.10,   # agent.decision entries
    "preferences": 0.15,   # conv.preference entries
}
```

Override by passing a custom `mix` dict.  Weights are normalised to
sum to 1.0.  Unknown slot names are treated as schema type names for
extensibility (e.g. `{"code.incident": 0.3, "relevant": 0.7}`).

### Type filtering

Pass `types=["agent.*", "code.*"]` to restrict all slots to entries
whose schema type matches one of the glob patterns.

---

## Extending with custom schemas

### From your own package

```python
# myagent/schemas.py
from soma.schemas import schema, field

@schema("myagent.task")
class Task:
    goal: str = field(searchable=True)
    status: str = field(filterable=True, choices=["todo", "done"])
```

Then in your entrypoint:

```python
import myagent.schemas  # registers Task at import time
```

That's it.  `list_schemas()` will now include `"myagent.task"` and
`retrieve_typed(Task, ...)` will work.

### From a third-party package

Third-party schema packages follow the same pattern.  The convention
is to put an `import` of the schema module in the package's
`__init__.py` so registration happens at `import mypackage` time.

```
mypackage/
  __init__.py      # imports .schemas
  schemas.py       # @schema("mypackage.foo") class Foo: ...
```

No configuration, no plugin system -- just Python imports.

### Naming conventions

- Use `domain.noun` format: `crm.lead`, `devops.deploy`, etc.
- Avoid generic names that might collide (`task`, `item`).
- Third-party packages should prefix with their package name.

---

## Backward compatibility

The untyped API is unchanged:

```python
mem.store("user lives in Portland", metadata={"user": "alex"})
hits = mem.retrieve("where does the user live?", k=3)
```

Typed entries stored via `store_typed` are just regular entries with
structured metadata.  `retrieve()` returns them as `Hit` objects with
a `.metadata` dict.  `retrieve_typed()` reconstructs them into schema
instances.  You can mix typed and untyped entries in the same
`MemoryLayer`.

---

## Serialisation: `to_metadata()` / `from_metadata()`

Every schema instance can round-trip through a plain dict:

```python
ticket = Ticket(customer_id="c-1", priority="p0", description="fire")
meta = ticket.to_metadata()
# {"type": "myapp.ticket", "customer_id": "c-1", "priority": "p0",
#  "status": "open", "description": "fire"}

restored = Ticket.from_metadata(meta)
assert restored == ticket
```

- `to_metadata()` includes all non-None fields plus a `"type"` key.
- `from_metadata(meta)` ignores the `"type"` key and any extra keys
  not in the schema.  This lets metadata evolve (add fields) without
  breaking deserialisation of old entries.
- `None`-valued optional fields are omitted from the dict.

---

## API reference

### `soma.schemas`

| Symbol          | Description |
|-----------------|-------------|
| `schema(type_name)` | Class decorator that registers a typed schema. |
| `field(...)` | Declare a schema field with metadata options. |
| `get_schema(type_name)` | Look up a registered schema class by name. |
| `list_schemas()` | List all registered type names. |
| `MISSING` | Sentinel for "no default" (re-exported from `dataclasses`). |

### `soma.schemas.packing`

| Symbol          | Description |
|-----------------|-------------|
| `pack_context(mem, query, ...)` | Assemble prompt-ready context from memory. |

### `soma.schemas.validation`

| Symbol          | Description |
|-----------------|-------------|
| `validate_instance(instance)` | Validate a schema instance (required fields, choices). |
| `validate_filter_kwargs(schema_cls, **kwargs)` | Validate filter kwargs against a schema, return Chroma-compatible dict. |

### `MemoryLayer` typed methods

| Method | Description |
|--------|-------------|
| `store_typed(instance)` | Validate + store a schema instance. |
| `retrieve_typed(cls, query, k, **filters)` | Retrieve and reconstruct typed instances. |

---

## Known limitations

- **No nested schemas.** Fields cannot be other schema types.
  Flatten to strings or IDs.
- **No list fields.** Use comma-separated strings (e.g.
  `tags: str = field(default="")`).
- **No enforcement on untyped store.** `store(text, metadata={})`
  bypasses all schema validation.  Only `store_typed` validates.
- **Choices validated at construction only.** Mutating a field after
  `__init__` can violate constraints silently.
- **Filter pushdown is backend-dependent.** `InProc` does Python-side
  filtering (correct, but not server-optimised).  Qdrant and LanceDB
  push filters to the engine.
- **`from_metadata()` ignores extra keys.** This is intentional for
  forward compatibility but means typos in metadata keys are silently
  dropped.
- **No migration tooling.** If you rename a field, old entries in the
  store will have the old key.  `from_metadata` will ignore it and
  use the default.  Handle migrations at the application level.
