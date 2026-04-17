# Phase 42: Typed Schema Framework

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Ship the core schema framework — `@schema` decorator,
`field()` descriptors, global registry, validation on store,
typed retrieval, and `MemoryLayer` integration. After this phase,
developers can define custom domain schemas and get validated
store/retrieve with zero boilerplate. Built-in domains (agent, conv,
knowledge) ship in Phase 43.

**Architecture:**

```
src/soma/schemas/
├── __init__.py      # public API: schema, field, get_schema, list_schemas
├── base.py          # @schema decorator, Field descriptor, SchemaBase
├── registry.py      # global registry, discovery, import-time registration
├── validation.py    # validate metadata dict against a schema
└── packing.py       # context packer (Phase 43 fleshes it out; stub here)
```

**Developer-facing API (the whole pitch):**

```python
from soma.schemas import schema, field

@schema("myapp.ticket")
class Ticket:
    customer_id: str = field(filterable=True)
    priority: str = field(filterable=True, choices=["p0","p1","p2","p3"])
    status: str = field(filterable=True, default="open")
    description: str = field(searchable=True)
    resolution: str | None = field(default=None)

    class Meta:
        consolidation = "supersede"   # latest by (customer_id, type) wins
        ttl_seconds = 86400 * 90      # 90-day expiry hint
        context_priority = 0.8        # high priority in context packing

# Store (typed):
mem.store_typed(Ticket(
    customer_id="cust-42",
    priority="p1",
    description="Login page returns 500 after password reset",
))

# Retrieve (typed):
tickets = mem.retrieve_typed(
    Ticket,
    query="login error",
    k=5,
    priority="p1",        # filter kwargs validated against schema
    status="open",
)
# Returns list[Ticket], not list[Hit]

# Untyped store still works (backward compat, no validation):
mem.store("hello world", metadata={"custom": "anything"})
```

**Design constraints:**

1. **Backward compatible.** `store(text, metadata={})` unchanged.
   Typed path is additive via `store_typed()` / `retrieve_typed()`.
2. **No metaclass magic.** The `@schema` decorator uses
   `dataclasses.dataclass` under the hood (or wraps an existing
   dataclass). No custom metaclasses, no descriptor protocols beyond
   what `field()` returns.
3. **Registration is import-time.** `@schema("name")` registers in a
   global dict. `get_schema("name")` returns the class.
   `list_schemas()` returns all registered names. Third-party
   packages register by importing their schema module.
4. **Validation is opt-in per call.** `store_typed()` validates;
   `store()` does not. `retrieve_typed()` reconstructs typed objects;
   `retrieve()` returns raw `Hit` objects.
5. **Filter kwargs on `retrieve_typed()` are validated against the
   schema.** Passing `priority="p5"` when `choices=["p0".."p3"]`
   raises `ValueError` at call time, not at query time.
6. **`searchable=True` fields are concatenated into the embedded
   text.** `filterable=True` fields go into metadata for filter
   pushdown. A field can be both.
7. **`Meta` inner class is optional.** Defaults: no consolidation
   hint, no TTL, priority=0.5.

**Out-of-scope (Phases 43-44):**
- Built-in domain schemas (agent, conv, knowledge, etc.)
- Context packer implementation (stub in this phase)
- Consolidation hooks (stub in this phase)
- Developer documentation site / cookbook recipes
- `CHANGELOG.md`, `deferred-items.md`

---

### Task 1: `field()` descriptor + `@schema` decorator

**Files:**
- Create: `src/soma/schemas/__init__.py`
- Create: `src/soma/schemas/base.py`
- Create: `src/soma/schemas/registry.py`
- Create: `tests/test_schemas/__init__.py`
- Create: `tests/test_schemas/test_base.py`
- Create: `tests/test_schemas/test_registry.py`

**`field()` API:**
```python
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
```

**`@schema` decorator:**
```python
def schema(type_name: str):
    """Register a class as a typed SOMA schema.

    Wraps the class as a dataclass (if not already one), generates
    to_metadata() and from_metadata() methods, and registers it in
    the global schema registry under `type_name`.
    """
```

**`to_metadata()` generated method:**
Returns `{"type": type_name, field_name: field_value, ...}` for all
fields. Only includes non-None values (sparse metadata).

**`from_metadata(meta: dict)` generated classmethod:**
Reconstructs the schema instance from a metadata dict. Missing
fields with defaults use the default; missing required fields raise.

**Tests:**
```python
def test_schema_decorator_creates_dataclass():
    @schema("test.widget")
    class Widget:
        name: str = field(filterable=True)
        color: str = field(default="red")
    w = Widget(name="gear")
    assert w.name == "gear"
    assert w.color == "red"

def test_to_metadata_includes_type():
    w = Widget(name="gear")
    meta = w.to_metadata()
    assert meta["type"] == "test.widget"
    assert meta["name"] == "gear"

def test_from_metadata_round_trips():
    w = Widget(name="gear", color="blue")
    w2 = Widget.from_metadata(w.to_metadata())
    assert w == w2

def test_field_choices_validated_on_construction():
    @schema("test.priority")
    class PrioItem:
        level: str = field(choices=["low","med","high"])
    PrioItem(level="low")  # ok
    with pytest.raises(ValueError):
        PrioItem(level="ultra")

def test_schema_registered_in_registry():
    assert get_schema("test.widget") is Widget
    assert "test.widget" in list_schemas()

def test_schema_double_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        @schema("test.widget")
        class Widget2:
            x: int = field()

def test_searchable_fields_extracted():
    @schema("test.doc")
    class Doc:
        title: str = field(searchable=True)
        body: str = field(searchable=True)
        tag: str = field(filterable=True)
    d = Doc(title="hello", body="world", tag="test")
    assert d._search_text() == "hello world"

def test_filterable_fields_listed():
    assert Widget._filterable_fields() == {"name"}
```

**Step 5:** `git commit -m "feat(schemas): @schema decorator + field() + registry"`

---

### Task 2: Validation module

**Files:**
- Create: `src/soma/schemas/validation.py`
- Create: `tests/test_schemas/test_validation.py`

**API:**
```python
def validate_instance(instance: Any) -> None:
    """Validate a schema instance.

    Checks: required fields present, choice fields in range,
    type annotations match (basic isinstance check).
    Raises ValueError with a clear message on failure.
    """

def validate_filter_kwargs(schema_cls: type, **kwargs) -> dict:
    """Validate filter kwargs against a schema.

    Returns the validated filter dict suitable for MemoryLayer.retrieve.
    Raises ValueError if a kwarg doesn't correspond to a filterable
    field, or if the value violates the field's choices constraint.
    """
```

**Tests:**
```python
def test_validate_instance_ok(): ...
def test_validate_instance_missing_required_raises(): ...
def test_validate_instance_bad_choice_raises(): ...
def test_validate_filter_kwargs_ok(): ...
def test_validate_filter_kwargs_non_filterable_raises(): ...
def test_validate_filter_kwargs_bad_choice_raises(): ...
def test_validate_filter_kwargs_converts_to_filter_spec():
    # priority="p1" → {"priority": {"$eq": "p1"}}
```

**Step 5:** `git commit -m "feat(schemas): validation for instances + filter kwargs"`

---

### Task 3: MemoryLayer integration (`store_typed` / `retrieve_typed`)

**Files:**
- Modify: `src/soma/memory/api.py` — add `store_typed()` and
  `retrieve_typed()` methods to `MemoryLayer`
- Extend: `tests/test_memory/test_api.py`

**`store_typed` method:**
```python
def store_typed(self, instance: Any, *, metadata: dict | None = None) -> str:
    """Store a typed schema instance.

    1. Validate the instance against its schema.
    2. Extract searchable fields → concatenate into the text to embed.
    3. Extract all fields → merge into metadata (with type= prefix).
    4. Merge any extra `metadata` kwargs (caller overrides).
    5. Call self.store(text, metadata=merged).
    """
```

**`retrieve_typed` method:**
```python
def retrieve_typed(
    self,
    schema_cls: type,
    query: str,
    k: int = 5,
    **filter_kwargs,
) -> list[Any]:
    """Retrieve and reconstruct typed schema instances.

    1. Validate filter_kwargs against schema_cls.
    2. Build filter dict: {"type": schema_cls._type_name, **validated_filters}.
    3. Call self.retrieve(query, k, filter=filter_dict).
    4. Reconstruct each hit into a schema_cls instance via from_metadata.
    5. Return list[schema_cls].
    """
```

**Tests:**
```python
def test_store_typed_round_trip(tmp_path):
    @schema("test.note")
    class Note:
        title: str = field(searchable=True)
        tag: str = field(filterable=True)
    mem = MemoryLayer(embed_fn=stub, embed_dim=128)
    mem.store_typed(Note(title="hello world", tag="test"))
    results = mem.retrieve_typed(Note, "hello", k=1, tag="test")
    assert len(results) == 1
    assert results[0].title == "hello world"
    assert results[0].tag == "test"

def test_store_typed_validates():
    @schema("test.strict")
    class Strict:
        level: str = field(choices=["a","b"])
    with pytest.raises(ValueError):
        mem.store_typed(Strict(level="z"))

def test_retrieve_typed_validates_filter_kwargs():
    with pytest.raises(ValueError, match="not filterable"):
        mem.retrieve_typed(Note, "hello", k=1, title="x")
        # title is searchable, not filterable

def test_untyped_store_still_works():
    # Backward compat: store(text, metadata={}) unchanged
    mem.store("raw text", metadata={"custom": "field"})
    hits = mem.retrieve("raw text", k=1)
    assert len(hits) == 1
```

**Step 5:** `git commit -m "feat(api): store_typed / retrieve_typed for schema-driven memory"`

---

### Task 4: Context packer stub + docs skeleton

**Files:**
- Create: `src/soma/schemas/packing.py` — stub `pack_context()` that
  returns a simple concatenation of retrieve results. Phase 43
  implements the real mixing strategy.
- Create: `docs/schemas.md` — developer guide skeleton: how to define
  a schema, how to store/retrieve typed instances, how to extend with
  custom domains. Short (100-150 lines); Phase 44 fills in the
  full reference.

**Step 5:** `git commit -m "feat(schemas): context packer stub + developer guide skeleton"`

---

### Final sanity

```bash
ruff check src/soma/schemas tests/test_schemas src/soma/memory
pytest tests/test_schemas tests/test_memory -q
```

Baseline post-Phase-37 memory suite: ~570 tests. Target +~20 schema
tests + 4 MemoryLayer integration tests, 0 regressions.

**Gotchas:**
- `@schema` must not break if the class already IS a dataclass
  (caller used `@dataclass` before `@schema`). Detect and skip the
  re-wrapping.
- `choices` validation runs on construction (in `__post_init__`), not
  on field assignment. Mutable instances can violate choices after
  construction — document this as a known limitation.
- Filter pushdown only works on backends that support it (InProc does
  not). `retrieve_typed` with filter kwargs against InProc should
  gracefully fall back to Python-side filtering (already the default
  behavior in MemoryLayer). No special handling needed.
- `from_metadata` must handle extra keys gracefully (ignore them).
  Metadata may carry fields from other code paths; the schema only
  cares about its own fields.
- The `type` field in metadata must not collide with Python's
  `type()` builtin. The `to_metadata()` method uses the string key
  `"type"`, which is fine — it's a dict key, not a variable name.
