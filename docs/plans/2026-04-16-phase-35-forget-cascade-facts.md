# Phase 35: `forget()` Actual Delete + Cascade to Derived Facts

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Wire the actual deletion path behind the inventory API from
Phase 34. `forget(..., dry_run=False)` now (a) enumerates the target
set via the Phase-34 matchers, (b) deletes the raw turns + derived
facts from `MemoryLayer`, (c) returns a `ForgetResult` with counts
and ids. Summary cascade ships in Phase 36.

**Architecture:**
- Flip `dry_run` default to `False`. Dry-run remains explicit via
  `dry_run=True`.
- New return type when `dry_run=False`:
  ```python
  @dataclass(frozen=True)
  class ForgetResult:
      deleted_turns: list[str]
      deleted_facts: list[str]
      deleted_summaries: list[str]    # always [] in Phase 35; Phase 36 fills
      total_deleted: int
  ```
- `forget(..., dry_run=True)` still returns a `ForgetPreview` (Phase
  34's shape). `forget(..., dry_run=False)` returns a `ForgetResult`.
  Different types — `typing.overload` clarifies the return type per
  dry_run value.
- Deletion order matters: delete facts FIRST (they reference the turns
  by `source_turn_id`; orphan fact records are harmless but confusing
  in debug output), THEN delete turns.
- Uses existing `MemoryLayer` delete APIs — no backend-specific code.
- Vector-store bulk delete: pass the full id list to
  `backend.remove(ids)` in one call per category.

**Safety:**
- Still requires at least one criterion; zero criteria raises
  `ValueError` (same guard as Phase 34).
- Async-extraction mode: `flush()` first so we don't wipe state that
  an in-flight extraction is about to write.
- Batch-extraction mode: drop pending buffer first (matches
  `clear_session` semantics — Phase 25 already did this for
  `clear_session`; same idea).

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/cookbook.md`, `deferred-items.md`.

---

### Task 1: `ForgetResult` + deletion path

**Files:**
- Modify: `src/soma/memory/conversational.py` (flip `dry_run` default,
  wire actual deletion, introduce `ForgetResult`)
- Extend: `tests/test_memory/test_conversational_forget_preview.py`
  OR create `tests/test_memory/test_conversational_forget_delete.py`
  (agent's call — second file is cleaner since the name changes
  from "preview" to "delete")

**Signature change:**
```python
@overload
def forget(self, *, dry_run: Literal[True], ...) -> ForgetPreview: ...
@overload
def forget(self, *, dry_run: Literal[False] = False, ...) -> ForgetResult: ...
def forget(self, *, dry_run: bool = False, ...):
    preview = self._forget_preview(...)
    if dry_run:
        return preview
    # Flush async / drop batch buffer before deletion
    if self._extraction_mode == "async":
        self.flush()
    elif self._extraction_mode == "batch":
        self._pending_batch = []
    # Delete facts first, then turns
    deleted_facts = self._memory.remove_ids(preview.derived_facts)
    deleted_turns = self._memory.remove_ids(preview.raw_turns)
    return ForgetResult(
        deleted_turns=deleted_turns,
        deleted_facts=deleted_facts,
        deleted_summaries=[],   # Phase 36
        total_deleted=len(deleted_turns) + len(deleted_facts),
    )
```

**Step 1: Failing tests.**
```python
def test_forget_delete_removes_raw_turns():
    cm.add_message("user", "gardening note 1")
    cm.add_message("user", "gardening note 2")
    baseline = cm.memory.ntotal
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert len(result.deleted_turns) == 2
    assert cm.memory.ntotal == baseline - 2 - len(result.deleted_facts)

def test_forget_delete_removes_derived_facts():
    # Stub LLM extracts: "user likes gardening"
    cm.add_message("user", "I like gardening")
    result = cm.forget(text_matches="gardening", dry_run=False)
    assert result.deleted_facts
    for fid in result.deleted_facts:
        assert fid not in cm.memory._id_to_node

def test_forget_delete_default_is_not_dry_run():
    # Calling forget() without dry_run= should actually delete.
    baseline = cm.memory.ntotal
    cm.add_message("user", "gardening")
    cm.forget(text_matches="gardening")
    assert cm.memory.ntotal < baseline + 1

def test_forget_dry_run_true_still_works():
    cm.add_message("user", "gardening")
    preview = cm.forget(text_matches="gardening", dry_run=True)
    assert preview.raw_turns
    assert cm.memory.ntotal > 0  # nothing was actually deleted

def test_forget_in_async_mode_flushes_first():
    cm = _make_cm(extraction_mode="async")
    cm.add_message("user", "gardening")   # extraction in-flight
    # Call forget before the executor drains
    result = cm.forget(text_matches="gardening")
    # The extracted fact should have landed first, then been deleted.
    # Net effect: no dangling in-flight extraction writes after forget
    # returns.
    assert result.deleted_facts   # fact landed + got deleted

def test_forget_in_batch_mode_drops_pending_buffer():
    cm = _make_cm(extraction_mode="batch", batch_size=8)
    for i in range(3):
        cm.add_message("user", f"gardening {i}")
    # Pending buffer has 3 turns; forget drops them without extracting
    result = cm.forget(text_matches="gardening")
    assert cm._pending_batch == []

def test_forget_returns_forget_result_when_not_dry_run():
    result = cm.forget(text_matches="x", dry_run=False)
    assert isinstance(result, ForgetResult)

def test_forget_returns_forget_preview_when_dry_run():
    preview = cm.forget(text_matches="x", dry_run=True)
    assert isinstance(preview, ForgetPreview)

def test_forget_empty_result_does_nothing_quietly():
    baseline = cm.memory.ntotal
    result = cm.forget(text_matches="nonexistent", dry_run=False)
    assert result.total_deleted == 0
    assert cm.memory.ntotal == baseline

def test_forget_summaries_list_empty_in_phase_35():
    # Summaries cascade is Phase 36; in Phase 35 the field is always [].
    result = cm.forget(text_matches="x")
    assert result.deleted_summaries == []
```

**Step 2-4: TDD.**

**Step 5:** `git commit -m "feat(conv): ConversationalMemory.forget() deletes raw turns + derived facts"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
SOMA_EMBED_MODEL=stub pytest tests/test_memory -q
```

Baseline post-Phase-34: +~12 tests. Target Phase 35: +~10 new, 0
regressions.

**Gotchas:**
- `MemoryLayer.remove_ids(ids)` may not exist yet. Check. If it's
  named `remove` or `delete`, use that. If bulk delete isn't exposed,
  add a thin bulk wrapper in `api.py` — that's in scope for this
  phase since it's directly required by the delete path.
- Vector-store bulk remove needs to flow to the underlying
  `VectorBackend.remove(ids)` which every adapter already implements
  (protocol contract). No backend code changes expected.
- Foreign-key-style integrity: facts reference turns by
  `source_turn_id` metadata. After deleting turns, facts would become
  orphans — we delete facts FIRST to avoid the window where orphans
  exist.
- The existing `clear_session` path stays — it's a different
  operation (wipe whole session, no match criteria). `forget` is the
  criterion-based GDPR path.
