# Phase 25: Batch Extraction Mode for `ConversationalMemory`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Add `extraction_mode="batch"` so K consecutive turns are
accumulated and a single LLM call extracts facts for all of them.
Cuts LLM-call cost ~K× for workloads that tolerate extraction latency
lagging by up to K turns. Default stays `"sync"`; `"async"` (Phase 22)
and `"batch"` are opt-in.

**Architecture:**
- New kwarg `batch_size: int = 8` (only used when `extraction_mode="batch"`).
- `add_message` appends `(role, text, turn_id)` to `self._pending_batch`
  and returns immediately after the raw-turn write. When
  `len(pending) >= batch_size`, call `_run_batch_extract_reconcile()`
  inline (still cheaper than K separate calls since it's one LLM hit).
- `flush(timeout=None)` drains partial batches: if 0 < pending < K,
  fire extraction on the partial set.
- `close()` calls `flush()`; context manager wired.
- `clear_session()` drops the pending buffer without extracting
  (semantics: wipe means wipe — don't write facts that the user is
  trying to forget).

**Batch LLM prompt:** the extractor LLM is already prompt-driven.
Extend the prompt to take a numbered list of turns and return facts
tagged by turn index. Parser routes facts back to the right
`source_turn_id` metadata key. Reuse the existing reconcile pipeline
per-fact (no batching there — reconcile ordering matters).

**Interaction with `extraction_mode="async"`:** not combined in this
phase. `batch` is synchronous-but-aggregated. Async+batch is a future
phase if anyone asks.

**Safety invariants:**
- Ordering preserved: turns flushed in submission order, so
  supersede/same-subject reconcile stays deterministic.
- `clear_session()` drops pending, doesn't extract.
- Exceptions from the batched LLM call surface synchronously on the
  `add_message` that triggered the flush (not the call that
  accumulated an earlier turn).

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/cookbook.md`, `deferred-items.md` strikethrough.

---

### Task 1: `batch_size` kwarg + buffering

**Files:**
- Modify: `src/soma/memory/conversational.py`
- Create: `tests/test_memory/test_conversational_batch_extraction.py`

**Constructor signature addition:**
```python
def __init__(
    self,
    *,
    # ... existing kwargs ...
    extraction_mode: Literal["sync", "async", "batch"] = "sync",
    batch_size: int = 8,
) -> None:
    if extraction_mode == "batch" and batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    self._batch_size = batch_size
    self._pending_batch: list[tuple[str, str, str]] = []  # (role, text, turn_id)
```

**`add_message` flow in batch mode:**
```python
if self._extraction_mode == "batch":
    self._pending_batch.append((role, text, turn_id))
    if len(self._pending_batch) >= self._batch_size:
        self._flush_batch()
elif self._extraction_mode == "async":
    # existing async path
    ...
else:
    # existing sync path
    ...
```

**`_flush_batch()`:**
```python
def _flush_batch(self) -> None:
    if not self._pending_batch:
        return
    batch = self._pending_batch
    self._pending_batch = []
    self._run_batch_extract_reconcile(batch)
```

**`flush(timeout=None)` update:**
```python
def flush(self, timeout: float | None = None) -> None:
    if self._extraction_mode == "async":
        # existing drain path
        ...
    elif self._extraction_mode == "batch":
        self._flush_batch()
    # sync mode: no-op
```

**`clear_session()` update:**
```python
def clear_session(self, *, session_id: str | None = None) -> None:
    if self._extraction_mode == "async":
        self.flush()
    elif self._extraction_mode == "batch":
        self._pending_batch = []  # drop, don't extract
    # existing wipe logic
```

**Step 1: Failing tests.**
```python
def test_batch_mode_accumulates_until_K():
    # 7 add_messages, batch_size=8: no extraction has fired yet
    # 8th add_message: single extract call observed, all 8 turns in it

def test_batch_mode_flush_drains_partial():
    # 3 add_messages, batch_size=8, then flush(): extraction fires on 3

def test_batch_mode_close_flushes_partial():
    with ConversationalMemory(extraction_mode="batch", batch_size=8, ...) as cm:
        cm.add_message("user", "a")
        cm.add_message("user", "b")
    # exit flushes; facts from both turns in memory

def test_batch_mode_clear_session_drops_pending_without_extracting():
    # 3 add_messages, batch_size=8, clear_session()
    # No LLM calls fired; pending buffer empty

def test_batch_mode_preserves_turn_order_in_facts():
    # 3 add_messages with distinguishable content; flush
    # Facts come back tagged with the right source_turn_id

def test_batch_mode_exceptions_surface_on_trigger_call():
    # Stub LLM raises; 8th add_message raises (not a prior one)

def test_sync_and_async_modes_unchanged():
    # batch_size=8 with mode="sync": batch_size ignored, classic sync fires each turn

def test_batch_size_validation():
    with pytest.raises(ValueError):
        ConversationalMemory(extraction_mode="batch", batch_size=0, ...)
```

**Step 2-4: TDD.**

**Step 5:** `git commit -m "feat(conv): extraction_mode=\"batch\" accumulates K turns per LLM call"`

---

### Task 2: Batched extractor prompt

**Files:**
- Modify: `src/soma/memory/conversational.py`

Extend the existing extractor prompt builder. The current single-turn
prompt asks for facts from one text; the batch version asks for facts
from N numbered turns, returning JSON with a `turn_index` field per
fact.

**Prompt shape:**
```
Extract facts from these {N} turns. Return a JSON array of
{{turn_index, subject, predicate, object, confidence}}.
Turn 0 (user): <text>
Turn 1 (assistant): <text>
...
```

Parser maps `turn_index` → `turn_id` from the original batch tuple and
stamps `source_turn_id` metadata the same as the single-turn path.

**Robustness:**
- If the LLM returns facts without `turn_index`, attribute them to the
  last turn in the batch (conservative fallback, logged at WARNING).
- If the LLM returns an empty array, no facts stored; no error.
- Reuses the existing JSON-parse helper; no new format.

**Tests:**
```python
def test_batch_prompt_routes_facts_to_right_turn():
    # Mock LLM returns facts with explicit turn_index; verify
    # source_turn_id metadata matches the expected turn ids.

def test_batch_prompt_missing_turn_index_falls_back_to_last():
    # Mock LLM returns facts without turn_index; they land on last turn
    # + WARNING log captured.
```

**Step 5:** `git commit -m "feat(conv): batched extractor prompt with per-turn routing"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
SOMA_EMBED_MODEL=stub pytest tests/test_memory -q
```

Baseline post-Phase-22: 422 tests in `tests/test_memory`. Target +8–10
new Phase 25 tests, 0 regressions.
