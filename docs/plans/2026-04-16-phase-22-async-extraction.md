# Phase 22: Async Extraction Mode for `ConversationalMemory`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Add `extraction_mode="async"` so `add_message` returns immediately and LLM extraction/reconcile runs on a background thread. Unlocks low-latency chat paths where the caller doesn't need facts ready synchronously. Default stays `sync` (today's behaviour).

**Architecture:** Thread-local `concurrent.futures.ThreadPoolExecutor(max_workers=1)` per `ConversationalMemory` instance. `add_message` submits the extract+reconcile work as a future; `flush()` drains all pending futures. Raw-turn + summary writes stay synchronous (they're cheap, ordered-dependent). Only the LLM calls move off the hot path.

**Safety invariants:**
- MemoryLayer writes happen only after `future.result()` — the executor thread calls `self._memory.store(...)` directly; the MemoryLayer is thread-safe per Phase 1's WAL design.
- Fact ordering within a turn is preserved because each turn gets one future; but cross-turn ordering is not guaranteed if the executor processes them out of order. Pin `max_workers=1` to preserve monotonic extraction order.
- `clear_session()` calls `flush()` first so no late facts arrive after the wipe.
- `close()` cancels pending futures + shuts down the executor.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/cookbook.md`.

---

### Task 1: Executor wiring + `extraction_mode` kwarg

**Files:**
- Modify: `src/soma/memory/conversational.py`
- Create: `tests/test_memory/test_conversational_async_extraction.py`

**Constructor signature:**
```python
def __init__(
    self,
    *,
    memory: MemoryLayer,
    llm: LLMBackend,
    session_id: str | None = None,
    user_id: str | None = None,
    near_dup_threshold: float = 0.92,
    ambiguous_threshold: float = 0.75,
    summary_every: int = 20,
    resummarize_every: int = 5,
    extract_assistant: bool = False,
    extractor_llm: LLMBackend | None = None,
    extraction_mode: Literal["sync", "async"] = "sync",   # NEW
) -> None:
```

When `extraction_mode="async"`, instantiate `self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"soma-conv-{session_id}")`. Otherwise `self._executor = None` and the existing sync path runs unchanged.

**`add_message` flow change:**
```python
def add_message(self, role, text, *, metadata=None, user_id=_UNSET):
    # (existing: stamp user_id, stamp metadata, store raw turn synchronously)
    turn_id = self._memory.store(text, metadata=turn_metadata)
    # Schedule extract+reconcile
    if self._executor is None:
        self._run_extract_reconcile(role, text, turn_id, ...)  # sync, today
    else:
        fut = self._executor.submit(
            self._run_extract_reconcile, role, text, turn_id, ...
        )
        self._pending_futures.append(fut)
    # (existing: summary rollover check)
    return turn_id
```

The summary rollover fires synchronously as before because it reads from `self._memory` which has accumulated writes from whichever completed futures exist. Document that summaries may lag behind in async mode.

**New methods:**
```python
def flush(self, timeout: float | None = None) -> None:
    """Block until all pending extraction futures complete. No-op in sync mode."""
    if self._executor is None:
        return
    done, not_done = concurrent.futures.wait(
        self._pending_futures, timeout=timeout,
    )
    for fut in done:
        fut.result()  # re-raise any exceptions
    self._pending_futures = list(not_done)

def close(self) -> None:
    """Flush + shutdown the executor. Safe to call multiple times."""
    if self._executor is None:
        return
    self.flush()
    self._executor.shutdown(wait=True)
    self._executor = None
```

Make `ConversationalMemory` a context manager:
```python
def __enter__(self) -> "ConversationalMemory":
    return self

def __exit__(self, exc_type, exc, tb) -> None:
    self.close()
```

**`clear_session()` addition:**
```python
def clear_session(self, *, session_id: str | None = None) -> None:
    self.flush()  # drain so we don't wipe state that's about to be written
    # (existing logic)
```

**Step 1: Write failing tests.**
```python
def test_async_mode_returns_immediately(monkeypatch):
    # Stub LLM that blocks on time.sleep(0.1); add_message returns
    # in under 10 ms. facts_stored increments only after flush().

def test_async_mode_flush_drains_pending():
    # 5 add_messages in flight; flush() waits for all; after flush,
    # all facts are present in memory.

def test_async_mode_close_via_context_manager():
    with ConversationalMemory(mode="async", ...) as cm:
        cm.add_message("user", "hello")
    # exit blocks on flush; memory has the fact.

def test_async_mode_preserves_extraction_order_within_session():
    # Submit 3 add_messages; max_workers=1 means extract fires in
    # submission order. Stored facts land in-order.

def test_async_mode_clear_session_flushes_first():
    # Submit 3 add_messages; immediately call clear_session.
    # flush fires (facts land), then wipe removes them.

def test_async_mode_exceptions_surface_on_flush():
    # Stub LLM raises; flush() re-raises.

def test_sync_mode_default_unchanged():
    # No extraction_mode kwarg → executor is None, no threads spawned.

def test_double_close_is_safe():
    # Calling close() twice doesn't raise.
```

**Step 2-4: TDD.**

**Step 5:** `git commit -m "feat(conv): async extraction_mode with flush() drain"`

---

### Task 2: Docs + adapter compatibility check

**Files:**
- Modify: `docs/cookbook.md` §18 — brief note on when to prefer `extraction_mode="async"` (low-latency chat paths, not batch ingest).
- Verify: `benchmarks/harness/adapters/soma.py`'s `ConversationalSomaAdapter` still works. It forwards kwargs via `ConversationalMemory(...)` — the new kwarg defaults to "sync" so adapter behaviour is unchanged. No adapter edit needed.

**Step 5:** `git commit -m "docs(conv): extraction_mode async recipe"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory docs
SOMA_EMBED_MODEL=stub pytest tests/test_memory benchmarks/tests -q
```

Baseline post-Phase-17: 409 tests in memory + bench. Target +~8 new Phase 22 tests, 0 regressions.

**Threading gotchas to audit:**
- Python GIL means no CPU parallelism; the win is I/O overlap (network calls to LLM backends). Document this so users don't expect CPU speedup.
- Tests that monkeypatch module globals need care — the executor thread sees the monkeypatched state iff the patch is applied before `submit()`. Use `monkeypatch.context()` liberally.
- Don't use `asyncio` — we'd need an async `LLMBackend` protocol which is a much bigger refactor. Threads are the right choice for now.
