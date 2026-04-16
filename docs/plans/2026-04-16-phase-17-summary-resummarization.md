# Phase 17: Summary Re-Summarization from Raw Turns

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Prevent compounding drift in `ConversationalMemory`'s rolling summaries. Today, every `summary_every` turns we summarize `previous_summary + recent_turns` → new summary. That chains — so after N summaries, hallucinations and omissions compound. Fix: every `M × summary_every` turns (default M=5), re-derive the summary from raw turns directly, bypassing the previous summary.

**Architecture:**
- Track `self._summaries_generated` count on `ConversationalMemory`.
- When `self._summaries_generated % resummarize_every == 0` (and `> 0`), instead of calling `SUMMARY_PROMPT(prev_summary, recent_turns)`, call a new `RESUMMARY_PROMPT(recent_turns_batch)` where the batch is the last `M × summary_every` raw turns.
- New constructor kwarg: `resummarize_every: int = 5` (meaning every 5th summary is a from-scratch pass). `resummarize_every = 0` disables.
- All storage / metadata stays the same; the summary still carries `metadata["type"] = "summary"`.

**Tech Stack:** no new deps; pure LLM-prompt change + state tracking.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/*.md`.

---

### Task 1: RESUMMARY_PROMPT + counter

**Files:**
- Modify: `src/soma/memory/conversational.py`
- Modify: `src/soma/memory/conversational_prompts.py` (add `RESUMMARY_PROMPT`)
- Create: `tests/test_memory/test_conversational_resummarization.py`

**Prompt** (paste into `conversational_prompts.py`):
```python
RESUMMARY_PROMPT = """Summarise the following conversation turns into \
a short, factual summary (3-5 sentences). Focus on stable information \
about the participants (names, locations, preferences, goals) and \
on decisions / commitments that were made. Ignore small-talk unless \
it reveals stable facts. Do not reference any prior summary — this \
summary is being re-derived from the raw turns below.

Turns:
{turns}

Summary:"""
```

**Constructor signature changes:**
```python
def __init__(
    self, *,
    memory: MemoryLayer,
    llm: LLMBackend,
    session_id: str | None = None,
    user_id: str | None = None,
    near_dup_threshold: float = 0.92,
    ambiguous_threshold: float = 0.75,
    summary_every: int = 20,
    resummarize_every: int = 5,          # NEW
    extract_assistant: bool = False,
    extractor_llm: LLMBackend | None = None,
) -> None:
```

**State:**
```python
self._summaries_generated: int = 0       # NEW, increments after every summary write
self._resummarize_every: int = int(resummarize_every)
```

**Step 1: Write failing tests.**
```python
def test_first_summary_uses_standard_prompt():
    # After first summary_every turns, we call SUMMARY_PROMPT
    # (not RESUMMARY_PROMPT).

def test_nth_summary_uses_resummary_prompt_at_boundary():
    # resummarize_every=3 → summaries 3, 6, 9 are re-derived (from-scratch)
    # summaries 1, 2, 4, 5, 7, 8 are chained (standard prompt)

def test_resummarization_disabled_by_zero():
    # resummarize_every=0 → always use standard SUMMARY_PROMPT.

def test_resummary_reads_raw_turns_only():
    # Capture the prompt passed to the LLM for a re-summary call;
    # assert no "previous_summary" / "earlier_summary" substring.

def test_resummarize_every_unset_backward_compat():
    # Default resummarize_every=5 works; old callers without the kwarg
    # continue to behave.
```

**Step 2: Fail.**

**Step 3: Implement.**
- `_maybe_roll_summary(self)` now checks `self._summaries_generated`.
- If `resummarize_every > 0 and self._summaries_generated > 0 and self._summaries_generated % resummarize_every == 0`, build the `RESUMMARY_PROMPT(turns=...)` with the last `resummarize_every × summary_every` turns. Otherwise fall through to `SUMMARY_PROMPT(...)` with previous summary + recent turns.
- Increment `self._summaries_generated` after the write.

**Step 4:** Tests pass.

**Step 5:** `git commit -m "feat(conv): summary re-summarization every M summaries"`

---

### Task 2: Docs touch-up (conversational section)

**Files:**
- Modify: `docs/cookbook.md` §18 (or wherever ConversationalMemory recipe lives) — add a paragraph about `resummarize_every` and when to tune it.

**Step 5:** `git commit -m "docs(conv): resummarize_every recipe"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
SOMA_EMBED_MODEL=stub pytest tests/test_memory -q
```

Baseline 388 tests; target +5 new Phase 17 tests, 0 regressions.

**Compatibility note:** the constructor gained a kwarg. All existing callers pass kwargs by name (`session_id=...`, `user_id=...`, etc.) so adding one before `extract_assistant` is safe. Double-check the benchmark harness adapter (`benchmarks/harness/adapters/soma.py`) — the `ConversationalSomaAdapter` forwards kwargs; verify it still works.
