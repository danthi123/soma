# Phase 11: `extractor_llm=` kwarg on `ConversationalMemory`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Let users pin a stronger LLM for the structured-output steps (extract + reconcile) while keeping a smaller LLM for chat and summary. Critical for 3B-local users running tiny chat models that can't reliably produce strict JSON.

**Architecture:** Add `extractor_llm=` kwarg to `ConversationalMemory.__init__`. When set, route the extract + reconcile LLM calls through it; fall back to `self._llm` when unset (backward compat). Summary stays on `self._llm` because summary is free-form text and small LLMs handle it fine.

**Tech Stack:** no new deps; just an added kwarg that defaults to `None`.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/positioning.md`.

---

### Task 1: `extractor_llm=` kwarg + routing

**Files:**
- Modify: `src/soma/memory/conversational.py` (add kwarg, store `self._extractor_llm`, route extract + reconcile calls)
- Create: `tests/test_memory/test_conversational_extractor_llm.py`

**Step 1: Write failing tests.**
```python
def test_extractor_llm_used_for_extract_when_set(stub_chat, stub_extract):
    cm = ConversationalMemory(
        memory=mem, llm=stub_chat, extractor_llm=stub_extract,
        session_id="s1",
    )
    cm.add_message("user", "I live in Seattle")
    # The stub_extract recorder captured EXTRACT_PROMPT;
    # stub_chat recorder saw zero calls.
    assert any("EXTRACT" in p for p in stub_extract.prompts_seen)
    assert all("EXTRACT" not in p for p in stub_chat.prompts_seen)

def test_extractor_llm_used_for_reconcile_when_set(...):
    # similar, but for RECONCILE_PROMPT

def test_summary_still_uses_main_llm_even_with_extractor_set(...):
    # After 20 turns, summary prompt should go to stub_chat (not stub_extract)

def test_extractor_llm_defaults_to_none_and_uses_main_llm(...):
    # Unchanged behaviour — extractor_llm unset, all prompts on stub_chat.
```

**Step 2:** Tests fail (kwarg missing).

**Step 3: Implement.**
- Add `extractor_llm: LLMBackend | None = None` to `__init__` signature.
- Store `self._extractor_llm = extractor_llm or llm` — simplest routing, no null checks at call sites.
- Leave the summary generate call site on `self._llm`.
- For extract + reconcile call sites (lines 117, 261, 420 per `grep`), swap to `self._extractor_llm.generate(...)`.
- Docstring update in the class docstring + new `:param extractor_llm:` entry on `__init__`.

**Step 4:** Tests pass.

**Step 5: Commit.**
```bash
git commit -m "feat(conv): extractor_llm kwarg for structured-output steps"
```

---

### Task 2: Adapter + docs touch-up

**Files:**
- Modify: `benchmarks/harness/adapters/soma.py` (ConversationalSomaAdapter) — accept optional `extractor_llm` and forward to `ConversationalMemory`
- Modify: `docs/conversational-memory.md` (if it exists; otherwise skip)

**Step 1: Test.**
```python
def test_conv_adapter_forwards_extractor_llm():
    adapter = ConversationalSomaAdapter(
        llm=chat_backend, extractor_llm=extract_backend,
    )
    adapter.prepare()
    assert adapter._cm._extractor_llm is extract_backend
```

**Step 2-4:** TDD.

**Step 5: Commit.**
```bash
git commit -m "feat(bench): ConversationalSomaAdapter forwards extractor_llm"
```

---

### Final sanity

```bash
pytest tests/test_memory benchmarks/tests -q
ruff check src/soma/memory/conversational.py benchmarks/harness/adapters/soma.py tests/test_memory
```
