# Phase 12: Multi-user scoping in `ConversationalMemory`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Support multiple users sharing the same bundle by scoping ConversationalMemory state (turns, facts, summaries) by `user_id` in addition to `session_id`. Unblocks multi-tenant Conversational deploys where each end-user's memory must be isolated but the bundle is shared (e.g. multi-user chat app on one server).

**Architecture:** Add `user_id=` kwarg to `ConversationalMemory.__init__` AND `add_message()`. When set, every stored turn / fact / summary gets `metadata["user_id"]`. `retrieve()` filters to the user's scope by default. Non-breaking: callers that don't pass `user_id` see today's behaviour.

REST surface: `POST /store`, `/retrieve`, etc. already accept arbitrary metadata; no new endpoint needed. Document the pattern.

Optional JWT claim: tokens with a `user_id` claim auto-scope their ConversationalMemory if the server routes it — but that ergonomic layer is Phase-13 territory. Phase 12 is the core `user_id` plumbing.

**Tech Stack:** metadata-only change; no new deps.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`. Also skipping REST/JWT auto-routing — just the core API change.

---

### Task 1: `user_id` threaded through ConversationalMemory

**Files:**
- Modify: `src/soma/memory/conversational.py`
- Create: `tests/test_memory/test_conversational_multi_user.py`

**API contract:**
```python
cm = ConversationalMemory(
    memory=mem, llm=llm,
    session_id="chat-1",
    user_id="alice",       # NEW — optional; stored in every write
)

# add_message can override per-call
cm.add_message("user", "my dog's name is Rex", user_id="alice")

# retrieve scopes to the user by default when constructed with user_id
hits = cm.retrieve("who is Rex")  # filters to user_id="alice"

# pass user_id=None explicitly to unscope (admin drill-down)
all_hits = cm.retrieve("who is Rex", user_id=None)
```

**Step 1: Write failing tests.**
- `test_user_id_stored_on_every_write` — turn, fact, summary metadata each have `user_id`.
- `test_retrieve_default_scopes_to_user_id` — constructed with `user_id="a"`, store two turns as "a" and two as "b"; retrieve returns only "a".
- `test_retrieve_user_id_none_sees_all` — explicit override.
- `test_per_call_user_id_overrides_constructor` — `add_message(..., user_id="b")` while constructor said "a" stores under "b".
- `test_user_id_unset_preserves_pre_phase12_behaviour` — no `user_id` kwarg anywhere → no `user_id` key in metadata (backward compat).
- `test_supersede_respects_user_id` — can't supersede another user's fact.
- `test_clear_session_filtered_by_user_id_when_set` — clears only this user's state.

**Step 2-4: TDD.**

Implementation notes for the subagent:
- Store `self._user_id` on the instance; every write threads it through `metadata`.
- `retrieve()` with user_id scoping uses the existing `where=` filter mechanism. Compose with any user-supplied `where` via AND.
- `supersede()` must verify the target fact's `user_id` matches before mutating — else raise `PermissionError`.

**Step 5: Commit.**
```bash
git commit -m "feat(conv): multi-user scoping via user_id"
```

---

### Task 2: REST pattern doc

**Files:**
- Modify: `docs/conversational-memory.md` (if exists) or create the section in `docs/quickstart.md`

Document the pattern: callers pass `user_id` in request body's `metadata` field. The server doesn't need new endpoints; clients handle the scoping via existing metadata.

No new tests. This is documentation only.

**Step 5: Commit.**
```bash
git commit -m "docs(conv): multi-user scoping pattern"
```

---

### Final sanity

```bash
pytest tests/test_memory -q
ruff check src/soma/memory/conversational.py tests/test_memory
```

**Known risk:** if Phase 11's `extractor_llm` kwarg lands between your dispatch and your PR, the tests need to cope with the extra `__init__` param. Check `git log --oneline` for commits touching `conversational.py` before starting — rebase your mental model on current head.
