# Phase 34: `forget()` Inventory API (Dry-Run Only)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Foundation for the GDPR-grade forgetting track
(Phases 34–37). This phase ships the **read-only** inventory API:
given a match criterion, return what WOULD be deleted if the call
were committed. Actual deletion ships in Phase 35; cascading through
summaries in Phase 36; audit trail + REST in Phase 37.

**Architecture:**
- New `ConversationalMemory.forget(...)` that, with `dry_run=True`
  (the only mode shipped in this phase), returns a dataclass
  summarising the target set:
  ```python
  @dataclass
  class ForgetPreview:
      raw_turns: list[str]       # turn ids
      derived_facts: list[str]   # fact ids with source_turn_id in raw_turns
      summaries: list[str]       # summary ids that overlap raw_turns
      total_vectors: int
  ```
- Three matcher modes:
  1. `forget(text_matches=pattern)` — substring search over raw turn
     text. Case-insensitive by default; case-sensitive via
     `case_sensitive=True`.
  2. `forget(subject=name)` — match extracted-fact `subject` field.
  3. `forget(user_id=uid)` — all data for a given user (delegates to
     the multi-user scoping landed in Phase 12 but doesn't call
     `clear_session` yet — just lists what would go).
- Non-exclusive: a caller can combine (e.g. `text_matches=X,
  user_id=alice`) and the match is intersection.
- `dry_run` defaults to `True` in this phase. Phase 35 flips the
  default to `False` and wires the deletion path.

**Safety:**
- Read-only — no writes, no LLM calls, no vector-store mutations.
- Returns empty lists on no match (not an error).
- Raises `ValueError` if zero criteria are passed (don't accidentally
  "forget everything" by omission).

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/cookbook.md`, `docs/auth.md` (GDPR posture — Phase 37 will
drive the final write), `deferred-items.md` (strike through when the
full track lands in Phase 37).

---

### Task 1: `ForgetPreview` + matcher primitives

**Files:**
- Modify: `src/soma/memory/conversational.py`
- Create: `tests/test_memory/test_conversational_forget_preview.py`

**API:**
```python
@dataclass(frozen=True)
class ForgetPreview:
    raw_turns: list[str]
    derived_facts: list[str]
    summaries: list[str]
    total_vectors: int

    def is_empty(self) -> bool:
        return not (self.raw_turns or self.derived_facts or self.summaries)


class ConversationalMemory:
    def forget(
        self,
        *,
        text_matches: str | None = None,
        subject: str | None = None,
        user_id: str | None = None,
        case_sensitive: bool = False,
        dry_run: bool = True,   # Phase 35 flips default
    ) -> ForgetPreview:
        """Return what would be deleted under this match.

        dry_run=True in Phase 34 (the only shipped mode); dry_run=False
        raises NotImplementedError until Phase 35.
        """
```

**Internal helpers (pure functions, easy to unit-test):**
```python
def _raw_turns_matching(
    memory: MemoryLayer,
    *,
    text_matches: str | None,
    subject: str | None,
    user_id: str | None,
    case_sensitive: bool,
) -> list[str]: ...

def _facts_for_turns(memory: MemoryLayer, turn_ids: list[str]) -> list[str]: ...

def _summaries_overlapping_turns(
    memory: MemoryLayer, turn_ids: list[str]
) -> list[str]: ...
```

Match logic:
- `text_matches`: iterate all `role:turn` entries, test `in`
  (case-adjusted) on `.text`.
- `subject`: iterate all `role:fact` entries, test equality on
  `metadata["subject"]`.
- `user_id`: iterate all entries, test equality on
  `metadata["user_id"]`.
- Intersection when multiple criteria given.

**Step 1: Failing tests.**
```python
def test_forget_preview_text_match_returns_matching_turns():
    cm = _make_cm()
    cm.add_message("user", "I love gardening")
    cm.add_message("user", "Coffee is good")
    cm.add_message("user", "Gardening on weekends is my hobby")
    preview = cm.forget(text_matches="gardening")
    assert len(preview.raw_turns) == 2
    assert preview.derived_facts  # some facts extracted from those turns
    assert preview.is_empty() is False

def test_forget_preview_case_insensitive_by_default():
    cm = _make_cm()
    cm.add_message("user", "I LOVE Gardening")
    preview = cm.forget(text_matches="gardening")
    assert len(preview.raw_turns) == 1

def test_forget_preview_case_sensitive_optional():
    cm = _make_cm()
    cm.add_message("user", "I LOVE Gardening")
    preview = cm.forget(text_matches="gardening", case_sensitive=True)
    assert preview.is_empty()

def test_forget_preview_subject_match():
    cm = _make_cm()
    # Stub LLM returns fact: subject="alice", predicate="likes", ...
    cm.add_message("user", "Alice likes mushrooms", ...)
    preview = cm.forget(subject="alice")
    assert len(preview.derived_facts) >= 1

def test_forget_preview_user_id_match():
    cm = _make_cm(user_id="alice")
    cm.add_message("user", "hello", user_id="alice")
    cm.add_message("user", "hi", user_id="bob")
    preview = cm.forget(user_id="alice")
    assert all(tid in preview.raw_turns for tid in _turns_for("alice"))
    assert not any(tid in preview.raw_turns for tid in _turns_for("bob"))

def test_forget_preview_intersection_of_criteria():
    cm = _make_cm()
    cm.add_message("user", "gardening", user_id="alice")
    cm.add_message("user", "gardening", user_id="bob")
    preview = cm.forget(text_matches="gardening", user_id="alice")
    assert len(preview.raw_turns) == 1  # only alice's

def test_forget_preview_no_criteria_raises():
    with pytest.raises(ValueError, match="at least one criterion"):
        cm.forget()

def test_forget_preview_empty_result_returns_empty_preview():
    preview = cm.forget(text_matches="nonexistent")
    assert preview.is_empty()

def test_forget_dry_run_default_true_no_writes():
    baseline = cm.memory.ntotal
    cm.forget(text_matches="gardening")
    assert cm.memory.ntotal == baseline

def test_forget_dry_run_false_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Phase 35"):
        cm.forget(text_matches="x", dry_run=False)

def test_forget_preview_includes_overlapping_summaries():
    # Ingest 25 turns (forces a summary rollover at 20).
    # forget(text_matches=...) where the match is in turns 0-10 should
    # return summaries whose source_turn_ids intersect that range.
    ...

def test_forget_preview_counts_total_vectors():
    # raw_turns + derived_facts + summaries = total_vectors.
```

**Step 2-4: TDD.**

**Step 5:** `git commit -m "feat(conv): ConversationalMemory.forget() inventory API (dry-run)"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
SOMA_EMBED_MODEL=stub pytest tests/test_memory -q
```

Baseline post-Phase-29: 504 passed, 33 skipped in `tests/test_memory`.
Target Phase 34: +~12 new tests, 0 regressions.

**Gotchas:**
- The matcher iterates `memory._entries` (or whatever
  `ConversationalMemory` uses to enumerate raw turns). Make sure
  `user_id` filtering respects the Phase 12 multi-user scoping
  (don't leak another user's matches).
- Substring match can be surprising on short patterns (`"al"` matches
  "alice", "alan", "all"). Document this in the docstring — callers
  who need word-boundary matching should pre-tokenize.
- `summaries` detection: each summary's metadata carries the range
  of turn ids it summarised (see the summary-rollover code from
  Phase 17). `_summaries_overlapping_turns` reads that range and
  checks set intersection.
