# Phase 36: `forget()` Cascade to Summaries

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Finish the cascade deletion started in Phase 35 by handling
summaries. A summary that covers a range of turns now being deleted
would otherwise still reference the forgotten content (summary text
mentions the scrubbed subject, etc.). Phase 36 either regenerates the
affected summary from the remaining raw turns, or drops it entirely
when no remaining turns cover the range.

**Architecture:**
- In `forget(..., dry_run=False)`, after facts + turns are deleted
  (Phase 35), iterate the summaries in `preview.summaries`:
  * If the summary's turn range has **remaining** raw turns (i.e.
    the overlap is partial), call the LLM to regenerate the summary
    from the surviving turns. Replace the summary in-place
    (`summary_id` stays stable; only `text` + `metadata.summary_text`
    change). Update the vector embedding.
  * If the summary's turn range is **fully** inside the deleted set
    (no survivors), delete the summary outright.
- **LLM unavailable fallback**: if the regeneration call fails
  (network, quota, dry-run extractor), delete the summary outright
  and log a WARNING. Principle: under user request to forget, prefer
  over-deletion to silent retention of derived content.
- `ForgetResult.deleted_summaries` now populated: ids of summaries
  that were dropped. Regenerated summaries are NOT listed in
  `deleted_summaries` (they're mutated, not deleted); track them
  separately via new `regenerated_summaries` field.

**Architecture extension:**
```python
@dataclass(frozen=True)
class ForgetResult:
    deleted_turns: list[str]
    deleted_facts: list[str]
    deleted_summaries: list[str]
    regenerated_summaries: list[str]   # NEW
    total_deleted: int
```

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/cookbook.md`, `deferred-items.md`.

---

### Task 1: Summary cascade

**Files:**
- Modify: `src/soma/memory/conversational.py` — extend `forget` to
  iterate `preview.summaries`, decide regenerate-vs-drop per summary,
  execute, and populate the new `regenerated_summaries` field.
- Extend: existing `tests/test_memory/test_conversational_forget_delete.py`
  (or whatever Phase 35 landed as) with summary-cascade cases.

**Internal helper:**
```python
def _cascade_summary(
    self,
    summary_id: str,
    surviving_turn_ids: list[str],
) -> Literal["regenerated", "deleted"]:
    if not surviving_turn_ids:
        self._memory.remove_ids([summary_id])
        return "deleted"
    # Regenerate
    surviving_text = self._load_turns_text(surviving_turn_ids)
    try:
        new_summary_text = self._llm.generate(
            SUMMARY_PROMPT.format(turns=surviving_text),
            max_tokens=256,
        )
    except Exception as exc:
        log.warning(
            "forget: summary %s regen failed (%s); dropping",
            summary_id, exc,
        )
        self._memory.remove_ids([summary_id])
        return "deleted"
    self._memory.update_text(summary_id, new_summary_text)
    return "regenerated"
```

**Step 1: Failing tests.**
```python
def test_forget_drops_fully_covered_summary():
    # Ingest 25 turns all matching "gardening" — rollover at 20 means
    # a summary exists covering turns 0-19. forget(text_matches="gardening")
    # should drop that summary entirely (no survivors in its range).
    ...
    result = cm.forget(text_matches="gardening")
    assert len(result.deleted_summaries) >= 1
    assert result.regenerated_summaries == []

def test_forget_regenerates_partially_covered_summary():
    # Ingest 25 turns, only turns 0-5 match "gardening"; the summary
    # at 0-19 has 14 survivors (6-19).
    ...
    result = cm.forget(text_matches="gardening")
    assert result.regenerated_summaries
    # The summary text should no longer mention "gardening"
    regenerated_id = result.regenerated_summaries[0]
    node = cm.memory._id_to_node[regenerated_id]
    assert "gardening" not in node.text.lower()

def test_forget_summary_regen_failure_falls_back_to_delete(monkeypatch):
    # Stub LLM.generate to raise; assert the summary is dropped.
    ...
    assert result.deleted_summaries
    assert regenerated_id not in [n.id for n in cm.memory]

def test_forget_summary_cascade_updates_total_deleted():
    result = cm.forget(text_matches="x")
    assert result.total_deleted == (
        len(result.deleted_turns)
        + len(result.deleted_facts)
        + len(result.deleted_summaries)
    )

def test_forget_preview_doesnt_modify_summaries():
    preview = cm.forget(text_matches="x", dry_run=True)
    # Preview lists affected summaries but doesn't touch them.
    for sid in preview.summaries:
        assert sid in cm.memory._id_to_node

def test_forget_summary_vector_updated_after_regen():
    # After regeneration, the summary's vector embedding reflects the
    # new text (not the old). Check by retrieving against a query that
    # matches the new text.
```

**Step 2-4: TDD.**

**Step 5:** `git commit -m "feat(conv): forget() cascade regenerates or drops summaries"`

---

### Final sanity

```bash
ruff check src/soma/memory tests/test_memory
SOMA_EMBED_MODEL=stub pytest tests/test_memory -q
```

Baseline post-Phase-35: ~+10 new tests. Target Phase 36: +~8 new, 0
regressions.

**Gotchas:**
- Regenerating a summary means re-embedding. The embedder is already
  wired into `ConversationalMemory` via `self._memory`; call
  `memory.update_text(id, new_text)` which handles re-embedding if
  that helper exists. If not, add it — in-place update is a general
  utility.
- Summary metadata's turn-range field name varies — check the Phase
  17 re-summarization code (`resummarize_every`) for the canonical
  field name. Reuse it.
- Don't regenerate if the survivors span a different range than the
  original summary — e.g. if turns 0-19 are covered by summary A and
  we delete turns 0-15, the surviving 16-19 is a sub-range. Regen is
  still correct; you're making a shorter summary from 4 turns, which
  is fine. If the regen prompt needs context about the original
  range, include it.
- Race with concurrent summary rollover: if the user calls `forget`
  at the same moment a new summary is being written, the extraction
  thread might still have a reference. The `flush()` in Phase 35
  already handles async mode — make sure the sync summary path is
  also atomic (probably already is; just worth confirming).
