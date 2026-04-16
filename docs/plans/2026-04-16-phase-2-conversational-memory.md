# Phase 2 — ConversationalMemory wrapper

> **For Claude:** Execute via TDD after Phase 1 (WAL) lands. Each task has failing tests first, then minimal impl, then commit.

**Goal:** A `ConversationalMemory(memory, llm, session_id)` wrapper that brings Mem0/Zep-style fact extraction, reconciliation, and rolling session summaries to SOMA. Opt-in sugar over `MemoryLayer` — raw store/retrieve API unchanged.

**Architecture:** Two-phase pipeline per user turn, mirroring Mem0's proven approach (arXiv 2504.19413; +26% LoCoMo QA accuracy vs raw RAG at 91% lower latency):

1. **Extract** atomic facts from the message via LLM prompt (closed-vocab category, JSON-only output, empty-list example).
2. **Reconcile** each fact against top-k existing memories — skip (≥0.92 cosine), ADD (<0.75), or LLM-decides ADD/UPDATE/SUPERSEDE/NOOP (0.75-0.92 ambiguous).

Borrow Zep's "invalidate, don't delete" via metadata `superseded_by` pointer — preserves history for audit and keeps Stage 3 plasticity able to see invalidated nodes. Roll session summaries every 20 turns. Raw turns also stored (`type: "turn"`) so LoCoMo eval stays compatible.

**Tech stack:** Existing `soma.llm.LLMBackend` (Ollama/OpenAI/Anthropic/HF/OpenAICompat already shipped), existing `MemoryLayer`. No new deps.

---

### Task 1: Prompt templates module

**Files:**
- Create: `src/soma/memory/conversational_prompts.py`
- Test: `tests/test_memory/test_conversational_prompts.py`

**Step 1:** failing tests:
- `test_extract_prompt_contains_empty_list_example` — small models need this anchor.
- `test_extract_prompt_is_model_agnostic_length` — under 1500 chars.
- `test_reconcile_prompt_lists_all_four_ops` — ADD/UPDATE/SUPERSEDE/NOOP all mentioned.
- `test_summary_prompt_preserves_entities_clause` — explicit "do not invent details".

**Step 2:** implement three string templates: `EXTRACT_PROMPT`, `RECONCILE_PROMPT`, `SUMMARY_PROMPT`. Exact wording from research report §2, §4, §5. Use `.format(message=...)` for interpolation — no jinja2.

**Step 3:** commit `feat(memory): conversational prompt templates`.

### Task 2: Fact extraction with parse-failure safety

**Files:**
- Create: `src/soma/memory/conversational.py` (initial skeleton)
- Test: `tests/test_memory/test_conversational_extract.py`

**Step 1:** failing tests (use `CapturingBackend` pattern from `tests/test_llm/test_rag.py`):
- `test_extracts_two_facts_from_compound_message` — "I'm Alex and live in Boston" → 2 facts.
- `test_extracts_zero_facts_from_greeting` — "Hey" → empty list.
- `test_json_parse_failure_returns_empty_list_and_logs` — LLM returns prose → no crash, log at WARNING.
- `test_category_outside_closed_vocab_is_mapped_to_other` — LLM returns "car_preference" → normalized to "other".
- `test_extraction_never_stores_assistant_utterances` — role="assistant" skipped by default (config gate).

**Step 2:** implement `_extract_facts(message: str) -> list[ExtractedFact]` using the LLM prompt. Wrap parse in try/except `json.JSONDecodeError` + `KeyError`; on failure log and return []. Normalize unknown categories to "other".

**Step 3:** commit `feat(memory): LLM-driven atomic fact extraction with safe parse`.

### Task 3: Reconcile with threshold short-circuit + LLM call

**Files:**
- Modify: `src/soma/memory/conversational.py`
- Test: `tests/test_memory/test_conversational_reconcile.py`

**Step 1:** failing tests:
- `test_near_dup_skips_without_llm_call` — retrieved top-1 has score ≥0.92; LLM never called.
- `test_low_similarity_adds_unconditionally` — max score <0.75; ADD without LLM.
- `test_ambiguous_range_calls_llm` — max score 0.8; LLM consulted once.
- `test_supersede_writes_metadata_pointer_not_forget` — old entry stays, `superseded_by` metadata set on it, `supersedes` on new.
- `test_update_replaces_text_but_keeps_history` — UPDATE op forgets old, stores new with `supersedes` pointer.
- `test_noop_stores_nothing` — LLM returns NOOP; memory unchanged.

**Step 2:** implement `_reconcile(fact: ExtractedFact) -> str | None`. Steps:
1. `top_candidates = self.memory.retrieve(fact.text, k=5)`.
2. If none OR max score < 0.75: `return self._add_fact(fact.text)`.
3. If max score ≥ 0.92: return None (skip).
4. Call LLM with `RECONCILE_PROMPT.format(new_fact=..., candidates=...)`, parse JSON, dispatch on op:
   - ADD: store new
   - UPDATE target_id: forget target, store new with `metadata={"supersedes": target_id}`.
   - SUPERSEDE target_id: store new with `supersedes`, then update target's metadata with `superseded_by=new_id`, `superseded_at_step=self.memory._step`. (Uses a new `MemoryLayer.update_metadata(node_id, patch)` helper; add that first if missing.)
   - NOOP: return None.

**Step 3:** commit `feat(memory): reconcile with threshold short-circuit + LLM ADD/UPDATE/SUPERSEDE/NOOP`.

### Task 4: MemoryLayer.update_metadata helper

**Files:**
- Modify: `src/soma/memory/api.py` — add `update_metadata(node_id, patch)`
- Test: `tests/test_memory/test_api.py` — extend

**Step 1:** failing tests:
- `test_update_metadata_merges_into_existing` — original keys preserved, patch keys overwrite.
- `test_update_metadata_raises_on_unknown_id` — KeyError with node_id.
- `test_update_metadata_writes_wal_record` — WAL append on metadata-only mutation (Phase 1 semantics).

**Step 2:** add `def update_metadata(self, node_id: str, patch: dict[str, Any]) -> None` that acquires the bundle lock (via WAL layer), appends a `{"op":"update_metadata","node_id":nid,"patch":{...}}` record, merges patch into `self._metadatas[idx]`.

**Step 3:** extend WAL replay to handle the new op type. Commit `feat(memory): update_metadata with WAL support`.

### Task 5: ConversationalMemory full API + session summaries

**Files:**
- Modify: `src/soma/memory/conversational.py`
- Test: `tests/test_memory/test_conversational_api.py`

**Step 1:** failing tests:
- `test_add_user_message_extracts_and_reconciles` — one user turn, facts stored, raw turn also stored (`type=turn`).
- `test_add_assistant_message_skipped_by_default` — no facts extracted from assistant.
- `test_summary_rolls_every_n_turns` — summary_every=5; at turn 5 a summary stored with `type=summary`.
- `test_retrieve_returns_facts_and_summaries_and_turns_together` — all types compete on cosine.
- `test_retrieve_excludes_superseded_by_default` — old facts marked superseded don't appear.
- `test_retrieve_include_superseded_true_returns_history` — explicit opt-in.
- `test_clear_session_keeps_summaries_by_default` — `clear_session()` removes turns+facts but preserves summaries.
- `test_list_facts_filters_by_session_id` — multi-session isolation.
- `test_get_summary_returns_most_recent` — ordered by step.
- `test_flush_is_noop_in_sync_mode` — doesn't raise.

**Step 2:** implement the class per research §6 API sketch. Session turn counter increments on every `add_message`; modulo `summary_every` triggers `_roll_summary()` which grabs last N raw turns by metadata filter, calls LLM, stores the summary. All entries carry `{session_id, type, ...}` metadata.

**Step 3:** commit `feat(memory): ConversationalMemory with add_message/retrieve/session summaries/superseded filtering`.

### Task 6: Integration smoke test (real MemoryLayer + DryRunBackend)

**Files:**
- Test: `tests/test_memory/test_conversational_smoke.py`

**Step 1:** failing test: the "Alex moved" scenario end-to-end with a stub LLM that always returns a crafted JSON for extraction/reconcile:
- Turn 1: "I'm Alex and live in Portland" → 2 facts (name, location=Portland).
- Turn 2: "I moved to Boston" → 1 fact "lives in Boston"; reconcile SUPERSEDEs Portland fact.
- Retrieve "where does the user live" → Boston (Portland excluded by default filter).
- `include_superseded=True` → both visible.
- `list_facts()` → 3 entries (name, old location marked superseded, new location).

**Step 2:** use `ExtractScriptedBackend(LLMBackend)` — scripted responses per call count. Commit `test(memory): conversational end-to-end scenario`.

### Task 7: LoCoMo benchmark adapter

**Files:**
- Modify: `benchmarks/run_locomo.py` — new `--conversational` flag
- Modify: `benchmarks/harness/adapters/soma.py` — add `ConversationalSomaAdapter` subclass
- Create: `benchmarks/reports/locomo_conversational.md` (generated)

**Step 1:** failing test (out-of-bench): `benchmarks/tests/test_run_locomo_conversational.py` — adapter can run on a 3-turn fixture with a scripted LLM.

**Step 2:** the adapter wraps `SomaAdapter` and replaces `store(text)` with `cm.add_message(role, text)`. `retrieve(query, k)` becomes `cm.retrieve(query, k)`. Add a "facts stored / turns processed" column to the report.

**Step 3:** run it on a real LLM (Ollama llama3.2) or against a dry-run adapter if Ollama isn't available. Document expected behavior. Commit `bench(locomo): --conversational mode with fact-extraction adapter`.

### Task 8: Docs + cookbook recipe

**Files:**
- Modify: `docs/cookbook.md` — add "Conversational memory" recipe
- Modify: `docs/comparison.md` — add Mem0/Zep parity row
- Modify: `CHANGELOG.md`

**Step 1:** cookbook recipe:
```python
from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory
from soma.llm import backend_from_env

mem = MemoryLayer.with_sbert(bundle_path="brain/")
llm = backend_from_env()
cm = ConversationalMemory(memory=mem, llm=llm, session_id="alex")

cm.add_message("user", "I just moved to Boston")
# → facts extracted, reconciled, summaries rolled every 20 turns

for hit in cm.retrieve("where does the user live?"):
    print(hit.text)  # "User lives in Boston" (Portland entry superseded)
```

**Step 2:** commit `docs: Phase 2 — ConversationalMemory recipe + Mem0/Zep parity in comparison`.

---

## Modes & Phases

**This plan covers Stage 1 (sync, default).** Two follow-ups tracked in TODO:
- **Stage 2: async mode.** `extraction_mode="async"` → `concurrent.futures.ThreadPoolExecutor`; `flush()` drains. Est. 1 day.
- **Stage 3: batch mode.** Accumulate K turns, extract in one LLM call. Est. 1 day.

## Risks

1. **Small-model extraction quality.** 3B Ollama may hallucinate categories or miss facts. Mitigation: test matrix (llama3.2-3b, qwen2.5-7b, gpt-4o-mini) + `extractor_llm=` kwarg for pinning a stronger model.
2. **Reconcile drift.** LLM over-merges or under-merges. Mitigation: log every op to `metadata["reconcile_reasoning"]` for audit; add regression suite from real cases.
3. **Threshold sensitivity.** 0.75/0.92 are sbert rule-of-thumb. Plan: calibration sweep during LoCoMo eval, document the curve.
4. **Summary drift.** Rolling summaries compound errors. Consider periodic re-summarization from raw turns every M × summary_every.
5. **Hook quirk.** Project's file-write hook rejects a specific 5-char substring (see `reference_eval_hook_quirk.md`); verify prompt strings don't trip it when writing source files.
6. **Privacy / GDPR-grade forgetting.** `clear_session` exists but scrubbing all derived facts+summaries referencing a piece of info is out-of-scope. Track as follow-up.

## Open questions

- **Retrieval ranking of summaries vs facts vs turns:** pure cosine by default. Optional `+0.05` boost for summaries on open-ended queries? Stage 2.5 question.
- **Multi-user scope:** session_id is single-user surface. Add `user_id` claim to metadata later without breaking API.
- **Extraction trigger:** every user turn, or every 2nd, or when message length exceeds N chars? Start with every user turn; add throttle later if needed.

## Related plans

- Phase 1 — `docs/plans/2026-04-16-phase-1-wal-autosave.md` (WAL lands first; Task 4 here depends on WAL's `update_metadata` op type)
- Phase 6 — `docs/plans/2026-04-16-phase-6-vector-backend.md` (backend-agnostic; ConversationalMemory wraps MemoryLayer regardless of backend)
