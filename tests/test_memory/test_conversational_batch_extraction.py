"""Phase 25: batch extraction mode for :class:`ConversationalMemory`.

Covers the contracts pinned in
``docs/plans/2026-04-16-phase-25-batch-extraction.md``:

1. Batch mode accumulates up to ``batch_size`` turns before a single
   extract call fires.
2. :meth:`flush` drains a partial batch.
3. Context-manager ``__exit__`` flushes partial batches via :meth:`close`.
4. :meth:`clear_session` drops the pending buffer without extracting —
   "wipe means wipe" — and then wipes the session like usual.
5. Facts come back tagged with the right ``source_turn_id`` metadata.
6. Exceptions from the batched extract surface on the triggering
   :meth:`add_message` call (not on a prior accumulating call).
7. Sync and async modes ignore ``batch_size`` and keep their
   pre-Phase-25 semantics.
8. ``batch_size < 1`` in batch mode raises at construction time.

Task 2 adds two more tests on top:

9. A LLM reply with explicit per-fact ``turn_index`` routes facts to
   the matching raw-turn's ``source_turn_id`` even when the order is
   scrambled; the batched prompt mentions each numbered turn.
10. A LLM reply missing ``turn_index`` falls back to the last turn in
    the batch (conservative fallback, logged at WARNING).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pytest
import torch

from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic 8-d embedding keyed on the raw text."""
    h = hash(text)
    return torch.tensor(
        [(h >> i) & 0xF for i in range(0, 32, 4)], dtype=torch.float32
    )


@dataclass
class _BatchBackend:
    """Scripted backend that records every extract / reconcile call.

    The batched extractor prompt lists N turns; this backend returns a
    canned reply per extract call. Reconcile always ADDs so the facts
    hit storage.
    """

    extract_replies: list[str] = field(default_factory=list)
    name: str = "batch-stub"
    _extract_calls: int = 0
    _reconcile_calls: int = 0
    extract_prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            i = self._extract_calls
            self._extract_calls += 1
            self.extract_prompts.append(prompt)
            if i < len(self.extract_replies):
                return self.extract_replies[i]
            return "[]"
        if "You reconcile a new fact" in prompt:
            self._reconcile_calls += 1
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        return "{}"


@dataclass
class _RaisingExtractBackend:
    """Backend whose extract call raises. Used to verify exceptions
    surface on the add_message call that triggered the flush, not on
    earlier accumulating calls.
    """

    name: str = "raising-batch"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            raise RuntimeError("boom from batched extractor")
        return "{}"


def _fact_reply_with_indices(
    entries: list[tuple[int, str]],
) -> str:
    """Build an extract reply with explicit ``turn_index`` per fact."""
    return json.dumps(
        [
            {"category": "other", "text": text, "turn_index": idx}
            for idx, text in entries
        ]
    )


def _fact_reply_no_indices(texts: list[str]) -> str:
    """Build an extract reply missing the ``turn_index`` field."""
    return json.dumps(
        [{"category": "other", "text": t} for t in texts]
    )


# ----------------------------------------------------------------------
# Task 1: buffering + flush + close + clear_session + exception surfacing
# ----------------------------------------------------------------------


def test_batch_mode_accumulates_until_K() -> None:
    """7 turns under batch_size=8 fires no extract; the 8th triggers one call."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[
            _fact_reply_with_indices(
                [(i, f"fact-{i}") for i in range(8)]
            )
        ]
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=8,
    )
    try:
        for i in range(7):
            cm.add_message("user", f"turn {i}")
        # No extract call yet — still accumulating.
        assert llm._extract_calls == 0
        assert len(cm._pending_batch) == 7
        cm.add_message("user", "turn 7")
        # 8th turn tips us over batch_size; exactly one batched extract call.
        assert llm._extract_calls == 1
        assert cm._pending_batch == []
        fact_texts = {f.text for f in cm.list_facts()}
        assert fact_texts == {f"fact-{i}" for i in range(8)}
    finally:
        cm.close()


def test_batch_mode_flush_drains_partial() -> None:
    """flush() on a partial batch fires one extract call on the partial set."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[
            _fact_reply_with_indices([(i, f"fact-{i}") for i in range(3)])
        ]
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=8,
    )
    try:
        for i in range(3):
            cm.add_message("user", f"turn {i}")
        assert llm._extract_calls == 0
        cm.flush()
        assert llm._extract_calls == 1
        assert cm._pending_batch == []
        fact_texts = {f.text for f in cm.list_facts()}
        assert fact_texts == {f"fact-{i}" for i in range(3)}
    finally:
        cm.close()


def test_batch_mode_close_flushes_partial() -> None:
    """Context-manager __exit__ via close() drains pending batch."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[
            _fact_reply_with_indices([(0, "fact-a"), (1, "fact-b")])
        ]
    )
    with ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=8,
    ) as cm:
        cm.add_message("user", "a")
        cm.add_message("user", "b")
        # Inside the block — still pending.
        assert llm._extract_calls == 0
    # After __exit__, one batched extract call fired and facts landed.
    assert llm._extract_calls == 1
    fresh = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
    )
    fact_texts = {f.text for f in fresh.list_facts()}
    assert fact_texts == {"fact-a", "fact-b"}


def test_batch_mode_clear_session_drops_pending_without_extracting() -> None:
    """clear_session in batch mode drops pending buffer; no extract fires."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[
            _fact_reply_with_indices([(i, f"fact-{i}") for i in range(3)])
        ]
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=8,
    )
    try:
        for i in range(3):
            cm.add_message("user", f"turn {i}")
        assert len(cm._pending_batch) == 3
        removed = cm.clear_session()
        # No extract call — we don't store facts the user is wiping.
        assert llm._extract_calls == 0
        # Pending buffer empty; session wipe ran on whatever turns/facts
        # already existed (just the 3 raw turns here since no facts
        # ever made it in).
        assert cm._pending_batch == []
        assert removed >= 3  # at least the 3 raw turns we stored
        assert cm.list_facts() == []
    finally:
        cm.close()


def test_batch_mode_preserves_turn_order_in_facts() -> None:
    """Facts land with source_turn_id matching the originating turn's id."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[
            _fact_reply_with_indices(
                [(0, "fact-alpha"), (1, "fact-beta"), (2, "fact-gamma")]
            )
        ]
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=3,
    )
    try:
        cm.add_message("user", "turn-alpha text")
        cm.add_message("user", "turn-beta text")
        cm.add_message("user", "turn-gamma text")
        assert llm._extract_calls == 1
        facts = cm.list_facts()
        by_text = {f.text: f for f in facts}
        assert set(by_text.keys()) == {"fact-alpha", "fact-beta", "fact-gamma"}
        # Each fact's source_turn_id points at its originating raw turn.
        turn_entries = [
            m for m in (mem.get(nid) for nid in mem._ids)
            if m is not None and m.metadata.get("type") == "turn"
        ]
        turn_id_by_text = {t.text: t.node_id for t in turn_entries}
        assert by_text["fact-alpha"].metadata["source_turn_id"] == (
            turn_id_by_text["turn-alpha text"]
        )
        assert by_text["fact-beta"].metadata["source_turn_id"] == (
            turn_id_by_text["turn-beta text"]
        )
        assert by_text["fact-gamma"].metadata["source_turn_id"] == (
            turn_id_by_text["turn-gamma text"]
        )
    finally:
        cm.close()


def test_batch_mode_exceptions_surface_on_trigger_call() -> None:
    """Exception from batched extract surfaces on the call that triggered it,
    not on earlier accumulating calls."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _RaisingExtractBackend()
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=3,
    )
    try:
        # First two calls accumulate silently; neither hits the extractor.
        cm.add_message("user", "a")
        cm.add_message("user", "b")
        # Third call triggers the flush -> extractor raises.
        with pytest.raises(RuntimeError, match="boom from batched extractor"):
            cm.add_message("user", "c")
    finally:
        # Pending buffer was cleared before the failing call (we don't
        # want to replay poison on every future flush), so close() is
        # clean here.
        cm.close()


def test_sync_and_async_modes_unchanged() -> None:
    """batch_size has no effect on sync mode. Sync fires an extract per turn."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[
            json.dumps([{"category": "other", "text": f"fact-{i}"}])
            for i in range(3)
        ]
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        # Default sync mode; batch_size is accepted but unused.
        batch_size=8,
    )
    for i in range(3):
        cm.add_message("user", f"turn {i}")
    # Sync: one extract call per turn, regardless of batch_size.
    assert llm._extract_calls == 3
    # No batch buffer state tracked in sync mode.
    assert cm._pending_batch == []
    fact_texts = {f.text for f in cm.list_facts()}
    assert fact_texts == {f"fact-{i}" for i in range(3)}
    cm.close()


def test_batch_size_validation() -> None:
    """batch_size < 1 in batch mode is a construction-time ValueError."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend()
    with pytest.raises(ValueError, match="batch_size"):
        ConversationalMemory(
            memory=mem, llm=llm, session_id="s1",
            extraction_mode="batch", batch_size=0,
        )


# ----------------------------------------------------------------------
# Task 2: batched extractor prompt — per-turn routing + fallback
# ----------------------------------------------------------------------


def test_batch_prompt_routes_facts_to_right_turn() -> None:
    """Explicit turn_index per fact routes source_turn_id correctly.

    Scramble the order: fact index 2 comes before fact index 0 in the
    reply. The router must still attribute each to the matching turn id.
    The batched prompt must also mention each of the N turns with its
    role label so the model knows how to index them.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[
            json.dumps(
                [
                    {"category": "other", "text": "fact-for-turn-2", "turn_index": 2},
                    {"category": "other", "text": "fact-for-turn-0", "turn_index": 0},
                    {"category": "other", "text": "fact-for-turn-1", "turn_index": 1},
                ]
            )
        ]
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=3,
    )
    try:
        cm.add_message("user", "turn zero text")
        cm.add_message("user", "turn one text")
        cm.add_message("user", "turn two text")
        assert llm._extract_calls == 1
        # The batched prompt must enumerate each turn with its role label
        # so the model can bind facts to a specific turn_index.
        prompt = llm.extract_prompts[0]
        assert "Turn 0 (user): turn zero text" in prompt
        assert "Turn 1 (user): turn one text" in prompt
        assert "Turn 2 (user): turn two text" in prompt
        # Routing: each fact's source_turn_id is the originating raw-turn id.
        turn_id_by_text = {
            mem.get(nid).text: nid
            for nid in mem._ids
            if mem.get(nid) is not None
            and mem.get(nid).metadata.get("type") == "turn"
        }
        facts_by_text = {f.text: f for f in cm.list_facts()}
        assert facts_by_text["fact-for-turn-0"].metadata["source_turn_id"] == (
            turn_id_by_text["turn zero text"]
        )
        assert facts_by_text["fact-for-turn-1"].metadata["source_turn_id"] == (
            turn_id_by_text["turn one text"]
        )
        assert facts_by_text["fact-for-turn-2"].metadata["source_turn_id"] == (
            turn_id_by_text["turn two text"]
        )
    finally:
        cm.close()


def test_batch_prompt_missing_turn_index_falls_back_to_last(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Facts without turn_index attribute to the last turn in the batch.

    Conservative: keep the fact rather than drop it, but log loudly
    so operators can spot a model that's systematically dropping the
    routing field.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _BatchBackend(
        extract_replies=[_fact_reply_no_indices(["orphan-fact"])]
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="batch",
        batch_size=3,
    )
    try:
        with caplog.at_level(logging.WARNING, logger="soma.memory"):
            cm.add_message("user", "turn zero text")
            cm.add_message("user", "turn one text")
            cm.add_message("user", "turn two text")
        assert llm._extract_calls == 1
        turn_id_by_text = {
            mem.get(nid).text: nid
            for nid in mem._ids
            if mem.get(nid) is not None
            and mem.get(nid).metadata.get("type") == "turn"
        }
        facts_by_text = {f.text: f for f in cm.list_facts()}
        assert "orphan-fact" in facts_by_text, (
            "fact without turn_index must be kept, attributed to last turn"
        )
        assert facts_by_text["orphan-fact"].metadata["source_turn_id"] == (
            turn_id_by_text["turn two text"]
        )
        # WARNING about the missing turn_index.
        assert any(
            "turn_index" in r.getMessage() for r in caplog.records
        ), [r.getMessage() for r in caplog.records]
    finally:
        cm.close()
