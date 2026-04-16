"""Phase 22: async extraction mode for :class:`ConversationalMemory`.

Covers the 8 contracts pinned in
``docs/plans/2026-04-16-phase-22-async-extraction.md``:

1. Async ``add_message`` returns immediately; facts only land after
   :meth:`flush`.
2. :meth:`flush` drains every pending future.
3. The context-manager ``__exit__`` path drains + shuts down the
   executor (``close`` is idempotent).
4. ``max_workers=1`` preserves extraction order within a session.
5. :meth:`clear_session` flushes first so pending facts land + are
   cleaned, never leak in after the wipe.
6. Exceptions from the executor thread surface on the next
   :meth:`flush` call (not silently swallowed).
7. The default ``extraction_mode="sync"`` path is unchanged and
   spawns no executor thread.
8. Calling :meth:`close` twice is safe (second call is a no-op on the
   executor side).
"""

from __future__ import annotations

import json
import threading
import time
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
class _SlowExtractBackend:
    """Scripted backend that sleeps ``latency`` seconds per extract call.

    Models a slow LLM network hop: the ``"async"`` win is overlap of
    this latency with the caller's next operation. The reconcile
    prompt path returns an ``ADD`` fall-through so we exercise the
    full extract+reconcile loop under the executor thread.
    """

    replies: list[str] = field(default_factory=list)
    latency: float = 0.0
    name: str = "slow-extract"
    _extract_calls: int = 0
    _reconcile_calls: int = 0
    # ``generate`` may run on the executor thread; we keep a lock-free
    # counter since Python list append + int write are GIL-atomic
    # enough for the assertions we make here.

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            if self.latency:
                time.sleep(self.latency)
            i = self._extract_calls
            self._extract_calls += 1
            if i < len(self.replies):
                return self.replies[i]
            return "[]"
        if "You reconcile a new fact" in prompt:
            self._reconcile_calls += 1
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        return "{}"


@dataclass
class _RaisingBackend:
    """Backend whose extract call raises. Used to assert exceptions
    surface on :meth:`flush` rather than being silently dropped.
    """

    name: str = "raising"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            raise RuntimeError("boom from extractor")
        return "{}"


def _fact_reply(text: str) -> str:
    """Extract reply representing a single atomic fact."""
    return json.dumps([{"category": "other", "text": text}])


def test_async_mode_returns_immediately() -> None:
    """add_message must not block on the slow extract call in async mode.

    We give the extractor a 100ms stall. If add_message were waiting on
    it, the second call would see the elapsed wall-clock >= 100ms.
    Under async we expect well under that (submission overhead only).
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _SlowExtractBackend(replies=[_fact_reply("user likes tea")], latency=0.1)
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="async",
    )
    try:
        t0 = time.perf_counter()
        cm.add_message("user", "I like tea")
        elapsed = time.perf_counter() - t0
        # Well under the 100ms extractor stall — the turn write
        # happens synchronously but the LLM call is on the executor.
        assert elapsed < 0.05, f"add_message blocked for {elapsed:.3f}s"
        # Fact has not necessarily landed yet — it's in flight.
        assert cm.list_facts() == [] or len(cm.list_facts()) <= 1
        # After flush, it's guaranteed to be present.
        cm.flush()
        facts = [f.text for f in cm.list_facts()]
        assert "user likes tea" in facts, facts
    finally:
        cm.close()


def test_async_mode_flush_drains_pending() -> None:
    """Five in-flight turns all land after one flush() call."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    replies = [_fact_reply(f"fact-{i}") for i in range(5)]
    llm = _SlowExtractBackend(replies=replies, latency=0.02)
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="async",
    )
    try:
        for i in range(5):
            cm.add_message("user", f"turn {i}")
        cm.flush()
        fact_texts = {f.text for f in cm.list_facts()}
        assert fact_texts == {f"fact-{i}" for i in range(5)}
        # Drained — no futures left behind.
        assert cm._pending_futures == []
    finally:
        cm.close()


def test_async_mode_close_via_context_manager() -> None:
    """Context-manager exit drains; the fact is present afterwards."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _SlowExtractBackend(
        replies=[_fact_reply("user lives in Boston")], latency=0.02,
    )
    with ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="async",
    ) as cm:
        cm.add_message("user", "I live in Boston")
    # After __exit__ the executor has drained + shut down.
    fact_texts = {f.text for f in ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
    ).list_facts()}
    assert "user lives in Boston" in fact_texts, fact_texts


def test_async_mode_preserves_extraction_order_within_session() -> None:
    """max_workers=1 means extract order matches submission order.

    We record the order in which the executor thread calls the
    extractor (one entry per call). max_workers=1 guarantees this
    matches add_message submission order.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    order: list[str] = []
    order_lock = threading.Lock()

    @dataclass
    class OrderedBackend:
        name: str = "ordered"
        _idx: int = 0
        _n: int = 0

        def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
            if "You extract atomic facts" in prompt:
                with order_lock:
                    i = self._idx
                    self._idx += 1
                    # Stagger slightly so any parallelism would interleave.
                    order.append(f"start-{i}")
                time.sleep(0.01)
                with order_lock:
                    order.append(f"done-{i}")
                return _fact_reply(f"fact-{i}")
            if "You reconcile a new fact" in prompt:
                return json.dumps({"op": "ADD", "target_id": None})
            return "{}"

    llm = OrderedBackend()
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="async",
    )
    try:
        for i in range(4):
            cm.add_message("user", f"turn {i}")
        cm.flush()
    finally:
        cm.close()
    # With max_workers=1 each extract runs to completion before the
    # next one starts: ``start-i`` is always immediately followed by
    # ``done-i`` — never interleaved.
    assert order == [
        "start-0", "done-0",
        "start-1", "done-1",
        "start-2", "done-2",
        "start-3", "done-3",
    ], order


def test_async_mode_clear_session_flushes_first() -> None:
    """clear_session must drain pending facts then wipe — not the other
    way round (otherwise late futures resurrect a 'cleared' session).
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _SlowExtractBackend(
        replies=[_fact_reply(f"fact-{i}") for i in range(3)],
        latency=0.02,
    )
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="async",
    )
    try:
        for i in range(3):
            cm.add_message("user", f"turn {i}")
        # Immediately clear — flush-first means every pending future
        # runs, lands its fact, and then we delete them all.
        cm.clear_session()
        # No facts, no turns, no leakage.
        assert cm.list_facts() == []
        # All 3 extract calls fired (drained before wipe).
        assert llm._extract_calls == 3
    finally:
        cm.close()


def test_async_mode_exceptions_surface_on_flush() -> None:
    """Executor-thread exceptions must re-raise on flush(), not be dropped."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _RaisingBackend()
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="async",
    )
    try:
        cm.add_message("user", "hello")
        with pytest.raises(RuntimeError, match="boom from extractor"):
            cm.flush()
    finally:
        # close() re-enters flush(); by now the pending future list is
        # empty (flush drained + raised), so close is clean.
        cm.close()


def test_sync_mode_default_unchanged() -> None:
    """No extraction_mode kwarg means no executor + no thread spawned."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _SlowExtractBackend(replies=[_fact_reply("user likes tea")])
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
    )
    # Sync mode: no executor.
    assert cm._executor is None
    cm.add_message("user", "I like tea")
    # Fact is present immediately — no flush needed.
    assert "user likes tea" in {f.text for f in cm.list_facts()}
    # flush() is a no-op beyond the MemoryLayer pass-through.
    cm.flush()
    # close() stays safe in sync mode too.
    cm.close()


def test_double_close_is_safe() -> None:
    """Calling close() twice must not raise, regardless of mode."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = _SlowExtractBackend(replies=[_fact_reply("stable fact")])
    cm = ConversationalMemory(
        memory=mem, llm=llm, session_id="s1",
        ambiguous_threshold=0.0, near_dup_threshold=0.99,
        summary_every=999,
        extraction_mode="async",
    )
    cm.add_message("user", "something")
    cm.close()
    # Second call is a no-op on the executor side (already None).
    cm.close()
    # And remains safe after further calls.
    assert cm._executor is None
