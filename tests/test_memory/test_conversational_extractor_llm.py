"""Tests for the ``extractor_llm=`` kwarg on :class:`ConversationalMemory`.

Users running small local chat models (3B-ish) need a stronger LLM for
the two structured-JSON steps — fact extraction and reconcile — while
keeping the small model for chat + free-form summary.

These tests pin the routing contract:

- When ``extractor_llm`` is provided, EXTRACT_PROMPT and RECONCILE_PROMPT
  calls must go through it, and the main ``llm`` must never see them.
- SUMMARY_PROMPT always stays on the main ``llm`` (free-form text is
  fine for small models; no reason to burn the bigger one on it).
- When ``extractor_llm`` is omitted, behaviour is unchanged — all three
  prompts go to ``llm`` (backward compat).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch

from soma.memory import MemoryLayer
from soma.memory.conversational import ConversationalMemory


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic per-text vector with small overlap between similar phrasings."""
    h = hash(text) & 0xFFFFFFFF
    torch.manual_seed(h)
    return torch.randn(16)


@dataclass
class _RecordingBackend:
    """LLM backend that records every prompt it sees and returns scripted replies.

    Replies are dispatched by substring match against the prompt:

    - EXTRACT_PROMPT contains "You extract atomic facts".
    - RECONCILE_PROMPT contains "You reconcile a new fact".
    - SUMMARY_PROMPT contains "You are summarizing".

    A default reply of ``"[]"`` keeps the extract path safe when a call
    is unexpectedly routed here (keeps tests focused on the routing
    assertion rather than on downstream parse behaviour).
    """

    name: str = "recording"
    extract_reply: str = "[]"
    reconcile_reply: str = '{"op": "ADD", "target_id": null, "reason": "new"}'
    summary_reply: str = "Conversation summary."
    prompts_seen: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts_seen.append(prompt)
        if "You extract atomic facts" in prompt:
            return self.extract_reply
        if "You reconcile a new fact" in prompt:
            return self.reconcile_reply
        if "You are summarizing" in prompt:
            return self.summary_reply
        return "[]"

    def saw(self, marker: str) -> bool:
        return any(marker in p for p in self.prompts_seen)


def _make_mem() -> MemoryLayer:
    return MemoryLayer(embed_fn=_stub_embed, embed_dim=16)


class _AmbiguousScoreLayer(MemoryLayer):
    """MemoryLayer whose retrieve() pins scores into the ambiguous band.

    Mirrors the ``_FixedScoreLayer`` pattern used in
    ``test_conversational_reconcile.py`` — lets tests drive the LLM
    reconcile path without fighting cosine over randomly-seeded vectors.
    """

    def __init__(self, *, embed_fn: Any, embed_dim: int) -> None:
        super().__init__(embed_fn=embed_fn, embed_dim=embed_dim)

    def retrieve(self, query: str, k: int = 5, **kwargs: Any):  # type: ignore[override]
        hits = super().retrieve(query, k=k, **kwargs)
        if not hits:
            return hits
        from soma.memory.api import MemoryHit

        # Score 0.8 sits in the default [0.75, 0.92) ambiguous band, so
        # reconcile will call the LLM. Secondary hits get 0.1 so there's
        # one clear top candidate.
        out = []
        for i, h in enumerate(hits):
            score = 0.8 if i == 0 else 0.1
            out.append(
                MemoryHit(
                    node_id=h.node_id,
                    text=h.text,
                    score=score,
                    metadata=dict(h.metadata),
                    timestamp_step=h.timestamp_step,
                )
            )
        return out


# ----------------------------------------------------------------------
# Routing when extractor_llm is set
# ----------------------------------------------------------------------


def test_extractor_llm_used_for_extract_when_set() -> None:
    """EXTRACT prompts go to stub_extract; stub_chat never sees them."""
    mem = _make_mem()
    stub_chat = _RecordingBackend(name="chat")
    stub_extract = _RecordingBackend(
        name="extract",
        extract_reply=json.dumps(
            [{"category": "location", "text": "User lives in Seattle"}]
        ),
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=stub_chat,
        extractor_llm=stub_extract,
        session_id="s1",
        summary_every=999,  # disable summary cadence for this test
    )
    cm.add_message("user", "I live in Seattle")

    assert stub_extract.saw("You extract atomic facts")
    assert not stub_chat.saw("You extract atomic facts")


def test_extractor_llm_used_for_reconcile_when_set() -> None:
    """RECONCILE prompts go to stub_extract; stub_chat never sees them.

    Drive the ambiguous reconcile branch (0.75 <= score < 0.92) with
    :class:`_AmbiguousScoreLayer` so the test doesn't fight cosine over
    random vectors.
    """
    mem = _AmbiguousScoreLayer(embed_fn=_stub_embed, embed_dim=16)
    stub_chat = _RecordingBackend(name="chat")
    stub_extract = _RecordingBackend(
        name="extract",
        extract_reply=json.dumps(
            [{"category": "location", "text": "User lives in Boston"}]
        ),
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=stub_chat,
        extractor_llm=stub_extract,
        session_id="s1",
        summary_every=999,
    )
    # Plant a baseline fact so the second turn has a candidate to
    # reconcile against.
    cm.add_message("user", "I live in Boston")
    # Second turn's extracted fact will land at score=0.8 against the
    # plant → LLM reconcile path fires.
    stub_extract.extract_reply = json.dumps(
        [{"category": "location", "text": "User lives in Boston, MA"}]
    )
    cm.add_message("user", "Actually it's Boston, MA")

    assert stub_extract.saw("You reconcile a new fact")
    assert not stub_chat.saw("You reconcile a new fact")


def test_summary_still_uses_main_llm_even_with_extractor_set() -> None:
    """Summary (free-form prose) stays on the main ``llm`` — that's the point
    of the split: small model handles chat + summary, big model handles
    structured JSON."""
    mem = _make_mem()
    stub_chat = _RecordingBackend(name="chat")
    stub_extract = _RecordingBackend(name="extract")
    cm = ConversationalMemory(
        memory=mem,
        llm=stub_chat,
        extractor_llm=stub_extract,
        session_id="s1",
        summary_every=2,  # roll summary after 2 turns
    )
    cm.add_message("user", "Hello there")
    cm.add_message("assistant", "Hi back")

    assert stub_chat.saw("You are summarizing")
    assert not stub_extract.saw("You are summarizing")


# ----------------------------------------------------------------------
# Backward compat — extractor_llm omitted
# ----------------------------------------------------------------------


def test_extractor_llm_defaults_to_main_llm_when_omitted() -> None:
    """With no ``extractor_llm``, all three prompt types route through ``llm``.

    This pins the backward-compat contract — the existing test suite
    relies on it (one-LLM construction is the common path).
    """
    mem = _make_mem()
    stub_chat = _RecordingBackend(
        name="chat",
        extract_reply=json.dumps(
            [{"category": "identity", "text": "User's name is Alex"}]
        ),
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=stub_chat,
        session_id="s1",
        summary_every=1,
    )
    cm.add_message("user", "I'm Alex")

    # Both extract + summary prompts landed on stub_chat.
    assert stub_chat.saw("You extract atomic facts")
    assert stub_chat.saw("You are summarizing")


def test_extractor_llm_attribute_falls_back_to_main_llm() -> None:
    """``self._extractor_llm`` is the main ``llm`` when the kwarg is unset,
    so call sites don't need None-checks."""
    mem = _make_mem()
    stub_chat = _RecordingBackend(name="chat")
    cm = ConversationalMemory(memory=mem, llm=stub_chat, session_id="s1")
    assert cm._extractor_llm is stub_chat


def test_extractor_llm_attribute_holds_separate_backend_when_set() -> None:
    """When the kwarg is provided, ``_extractor_llm`` is that backend
    (not the main ``llm``)."""
    mem = _make_mem()
    stub_chat = _RecordingBackend(name="chat")
    stub_extract = _RecordingBackend(name="extract")
    cm = ConversationalMemory(
        memory=mem,
        llm=stub_chat,
        extractor_llm=stub_extract,
        session_id="s1",
    )
    assert cm._extractor_llm is stub_extract
    assert cm._llm is stub_chat
