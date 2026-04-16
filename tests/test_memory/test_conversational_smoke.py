"""End-to-end smoke test: the 'Alex moved' scenario.

Walks ConversationalMemory through a realistic two-turn exchange:
1. User: "I'm Alex and live in Portland"  -> 2 facts extracted, both ADD.
2. User: "I moved to Boston"               -> 1 fact extracted; reconcile
   SUPERSEDEs the Portland fact with the Boston one.

Assertions:
- After turn 2 the default retrieve sees the Boston fact but not Portland.
- include_superseded=True shows both.
- list_facts() returns only live entries (name + Boston), but the bundle
  still contains the superseded Portland entry with the correct pointer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import torch

from soma.memory import ConversationalMemory, MemoryLayer


def _stable_hash(text: str) -> int:
    """Process-independent hash; the builtin hash() is PYTHONHASHSEED-salted."""
    return int.from_bytes(
        hashlib.sha1(text.encode("utf-8")).digest()[:4], "little"
    )


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic per-text vector that maps overlapping tokens near each other.

    Uses a stable hash (SHA1 prefix) so embeddings are reproducible
    across processes; the builtin ``hash()`` is PYTHONHASHSEED-salted
    and would produce different cosines on every test run.
    """
    vec = torch.zeros(16)
    for tok in text.lower().split():
        seed = _stable_hash(tok)
        gen = torch.Generator().manual_seed(seed)
        vec = vec + torch.randn(16, generator=gen)
    return vec


@dataclass
class ExtractScriptedBackend:
    """Scripted backend for the smoke scenario.

    Generates appropriate replies based on prompt content (extract /
    reconcile / summary) and call-order within each. Mirrors the
    CapturingBackend pattern from tests/test_llm/test_rag.py.
    """

    extract_replies: list[str] = field(default_factory=list)
    reconcile_replies: list[str] = field(default_factory=list)
    name: str = "smoke-scripted"
    _extract_calls: int = 0
    _reconcile_calls: int = 0
    reconcile_prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            i = self._extract_calls
            self._extract_calls += 1
            if i < len(self.extract_replies):
                return self.extract_replies[i]
            return "[]"
        if "You reconcile a new fact" in prompt:
            i = self._reconcile_calls
            self._reconcile_calls += 1
            self.reconcile_prompts.append(prompt)
            if i < len(self.reconcile_replies):
                return self.reconcile_replies[i]
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        return "{}"


def test_alex_moved_end_to_end_supersede() -> None:
    """The headline scenario: retrieve a moved-to location, preserved history."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)

    # Pre-seed the bundle with the turn-1 facts so we know their ids when
    # scripting the turn-2 reconcile SUPERSEDE reply. This is the same
    # shape an earlier add_message call would produce.
    mem.store(
        "User's name is Alex",
        metadata={
            "session_id": "alex",
            "type": "fact",
            "category": "identity",
        },
    )
    portland_id = mem.store(
        "User lives in Portland",
        metadata={
            "session_id": "alex",
            "type": "fact",
            "category": "location",
        },
    )

    # Turn 2: "I moved to Boston" -> extract 1 fact about Boston;
    # reconcile SUPERSEDEs the Portland fact using its id.
    llm = ExtractScriptedBackend(
        extract_replies=[
            json.dumps(
                [{"category": "location", "text": "User lives in Boston"}]
            ),
        ],
        reconcile_replies=[
            json.dumps(
                {
                    "op": "SUPERSEDE",
                    "target_id": portland_id,
                    "reason": "user moved",
                }
            ),
        ],
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="alex",
        # Lower ambiguous threshold so our synthetic embedder lands inside
        # the LLM-consult band (the stub vectors have modest overlap).
        ambiguous_threshold=0.0,
        near_dup_threshold=0.99,
    )

    cm.add_message("user", "I moved to Boston")

    # Exactly one reconcile call (for the single Boston fact).
    assert llm._reconcile_calls == 1
    # Exactly one extract call.
    assert llm._extract_calls == 1

    # Default retrieve: Boston visible, Portland filtered.
    hits = cm.retrieve("where does the user live?", k=10)
    texts = [h.text for h in hits]
    assert "User lives in Boston" in texts, texts
    assert "User lives in Portland" not in texts, texts

    # include_superseded=True: both visible.
    all_hits = cm.retrieve(
        "where does the user live?", k=10, include_superseded=True,
    )
    all_texts = [h.text for h in all_hits]
    assert "User lives in Boston" in all_texts
    assert "User lives in Portland" in all_texts

    # list_facts() shows name + Boston, NOT Portland.
    facts = cm.list_facts()
    fact_texts = {f.text for f in facts}
    assert "User's name is Alex" in fact_texts
    assert "User lives in Boston" in fact_texts
    assert "User lives in Portland" not in fact_texts

    # The Portland entry still exists in the bundle and points at Boston.
    portland_hit = mem.get(portland_id)
    assert portland_hit is not None
    boston_id = None
    for nid in mem._ids:
        h = mem.get(nid)
        if h and h.text == "User lives in Boston":
            boston_id = nid
            break
    assert boston_id is not None
    assert portland_hit.metadata["superseded_by"] == boston_id
    boston_hit = mem.get(boston_id)
    assert boston_hit is not None
    assert boston_hit.metadata["supersedes"] == portland_id

    # The raw turn was also stored.
    turns = [
        mem.get(nid)
        for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "turn"
    ]
    assert any(
        t is not None and t.text == "I moved to Boston" for t in turns
    )


def test_alex_full_two_turn_flow_from_scratch() -> None:
    """Both turns via add_message(); no pre-seeding.

    Turn 1 extracts 2 facts (name + Portland). Both land via the
    low-similarity ADD path (no LLM round-trip) given the synthetic
    embedding's cosine profile here. Turn 2 extracts 1 fact (Boston),
    whose cosine against the Portland fact lands in the ambiguous
    band, so the LLM is consulted and we script a SUPERSEDE reply.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = ExtractScriptedBackend(
        extract_replies=[
            json.dumps(
                [
                    {"category": "identity", "text": "User's name is Alex"},
                    {"category": "location", "text": "User lives in Portland"},
                ]
            ),
            json.dumps(
                [{"category": "location", "text": "User lives in Boston"}]
            ),
        ],
    )
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="alex",
        ambiguous_threshold=0.0,
        near_dup_threshold=0.99,
        summary_every=999,  # disable summaries for this test
    )

    cm.add_message("user", "I'm Alex and live in Portland")
    # After turn 1: 1 turn + 2 facts.
    assert len(mem) == 3

    # Find the Portland id so we can script the reconcile reply now.
    portland_id: str | None = None
    for nid in mem._ids:
        h = mem.get(nid)
        if h and h.text == "User lives in Portland":
            portland_id = nid
            break
    assert portland_id is not None
    # The Boston reconcile is the first LLM reconcile call across both
    # turns (turn-1 facts both land via the low-sim short-circuit), so
    # the SUPERSEDE reply goes at index 0 of the scripted list.
    llm.reconcile_replies = [
        json.dumps(
            {
                "op": "SUPERSEDE",
                "target_id": portland_id,
                "reason": "user moved",
            }
        )
    ]

    cm.add_message("user", "I moved to Boston")

    # Default retrieve excludes Portland.
    hits = cm.retrieve("where does the user live?", k=10)
    texts = [h.text for h in hits]
    assert "User lives in Boston" in texts
    assert "User lives in Portland" not in texts
