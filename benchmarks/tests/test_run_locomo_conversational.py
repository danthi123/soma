"""Smoke test for ConversationalSomaAdapter — the ``--conversational``
LoCoMo variant.

Runs a 3-turn fixture through the adapter with a scripted LLM so the
test has no external dependency (no Ollama, no network). Verifies
that:
- The adapter can store raw turns via ``add_message(speaker, text)``.
- Retrieve returns BenchmarkHit objects the harness understands.
- The "facts stored / turns processed" ratio is non-zero when the
  scripted extractor returns facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import torch

from benchmarks.harness.adapters.soma import ConversationalSomaAdapter


def _stable_hash(text: str) -> int:
    return int.from_bytes(
        hashlib.sha1(text.encode("utf-8")).digest()[:4], "little"
    )


def _stub_embed(text: str) -> torch.Tensor:
    vec = torch.zeros(16)
    for tok in text.lower().split():
        gen = torch.Generator().manual_seed(_stable_hash(tok))
        vec = vec + torch.randn(16, generator=gen)
    return vec


@dataclass
class _ScriptedLLM:
    """Scripted LLM that returns per-extract replies and default ADD for reconcile."""

    extract_replies: list[str] = field(default_factory=list)
    name: str = "scripted"
    _ecalls: int = 0

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        if "You extract atomic facts" in prompt:
            i = self._ecalls
            self._ecalls += 1
            if i < len(self.extract_replies):
                return self.extract_replies[i]
            return "[]"
        if "You reconcile a new fact" in prompt:
            return json.dumps({"op": "ADD", "target_id": None, "reason": "new"})
        return "{}"


def test_conversational_adapter_stores_and_retrieves() -> None:
    """Store 3 turns, retrieve one - adapter round-trips through
    ConversationalMemory + reports a non-zero facts-per-turn ratio."""
    llm = _ScriptedLLM(
        extract_replies=[
            json.dumps(
                [{"category": "location", "text": "User lives in Boston"}]
            ),
            json.dumps(
                [{"category": "preference", "text": "User prefers Thai food"}]
            ),
            json.dumps([]),  # greeting
        ]
    )
    adapter = ConversationalSomaAdapter(
        llm=llm, embed_fn=_stub_embed, embed_dim=16, session_id="test",
    )
    adapter.prepare()

    # Same method signature as SomaAdapter.store(text, metadata=...)
    # so the harness runner stays compatible.
    adapter.store(
        "I live in Boston", metadata={"sample_id": "s", "dia_id": "D1:0"}
    )
    adapter.store(
        "I prefer Thai food", metadata={"sample_id": "s", "dia_id": "D1:1"}
    )
    adapter.store("Hello!", metadata={"sample_id": "s", "dia_id": "D1:2"})

    assert adapter.turns_processed == 3
    # Two facts extracted (the greeting yielded none).
    assert adapter.facts_stored == 2

    # Retrieve works and returns BenchmarkHit.
    hits = adapter.retrieve("where does the user live?", k=3)
    assert len(hits) >= 1
    # Retrieve must return at least one entry that matches either a fact
    # ('User lives in Boston') or the raw turn ('I live in Boston').
    texts = [h.text for h in hits]
    assert any("Boston" in t for t in texts), texts

    adapter.teardown()


def test_facts_per_turn_ratio_reported() -> None:
    """Adapter exposes facts_stored / turns_processed counters the report
    can surface as a 'facts per turn' column."""
    llm = _ScriptedLLM(
        extract_replies=[
            json.dumps(
                [
                    {"category": "identity", "text": "User's name is Alex"},
                    {"category": "location", "text": "User lives in Boston"},
                ]
            )
        ]
    )
    adapter = ConversationalSomaAdapter(
        llm=llm, embed_fn=_stub_embed, embed_dim=16, session_id="test",
    )
    adapter.prepare()
    adapter.store(
        "I'm Alex and live in Boston",
        metadata={"sample_id": "s", "dia_id": "D1:0"},
    )
    assert adapter.turns_processed == 1
    assert adapter.facts_stored == 2
    adapter.teardown()
