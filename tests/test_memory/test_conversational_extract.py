"""Tests for ConversationalMemory._extract_facts — LLM-driven atomic fact
extraction with safe JSON parse.

The extraction step is the core of the Mem0/Zep pipeline. These tests
pin the behaviours that matter for quality on small local LLMs:

- Compound messages yield multiple atomic facts.
- Greetings / filler yield an empty list (no hallucinated facts).
- A broken LLM reply (prose, bad JSON, missing keys) must never crash;
  it returns ``[]`` and logs at WARNING.
- Categories outside the closed vocabulary are normalized to "other"
  (so a llama3.2 "car_preference" hallucination still rounds-trips).
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
    h = hash(text)
    return torch.tensor(
        [(h >> i) & 0xF for i in range(0, 32, 4)], dtype=torch.float32
    )


@dataclass
class ExtractScriptedBackend:
    """Scripted LLM backend. Returns the next entry in ``replies`` per call.

    Mirrors the ``CapturingBackend`` pattern from tests/test_llm/test_rag.py:
    the test seeds a list of canned replies; each ``generate()`` call
    pops one. Last call repeats if replies is exhausted.
    """

    replies: list[str] = field(default_factory=list)
    name: str = "extract-scripted"
    prompts: list[str] = field(default_factory=list)

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            return "[]"
        if len(self.prompts) <= len(self.replies):
            return self.replies[len(self.prompts) - 1]
        return self.replies[-1]


def _make_cm(replies: list[str]) -> tuple[ConversationalMemory, ExtractScriptedBackend]:
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=8)
    llm = ExtractScriptedBackend(replies=replies)
    cm = ConversationalMemory(memory=mem, llm=llm, session_id="s1")
    return cm, llm


def test_extracts_two_facts_from_compound_message() -> None:
    replies = [
        json.dumps(
            [
                {"category": "identity", "text": "User's name is Alex"},
                {"category": "location", "text": "User lives in Boston"},
            ]
        )
    ]
    cm, _ = _make_cm(replies)
    facts = cm._extract_facts("I'm Alex and live in Boston")
    assert len(facts) == 2
    texts = [f.text for f in facts]
    assert "User's name is Alex" in texts
    assert "User lives in Boston" in texts


def test_extracts_zero_facts_from_greeting() -> None:
    cm, _ = _make_cm(["[]"])
    facts = cm._extract_facts("Hey")
    assert facts == []


def test_json_parse_failure_returns_empty_list_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLM returns prose → parse fails, no crash, WARNING log line."""
    cm, _ = _make_cm(["I'd love to help with that! Here are some facts:"])
    with caplog.at_level(logging.WARNING, logger="soma.memory"):
        facts = cm._extract_facts("I live in Boston")
    assert facts == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "extract must log WARNING when parse fails"


def test_missing_keys_returns_empty_list_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """JSON valid but missing ``category``/``text`` keys → empty list + WARNING."""
    cm, _ = _make_cm([json.dumps([{"kind": "identity", "body": "Alex"}])])
    with caplog.at_level(logging.WARNING, logger="soma.memory"):
        facts = cm._extract_facts("I'm Alex")
    assert facts == []
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_category_outside_closed_vocab_is_mapped_to_other() -> None:
    """llama3.2 sometimes invents categories; map them to "other" instead
    of dropping the fact."""
    replies = [
        json.dumps(
            [{"category": "car_preference", "text": "User drives a Subaru"}]
        )
    ]
    cm, _ = _make_cm(replies)
    facts = cm._extract_facts("I drive a Subaru")
    assert len(facts) == 1
    assert facts[0].category == "other"
    assert facts[0].text == "User drives a Subaru"


def test_extract_skips_non_dict_entries() -> None:
    """JSON list with garbage entries (strings, nulls) is filtered, not
    crashed. Small models sometimes emit mixed outputs."""
    replies = [
        json.dumps(
            [
                "random string",
                None,
                {"category": "identity", "text": "User's name is Alex"},
            ]
        )
    ]
    cm, _ = _make_cm(replies)
    facts = cm._extract_facts("I'm Alex")
    assert len(facts) == 1
    assert facts[0].text == "User's name is Alex"


def test_extraction_skips_empty_text_field() -> None:
    """Some models emit objects with empty text. Drop them silently."""
    replies = [
        json.dumps(
            [
                {"category": "identity", "text": ""},
                {"category": "location", "text": "   "},
                {"category": "identity", "text": "User's name is Alex"},
            ]
        )
    ]
    cm, _ = _make_cm(replies)
    facts = cm._extract_facts("I'm Alex")
    assert len(facts) == 1
    assert facts[0].text == "User's name is Alex"
