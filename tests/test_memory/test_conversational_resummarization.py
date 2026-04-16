"""Phase 17: summary re-summarization from raw turns.

Every Mth rolled summary (``resummarize_every``, default 5) is re-
derived from the last ``M × summary_every`` raw turns using
:data:`RESUMMARY_PROMPT`, bypassing the previous summary entirely.
That breaks the chained-summarization drift loop — where each summary
is built on the previous one — in long-lived sessions.

These tests pin the cadence + prompt selection:

- First summary always uses the standard :data:`SUMMARY_PROMPT`.
- With ``resummarize_every=3``, summary #3, #6, #9 use
  :data:`RESUMMARY_PROMPT`; #1, #2, #4, #5, #7, #8 use ``SUMMARY_PROMPT``.
- ``resummarize_every=0`` disables the feature; every summary uses
  ``SUMMARY_PROMPT``.
- The re-summary prompt contains no "previous summary" language — we
  capture the actual prompt text passed to the LLM and assert that
  directly.
- Omitting the kwarg at construction time keeps the default (5) and
  old callers keep working.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from soma.memory import ConversationalMemory, MemoryLayer


def _stable_hash(text: str) -> int:
    return int.from_bytes(
        hashlib.sha1(text.encode("utf-8")).digest()[:4], "little"
    )


def _stub_embed(text: str) -> torch.Tensor:
    """Deterministic per-text vector; PYTHONHASHSEED-independent."""
    vec = torch.zeros(16)
    for tok in text.lower().split():
        seed = _stable_hash(tok)
        gen = torch.Generator().manual_seed(seed)
        vec = vec + torch.randn(16, generator=gen)
    return vec


@dataclass
class CapturingBackend:
    """LLM backend that records every prompt it's asked to generate.

    Returns ``[]`` for extract prompts so no facts get reconciled
    (keeping the test focused on the summary path), canned JSON for
    reconcile prompts (none should fire here anyway), and a stable
    summary string we can recognise for summary / resummary prompts.
    """

    name: str = "capturing-resummary"
    prompts: list[str] = field(default_factory=list)
    summary_prompts: list[str] = field(default_factory=list)
    resummary_prompts: list[str] = field(default_factory=list)
    _summary_calls: int = 0

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        if "You extract atomic facts" in prompt:
            return "[]"
        if "You reconcile a new fact" in prompt:
            return '{"op": "NOOP", "target_id": null, "reason": "skip"}'
        # Summary branch — split on the RESUMMARY_PROMPT's distinctive
        # marker so we can classify which template produced the call.
        if "re-derived from the raw turns" in prompt:
            self.resummary_prompts.append(prompt)
        else:
            self.summary_prompts.append(prompt)
        self._summary_calls += 1
        return f"summary-{self._summary_calls}"


def _drive_turns(cm: ConversationalMemory, count: int) -> None:
    for i in range(count):
        cm.add_message("user", f"turn number {i} with unique phrase")


def test_first_summary_uses_standard_prompt() -> None:
    """Summary #1 always uses SUMMARY_PROMPT regardless of resummarize_every."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = CapturingBackend()
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="phase17-a",
        summary_every=3,
        resummarize_every=5,
    )

    _drive_turns(cm, 3)

    assert len(llm.summary_prompts) == 1, (
        f"expected exactly one standard summary, got "
        f"{len(llm.summary_prompts)} summaries + "
        f"{len(llm.resummary_prompts)} resummaries"
    )
    assert len(llm.resummary_prompts) == 0
    # Counter advanced exactly once for the single write.
    assert cm._summaries_generated == 1


def test_nth_summary_uses_resummary_prompt_at_boundary() -> None:
    """resummarize_every=3 → summaries #3, #6, #9 are RESUMMARY_PROMPT.

    Summaries #1, #2, #4, #5, #7, #8 use SUMMARY_PROMPT. Drive nine
    full summary cycles and classify every call by its prompt text.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = CapturingBackend()
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="phase17-b",
        summary_every=2,
        resummarize_every=3,
    )

    # 2 turns per summary × 9 summaries = 18 turns.
    _drive_turns(cm, 2 * 9)

    assert cm._summaries_generated == 9
    assert len(llm.summary_prompts) == 6  # #1, #2, #4, #5, #7, #8
    assert len(llm.resummary_prompts) == 3  # #3, #6, #9

    # Every resummary prompt must carry the distinctive marker.
    for p in llm.resummary_prompts:
        assert "re-derived from the raw turns" in p
    for p in llm.summary_prompts:
        assert "re-derived from the raw turns" not in p


def test_resummarization_disabled_by_zero() -> None:
    """resummarize_every=0 → every summary uses SUMMARY_PROMPT."""
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = CapturingBackend()
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="phase17-c",
        summary_every=2,
        resummarize_every=0,
    )

    _drive_turns(cm, 2 * 6)

    assert cm._summaries_generated == 6
    assert len(llm.summary_prompts) == 6
    assert len(llm.resummary_prompts) == 0


def test_resummary_reads_raw_turns_only() -> None:
    """The re-summary prompt must not interpolate a previous summary.

    The RESUMMARY_PROMPT explicitly instructs the LLM "do not reference
    any prior summary" — that exact instruction string is expected in
    the prompt. What must NOT appear is the content of any previously-
    written summary (i.e. the template must not have interpolated the
    earlier summary text into the prompt). We test by priming the
    session with a distinctive first summary, then checking that its
    content does not leak into the subsequent re-summary call.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)

    class MarkedCapturingBackend(CapturingBackend):
        def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
            self.prompts.append(prompt)
            if "You extract atomic facts" in prompt:
                return "[]"
            if "You reconcile a new fact" in prompt:
                return '{"op": "NOOP", "target_id": null, "reason": "skip"}'
            if "re-derived from the raw turns" in prompt:
                self.resummary_prompts.append(prompt)
                return "RESUMMARY-CANARY"
            # Distinctive, unmistakable first-summary body.
            self.summary_prompts.append(prompt)
            return "FIRST-SUMMARY-CANARY-XYZ"

    llm = MarkedCapturingBackend()
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="phase17-d",
        summary_every=2,
        resummarize_every=2,
    )

    # 2 turns per summary × 2 summaries = 4 turns. The 2nd is a resummary.
    _drive_turns(cm, 4)

    assert len(llm.resummary_prompts) == 1
    resummary_prompt = llm.resummary_prompts[0]

    # The prior summary's body must NOT be interpolated into the
    # re-summary prompt — that's the whole point of Phase 17.
    assert "FIRST-SUMMARY-CANARY-XYZ" not in resummary_prompt, (
        "re-summary prompt leaked the prior summary text — "
        "it must be re-derived from raw turns only"
    )

    # The template's instruction to ignore prior summaries is pinned.
    assert "do not reference any prior summary" in resummary_prompt.lower()
    # Positive anchor: the re-derivation phrasing stays pinned.
    assert "re-derived from the raw turns" in resummary_prompt.lower()

    # The re-summary was written with metadata.resummary=True so the
    # store can be audited later (is this entry chained or re-derived?).
    summaries = [
        mem.get(nid)
        for nid in mem._ids
        if (h := mem.get(nid)) and h.metadata.get("type") == "summary"
    ]
    assert len(summaries) == 2
    assert summaries[0] is not None and summaries[1] is not None
    assert summaries[0].metadata.get("resummary") is False
    assert summaries[1].metadata.get("resummary") is True


def test_resummarize_every_unset_backward_compat() -> None:
    """Omitting resummarize_every keeps the default (5) and still works.

    Old callers that construct ConversationalMemory without the kwarg
    continue to behave. With the default of 5 and summary_every=2, the
    5th summary (turn 10) fires resummary; the preceding four are
    chained summaries.
    """
    mem = MemoryLayer(embed_fn=_stub_embed, embed_dim=16)
    llm = CapturingBackend()
    cm = ConversationalMemory(
        memory=mem,
        llm=llm,
        session_id="phase17-e",
        summary_every=2,
        # No resummarize_every: defaults to 5.
    )
    # Sanity-check the default landed.
    assert cm._resummarize_every == 5

    # 2 turns per summary × 5 summaries = 10 turns.
    _drive_turns(cm, 10)

    assert cm._summaries_generated == 5
    assert len(llm.summary_prompts) == 4  # #1..#4 chained
    assert len(llm.resummary_prompts) == 1  # #5 re-derived
