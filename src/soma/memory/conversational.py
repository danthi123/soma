"""ConversationalMemory — Mem0/Zep-style sugar over :class:`MemoryLayer`.

Wraps a raw MemoryLayer with a two-phase pipeline per user turn:

1. **Extract**: LLM prompt that returns a JSON list of atomic facts
   from the user's message. Closed-vocab category, empty-list example
   as a small-model anchor.
2. **Reconcile**: each fact is compared against the top-k existing
   memories. Threshold short-circuit: >=0.92 cosine → skip (near-dup),
   <0.75 → ADD without calling the LLM, otherwise the LLM chooses
   ADD / UPDATE / SUPERSEDE / NOOP.

SUPERSEDE borrows from Zep — we don't delete the old entry, we set
``metadata.superseded_by = new_id`` on it. Default retrieval filters
superseded entries out but callers can pass ``include_superseded=True``
to see history.

Raw turns are also stored (``metadata.type = "turn"``) so downstream
LoCoMo-style eval stays compatible. Rolling session summaries are
written every N turns (``metadata.type = "summary"``).

See docs/plans/2026-04-16-phase-2-conversational-memory.md for the
full design and ``tests/test_memory/test_conversational_*.py`` for the
behavioural contract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soma.llm.backends import LLMBackend
    from soma.memory.api import MemoryLayer

from soma.memory.conversational_prompts import EXTRACT_PROMPT

logger = logging.getLogger("soma.memory")

# Closed vocabulary matching EXTRACT_PROMPT. LLM outputs anything else
# get normalized to "other" so unknown-category doesn't drop the fact.
_CATEGORIES: frozenset[str] = frozenset(
    {"identity", "location", "preference", "relationship", "goal", "other"}
)


@dataclass(frozen=True)
class ExtractedFact:
    """One atomic fact returned by the extractor LLM."""

    category: str
    text: str


class ConversationalMemory:
    """Mem0/Zep-style conversational wrapper over :class:`MemoryLayer`.

    See module docstring for the design. Public surface:

    - :meth:`add_message` — the main entry point; extracts + reconciles
      facts, stores the raw turn, rolls summaries on N-turn cadence.
    - :meth:`retrieve` — like ``MemoryLayer.retrieve`` but scoped to this
      session by default and filtering out superseded entries.
    - :meth:`list_facts` / :meth:`get_summary` — introspection.
    - :meth:`supersede` / :meth:`clear_session` — lifecycle helpers.
    - :meth:`flush` — durability hand-off (sync mode: no-op).
    """

    def __init__(
        self,
        *,
        memory: MemoryLayer,
        llm: LLMBackend,
        session_id: str | None = None,
        near_dup_threshold: float = 0.92,
        ambiguous_threshold: float = 0.75,
        summary_every: int = 20,
        extract_assistant: bool = False,
    ) -> None:
        self._memory = memory
        self._llm = llm
        # session_id default="default" so simple callers don't fight the
        # API. Multi-session callers pass an explicit id.
        self._session_id: str = session_id or "default"
        if not 0.0 <= ambiguous_threshold <= near_dup_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= ambiguous <= near_dup <= 1, "
                f"got ambiguous={ambiguous_threshold}, near_dup={near_dup_threshold}"
            )
        self._near_dup_threshold: float = float(near_dup_threshold)
        self._ambiguous_threshold: float = float(ambiguous_threshold)
        self._summary_every: int = int(summary_every)
        self._extract_assistant: bool = bool(extract_assistant)
        self._turn_counter: int = 0

    # ------------------------------------------------------------------
    # Internals — extraction
    # ------------------------------------------------------------------
    def _extract_facts(self, message: str) -> list[ExtractedFact]:
        """Call the LLM with :data:`EXTRACT_PROMPT` and parse the JSON output.

        Safe parse: a JSONDecodeError / KeyError / TypeError anywhere in
        the parse path returns an empty list and logs at WARNING. The
        caller never sees a crash from a broken LLM reply.

        Categories outside :data:`_CATEGORIES` are normalized to
        ``"other"`` so a llama3.2-style hallucination like
        ``"car_preference"`` still round-trips as a usable fact.
        """
        prompt = EXTRACT_PROMPT.format(message=message)
        raw = self._llm.generate(prompt, max_tokens=512)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "extract: LLM returned non-JSON output; dropping this turn's facts",
                extra={
                    "event": "extract_parse_failure",
                    "session_id": self._session_id,
                    "reply_prefix": raw[:80],
                },
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "extract: LLM returned non-list JSON; dropping this turn's facts",
                extra={
                    "event": "extract_shape_failure",
                    "session_id": self._session_id,
                    "got_type": type(parsed).__name__,
                },
            )
            return []

        facts: list[ExtractedFact] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            try:
                raw_cat = entry["category"]
                raw_text = entry["text"]
            except KeyError:
                logger.warning(
                    "extract: fact missing category/text keys; skipping",
                    extra={
                        "event": "extract_missing_keys",
                        "session_id": self._session_id,
                        "entry_keys": sorted(entry.keys()),
                    },
                )
                continue
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            cat = raw_cat if raw_cat in _CATEGORIES else "other"
            facts.append(ExtractedFact(category=cat, text=raw_text.strip()))
        if not facts and parsed:
            # Parsed shape was OK but every entry was dropped. Log once
            # so operators can notice a model that's systematically off.
            logger.warning(
                "extract: all %d parsed entries were dropped",
                len(parsed),
                extra={
                    "event": "extract_all_dropped",
                    "session_id": self._session_id,
                },
            )
        return facts
