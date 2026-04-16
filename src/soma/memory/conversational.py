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

from soma.memory.conversational_prompts import EXTRACT_PROMPT, RECONCILE_PROMPT

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

    # ------------------------------------------------------------------
    # Internals - reconcile
    # ------------------------------------------------------------------
    def _add_fact(
        self,
        text: str,
        *,
        category: str = "other",
        extra_meta: dict[str, object] | None = None,
    ) -> str:
        """Store a new fact entry tagged with this session. Returns node_id."""
        meta: dict[str, object] = {
            "session_id": self._session_id,
            "type": "fact",
            "category": category,
        }
        if extra_meta:
            meta.update(extra_meta)
        return self._memory.store(text, metadata=meta)

    def _reconcile(self, fact: ExtractedFact) -> str | None:
        """Decide what to do with ``fact`` given the top-k nearest stored entries.

        Threshold short-circuits before any LLM round-trip:

        - ``max_score >= near_dup_threshold`` (default 0.92) -> return
          ``None``; the fact is effectively a duplicate.
        - ``max_score < ambiguous_threshold`` (default 0.75) OR no
          candidates -> ADD without asking the LLM.

        In the ambiguous range we call the LLM once with RECONCILE_PROMPT
        and dispatch on ``op``:

        - ``ADD`` -> store new.
        - ``UPDATE target_id`` -> forget old, store new with
          ``metadata.supersedes = target_id``.
        - ``SUPERSEDE target_id`` -> store new with supersedes pointer,
          then call ``memory.update_metadata(target_id, ...)`` to mark
          the old entry with ``superseded_by = new_id``. Preserves
          history (Zep-style "invalidate, don't delete").
        - ``NOOP`` -> return ``None``.

        Any parse failure or unknown op in the ambiguous branch falls
        back to ADD - safer to keep a fact than lose it on a broken
        LLM reply. Logged at WARNING for operator visibility.
        """
        # Scope retrieve to this session so reconcile doesn't try to
        # merge across users.
        candidates = self._memory.retrieve(
            fact.text,
            k=5,
            where={
                "session_id": self._session_id,
                "type": "fact",
            },
        )
        # Skip already-superseded entries - they're history, not live state.
        live_candidates = [
            c for c in candidates
            if c.metadata.get("superseded_by") is None
        ]
        if not live_candidates:
            return self._add_fact(fact.text, category=fact.category)

        max_score = max(c.score for c in live_candidates)
        if max_score >= self._near_dup_threshold:
            return None
        if max_score < self._ambiguous_threshold:
            return self._add_fact(fact.text, category=fact.category)

        # Ambiguous range: consult the LLM.
        return self._reconcile_with_llm(fact, live_candidates)

    def _reconcile_with_llm(
        self,
        fact: ExtractedFact,
        candidates: list[object],  # list[MemoryHit]
    ) -> str | None:
        candidate_block = "\n".join(
            f"[id={c.node_id}] (score={c.score:.3f}) {c.text}"  # type: ignore[attr-defined]
            for c in candidates
        )
        prompt = RECONCILE_PROMPT.format(
            new_fact=fact.text, candidates=candidate_block
        )
        raw = self._llm.generate(prompt, max_tokens=256)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise TypeError(f"expected JSON object, got {type(parsed).__name__}")
            op = parsed["op"]
            target_id = parsed.get("target_id")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "reconcile: LLM reply unparseable (%s); falling back to ADD",
                type(exc).__name__,
                extra={
                    "event": "reconcile_parse_failure",
                    "session_id": self._session_id,
                    "reply_prefix": raw[:80],
                },
            )
            return self._add_fact(fact.text, category=fact.category)

        if op == "NOOP":
            return None
        if op == "ADD":
            return self._add_fact(fact.text, category=fact.category)
        if op == "UPDATE":
            if not isinstance(target_id, str) or target_id not in self._memory:
                logger.warning(
                    "reconcile: UPDATE missing valid target_id; falling back to ADD",
                    extra={
                        "event": "reconcile_update_no_target",
                        "session_id": self._session_id,
                    },
                )
                return self._add_fact(fact.text, category=fact.category)
            self._memory.forget(target_id)
            return self._add_fact(
                fact.text,
                category=fact.category,
                extra_meta={"supersedes": target_id},
            )
        if op == "SUPERSEDE":
            if not isinstance(target_id, str) or target_id not in self._memory:
                logger.warning(
                    "reconcile: SUPERSEDE missing valid target_id; "
                    "falling back to ADD",
                    extra={
                        "event": "reconcile_supersede_no_target",
                        "session_id": self._session_id,
                    },
                )
                return self._add_fact(fact.text, category=fact.category)
            new_id = self._add_fact(
                fact.text,
                category=fact.category,
                extra_meta={"supersedes": target_id},
            )
            self._memory.update_metadata(
                target_id,
                {
                    "superseded_by": new_id,
                    "superseded_at_step": self._memory._step,
                },
            )
            return new_id
        # Unknown op - log and fall back to ADD so we don't lose the fact.
        logger.warning(
            "reconcile: unknown op %r; falling back to ADD",
            op,
            extra={
                "event": "reconcile_unknown_op",
                "session_id": self._session_id,
            },
        )
        return self._add_fact(fact.text, category=fact.category)

    # ------------------------------------------------------------------
    # Internals — reconcile
    # ------------------------------------------------------------------
    def _add_fact(
        self,
        text: str,
        *,
        category: str = "other",
        extra_meta: dict[str, object] | None = None,
    ) -> str:
        """Store a new fact entry tagged with this session. Returns node_id."""
        meta: dict[str, object] = {
            "session_id": self._session_id,
            "type": "fact",
            "category": category,
        }
        if extra_meta:
            meta.update(extra_meta)
        return self._memory.store(text, metadata=meta)

    def _reconcile(self, fact: ExtractedFact) -> str | None:
        """Decide what to do with ``fact`` given the top-k nearest stored entries.

        Threshold short-circuits before any LLM round-trip:

        - ``max_score >= near_dup_threshold`` (default 0.92) → return
          ``None``; the fact is effectively a duplicate.
        - ``max_score < ambiguous_threshold`` (default 0.75) OR no
          candidates → ADD without asking the LLM.

        In the ambiguous range we call the LLM once with RECONCILE_PROMPT
        and dispatch on ``op``:

        - ``ADD`` → store new.
        - ``UPDATE target_id`` → forget old, store new with
          ``metadata.supersedes = target_id``.
        - ``SUPERSEDE target_id`` → store new with supersedes pointer,
          then call ``memory.update_metadata(target_id, ...)`` to mark
          the old entry with ``superseded_by = new_id``. Preserves
          history (Zep-style "invalidate, don't delete").
        - ``NOOP`` → return ``None``.

        Any parse failure or unknown op in the ambiguous branch falls
        back to ADD — safer to keep a fact than lose it on a broken
        LLM reply. Logged at WARNING for operator visibility.
        """
        # Scope retrieve to this session so reconcile doesn't try to
        # merge across users. Also exclude already-superseded entries
        # from the similarity check — they're history, not live state.
        candidates = self._memory.retrieve(
            fact.text,
            k=5,
            where={
                "session_id": self._session_id,
                "type": "fact",
            },
        )
        live_candidates = [
            c for c in candidates
            if c.metadata.get("superseded_by") is None
        ]
        if not live_candidates:
            return self._add_fact(fact.text, category=fact.category)

        max_score = max(c.score for c in live_candidates)
        if max_score >= self._near_dup_threshold:
            return None
        if max_score < self._ambiguous_threshold:
            return self._add_fact(fact.text, category=fact.category)

        # Ambiguous range: consult the LLM.
        return self._reconcile_with_llm(fact, live_candidates)

    def _reconcile_with_llm(
        self,
        fact: ExtractedFact,
        candidates: list[object],  # list[MemoryHit]
    ) -> str | None:
        candidate_block = "\n".join(
            f"[id={c.node_id}] (score={c.score:.3f}) {c.text}"  # type: ignore[attr-defined]
            for c in candidates
        )
        prompt = RECONCILE_PROMPT.format(
            new_fact=fact.text, candidates=candidate_block
        )
        raw = self._llm.generate(prompt, max_tokens=256)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise TypeError(f"expected JSON object, got {type(parsed).__name__}")
            op = parsed["op"]
            target_id = parsed.get("target_id")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "reconcile: LLM reply unparseable (%s); falling back to ADD",
                type(exc).__name__,
                extra={
                    "event": "reconcile_parse_failure",
                    "session_id": self._session_id,
                    "reply_prefix": raw[:80],
                },
            )
            return self._add_fact(fact.text, category=fact.category)

        if op == "NOOP":
            return None
        if op == "ADD":
            return self._add_fact(fact.text, category=fact.category)
        if op == "UPDATE":
            if not isinstance(target_id, str) or target_id not in self._memory:
                logger.warning(
                    "reconcile: UPDATE missing valid target_id; falling back to ADD",
                    extra={
                        "event": "reconcile_update_no_target",
                        "session_id": self._session_id,
                    },
                )
                return self._add_fact(fact.text, category=fact.category)
            self._memory.forget(target_id)
            return self._add_fact(
                fact.text,
                category=fact.category,
                extra_meta={"supersedes": target_id},
            )
        if op == "SUPERSEDE":
            if not isinstance(target_id, str) or target_id not in self._memory:
                logger.warning(
                    "reconcile: SUPERSEDE missing valid target_id; "
                    "falling back to ADD",
                    extra={
                        "event": "reconcile_supersede_no_target",
                        "session_id": self._session_id,
                    },
                )
                return self._add_fact(fact.text, category=fact.category)
            new_id = self._add_fact(
                fact.text,
                category=fact.category,
                extra_meta={"supersedes": target_id},
            )
            self._memory.update_metadata(
                target_id,
                {
                    "superseded_by": new_id,
                    "superseded_at_step": self._memory._step,
                },
            )
            return new_id
        # Unknown op — log and fall back to ADD so we don't lose the fact.
        logger.warning(
            "reconcile: unknown op %r; falling back to ADD",
            op,
            extra={
                "event": "reconcile_unknown_op",
                "session_id": self._session_id,
            },
        )
        return self._add_fact(fact.text, category=fact.category)
