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

import concurrent.futures
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from types import TracebackType

    from soma.llm.backends import LLMBackend
    from soma.memory.api import MemoryHit, MemoryLayer

from soma.memory.conversational_prompts import (
    EXTRACT_PROMPT,
    RECONCILE_PROMPT,
    RESUMMARY_PROMPT,
    SUMMARY_PROMPT,
)

logger = logging.getLogger("soma.memory")

# Sentinel for the per-call ``user_id`` override on ``add_message`` /
# ``retrieve``. Distinguishes "caller did not pass user_id" (fall back to
# the constructor value) from "caller passed user_id=None" (explicit
# unscope / admin drill-down). Private by convention — never leak.
_UNSET: Any = object()

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
    - :meth:`flush` — drain pending async extractions (sync mode: no-op
      beyond the underlying MemoryLayer flush).
    - :meth:`close` / ``__enter__`` / ``__exit__`` — context-manager
      lifecycle; drains the background executor and shuts it down.

    The ``extractor_llm`` kwarg lets callers pin a stronger model for
    the two structured-JSON steps (extract + reconcile) while leaving
    free-form chat + summary on a smaller model. When unset, the
    main ``llm`` is used for everything (backward compat).

    ``extraction_mode="async"`` (Phase 22) runs extract+reconcile on a
    single background thread so ``add_message`` can return as soon as
    the raw turn is persisted. Raw-turn writes and summary rollover
    stay synchronous. Because of the GIL the speedup is **I/O overlap
    with the LLM network call**, not CPU parallelism — local CPU-bound
    backends will not see a win. ``max_workers=1`` is pinned so
    within-session extraction order is preserved.
    """

    def __init__(
        self,
        *,
        memory: MemoryLayer,
        llm: LLMBackend,
        extractor_llm: LLMBackend | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        near_dup_threshold: float = 0.92,
        ambiguous_threshold: float = 0.75,
        summary_every: int = 20,
        resummarize_every: int = 5,
        extract_assistant: bool = False,
        extraction_mode: Literal["sync", "async"] = "sync",
    ) -> None:
        """Build a ConversationalMemory wrapper.

        :param memory: underlying :class:`MemoryLayer`.
        :param llm: backend used for the rolling summary, and as the
            fallback for extract/reconcile when ``extractor_llm`` is not
            set. Small local chat models are fine here.
        :param extractor_llm: optional separate backend for the two
            structured-JSON steps (fact extraction + reconcile). Use a
            stronger JSON-reliable model (e.g. a 7B+ instruct) when
            ``llm`` is a tiny (3B-ish) chat model that struggles to emit
            strict JSON. When ``None`` (default), ``llm`` is used for
            all three prompt types.
        :param session_id: scope id; defaults to ``"default"``.
        :param user_id: optional per-user scope id. When set, every
            stored turn / fact / summary gets ``metadata.user_id``, and
            :meth:`retrieve` / :meth:`clear_session` / :meth:`supersede`
            are automatically scoped to this user. Enables multi-tenant
            deploys where several end-users share one bundle but each
            user's memory must be isolated. Pre-Phase-12 callers that
            leave this unset see byte-identical metadata to before.
        :param near_dup_threshold: cosine cutoff above which a new fact
            is treated as a near-duplicate (no LLM call).
        :param ambiguous_threshold: cosine cutoff below which a new
            fact is ADDed without asking the LLM.
        :param summary_every: roll a summary every N turns (set very
            large to disable).
        :param resummarize_every: every Mth rolled summary is re-derived
            from the raw turns only (bypassing the previous summary)
            to break the chained-summarization drift loop. Default 5.
            Set to ``0`` to disable re-summarization entirely and keep
            the pre-Phase-17 behaviour where every summary chains off
            the previous one.
        :param extract_assistant: if True, also extract facts from
            ``role="assistant"`` turns; defaults to user-only.
        :param extraction_mode: ``"sync"`` (default) runs LLM
            extract+reconcile inline on :meth:`add_message` — cheapest
            path, strict ordering, pre-Phase-22 behaviour. ``"async"``
            offloads extract+reconcile to a single-worker
            :class:`concurrent.futures.ThreadPoolExecutor` so
            :meth:`add_message` returns as soon as the raw turn is
            persisted; call :meth:`flush` (or use the context-manager
            protocol) before reading extracted facts. Python's GIL
            means the async win is I/O overlap with the LLM network
            call, not CPU parallelism — local CPU-bound backends will
            not benefit. ``max_workers`` is pinned to 1 so within-
            session extraction order is preserved.
        """
        self._memory = memory
        self._llm = llm
        # Fall back to the main llm when no dedicated extractor is set.
        # Storing the resolved backend (rather than keeping an Optional)
        # means every call site stays free of None-checks.
        self._extractor_llm = extractor_llm or llm
        # session_id default="default" so simple callers don't fight the
        # API. Multi-session callers pass an explicit id.
        self._session_id: str = session_id or "default"
        # Kept as Optional so "unset" is preserved across metadata
        # writes — see _user_meta() / _user_where() for the conversion
        # into metadata/where dicts.
        self._user_id: str | None = user_id
        if not 0.0 <= ambiguous_threshold <= near_dup_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= ambiguous <= near_dup <= 1, "
                f"got ambiguous={ambiguous_threshold}, near_dup={near_dup_threshold}"
            )
        self._near_dup_threshold: float = float(near_dup_threshold)
        self._ambiguous_threshold: float = float(ambiguous_threshold)
        self._summary_every: int = int(summary_every)
        # Phase 17: every Mth summary is re-derived from raw turns (no
        # previous-summary dependency) so chained summaries can't
        # compound drift over long sessions. 0 disables.
        self._resummarize_every: int = int(resummarize_every)
        self._summaries_generated: int = 0
        self._extract_assistant: bool = bool(extract_assistant)
        self._turn_counter: int = 0

        # Phase 22: optional async extraction. ``max_workers=1`` is
        # deliberate — it preserves within-session extraction ordering
        # (each add_message submits one future; FIFO serialization
        # matches submission order). The executor thread calls through
        # to the MemoryLayer directly; MemoryLayer is thread-safe per
        # Phase 1's WAL design so cross-thread writes are fine.
        if extraction_mode not in ("sync", "async"):
            raise ValueError(
                f"extraction_mode must be 'sync' or 'async', got {extraction_mode!r}"
            )
        self._extraction_mode: Literal["sync", "async"] = extraction_mode
        self._executor: ThreadPoolExecutor | None = None
        # list appended-to by the submitting thread and drained by
        # flush(). Python list.append is atomic under the GIL so a
        # dedicated lock isn't required here.
        self._pending_futures: list[Future[None]] = []
        if extraction_mode == "async":
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"soma-conv-{self._session_id}",
            )

    # ------------------------------------------------------------------
    # Multi-user scoping helpers (Phase 12)
    # ------------------------------------------------------------------
    def _resolve_user_id(self, override: Any) -> str | None:
        """Pick the active user_id for a call.

        ``override`` is either :data:`_UNSET` (caller didn't pass one →
        fall back to the constructor value) or any other value (which
        includes ``None`` for the explicit unscope).
        """
        if override is _UNSET:
            return self._user_id
        return override

    @staticmethod
    def _stamp_user_id(meta: dict[str, object], user_id: str | None) -> None:
        """Stamp ``metadata.user_id`` in place when ``user_id`` is set.

        When ``user_id is None`` we deliberately leave the key absent —
        pre-Phase-12 bundles must stay byte-identical on write when no
        user_id is in play.
        """
        if user_id is not None:
            meta["user_id"] = user_id

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
        raw = self._extractor_llm.generate(prompt, max_tokens=512)
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
    # Internals — reconcile
    # ------------------------------------------------------------------
    def _add_fact(
        self,
        text: str,
        *,
        category: str = "other",
        extra_meta: dict[str, object] | None = None,
        user_id: str | None = None,
    ) -> str:
        """Store a new fact entry tagged with this session. Returns node_id.

        ``user_id`` is stamped into metadata when set. Callers in the
        reconcile path already resolved the effective user_id for the
        current add_message call and thread it through here so the fact
        gets the same owner as the turn that produced it.
        """
        meta: dict[str, object] = {
            "session_id": self._session_id,
            "type": "fact",
            "category": category,
        }
        self._stamp_user_id(meta, user_id)
        if extra_meta:
            meta.update(extra_meta)
        return self._memory.store(text, metadata=meta)

    def _reconcile(
        self, fact: ExtractedFact, *, user_id: str | None = None
    ) -> str | None:
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

        ``user_id`` scopes the candidate search so reconcile never
        merges across users; it's also stamped into the new fact so
        the store stays owned by the turn's author.
        """
        # Scope retrieve to this session AND this user so reconcile
        # doesn't try to merge across users. Also exclude already-
        # superseded entries from the similarity check — they're
        # history, not live state.
        where: dict[str, object] = {
            "session_id": self._session_id,
            "type": "fact",
        }
        if user_id is not None:
            where["user_id"] = user_id
        candidates = self._memory.retrieve(
            fact.text,
            k=5,
            where=where,
        )
        live_candidates = [
            c for c in candidates
            if c.metadata.get("superseded_by") is None
        ]
        if not live_candidates:
            return self._add_fact(
                fact.text, category=fact.category, user_id=user_id
            )

        max_score = max(c.score for c in live_candidates)
        if max_score >= self._near_dup_threshold:
            return None
        if max_score < self._ambiguous_threshold:
            return self._add_fact(
                fact.text, category=fact.category, user_id=user_id
            )

        # Ambiguous range: consult the LLM.
        return self._reconcile_with_llm(fact, live_candidates, user_id=user_id)

    def _reconcile_with_llm(
        self,
        fact: ExtractedFact,
        candidates: list[object],  # list[MemoryHit]
        *,
        user_id: str | None = None,
    ) -> str | None:
        candidate_block = "\n".join(
            f"[id={c.node_id}] (score={c.score:.3f}) {c.text}"  # type: ignore[attr-defined]
            for c in candidates
        )
        prompt = RECONCILE_PROMPT.format(
            new_fact=fact.text, candidates=candidate_block
        )
        raw = self._extractor_llm.generate(prompt, max_tokens=256)
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
            return self._add_fact(
                fact.text, category=fact.category, user_id=user_id
            )

        if op == "NOOP":
            return None
        if op == "ADD":
            return self._add_fact(
                fact.text, category=fact.category, user_id=user_id
            )
        if op == "UPDATE":
            if not isinstance(target_id, str) or target_id not in self._memory:
                logger.warning(
                    "reconcile: UPDATE missing valid target_id; falling back to ADD",
                    extra={
                        "event": "reconcile_update_no_target",
                        "session_id": self._session_id,
                    },
                )
                return self._add_fact(
                    fact.text, category=fact.category, user_id=user_id
                )
            self._memory.forget(target_id)
            return self._add_fact(
                fact.text,
                category=fact.category,
                extra_meta={"supersedes": target_id},
                user_id=user_id,
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
                return self._add_fact(
                    fact.text, category=fact.category, user_id=user_id
                )
            new_id = self._add_fact(
                fact.text,
                category=fact.category,
                extra_meta={"supersedes": target_id},
                user_id=user_id,
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
        return self._add_fact(
            fact.text, category=fact.category, user_id=user_id
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_message(
        self,
        role: str,
        text: str,
        *,
        metadata: dict[str, object] | None = None,
        user_id: Any = _UNSET,
    ) -> None:
        """Ingest one conversational turn.

        Pipeline:
          1. Store the raw turn with ``metadata.type="turn"`` so
             LoCoMo-style evaluations that expect raw turns stay
             compatible.
          2. If ``role == "user"`` (or if ``extract_assistant=True``),
             extract atomic facts via the LLM and reconcile each.
          3. Advance the turn counter; every ``summary_every`` turns
             roll a summary entry with ``metadata.type="summary"``.

        ``metadata`` is merged into the raw turn's metadata. The
        ``session_id``, ``type``, ``role``, ``user_id`` keys are always
        set by this method and will override anything the caller passes.

        ``user_id`` overrides the constructor-set user_id for this one
        call. The sentinel default preserves the constructor value;
        pass ``user_id=None`` to explicitly drop the scope on this turn.
        """
        if not text or not text.strip():
            return
        effective_user = self._resolve_user_id(user_id)
        turn_meta: dict[str, object] = {}
        if metadata:
            turn_meta.update(metadata)
        turn_meta.update(
            {
                "session_id": self._session_id,
                "type": "turn",
                "role": role,
                "turn_index": self._turn_counter,
            }
        )
        self._stamp_user_id(turn_meta, effective_user)
        self._memory.store(text, metadata=turn_meta)

        should_extract = role == "user" or self._extract_assistant
        if should_extract:
            if self._executor is None:
                self._run_extract_reconcile(text, user_id=effective_user)
            else:
                fut = self._executor.submit(
                    self._run_extract_reconcile,
                    text,
                    user_id=effective_user,
                )
                self._pending_futures.append(fut)

        self._turn_counter += 1
        if (
            self._summary_every > 0
            and self._turn_counter % self._summary_every == 0
        ):
            self._roll_summary(user_id=effective_user)

    def _run_extract_reconcile(
        self, text: str, *, user_id: str | None = None
    ) -> None:
        """Extract atomic facts from ``text`` and reconcile each.

        Single helper so :meth:`add_message` can either call it inline
        (sync mode) or submit it to the background executor (async
        mode). In async mode this runs on the executor thread; both
        ``self._extractor_llm.generate`` and the reconcile writes to
        ``self._memory`` are safe to call from a worker thread — the
        MemoryLayer has per-Phase-1 WAL-backed thread safety.
        """
        facts = self._extract_facts(text)
        for f in facts:
            self._reconcile(f, user_id=user_id)

    def _roll_summary(self, *, user_id: str | None = None) -> None:
        """Summarize recent raw turns, store as a ``type=summary`` entry.

        The turns block is formatted as ``role: text`` lines in order.
        The resulting summary is stored with ``metadata.type=summary``
        so :meth:`get_summary` and :meth:`retrieve` can find it.

        Phase 17: every ``resummarize_every``-th summary is re-derived
        from the raw turns of the last ``resummarize_every ×
        summary_every`` turns using :data:`RESUMMARY_PROMPT`, bypassing
        the previous-summary input. This breaks the chained-
        summarization drift loop. ``resummarize_every=0`` disables the
        re-summary cadence and every summary falls through to the
        standard :data:`SUMMARY_PROMPT`.

        ``user_id`` scopes both the turns considered for the summary
        (so multi-tenant bundles don't leak across users) and the
        stored summary's own metadata.
        """
        recent_turns = self._session_entries(
            type_filter="turn", user_id=user_id
        )
        if not recent_turns:
            return

        # Decide re-summary vs. standard. The upcoming summary's
        # 1-indexed position is ``_summaries_generated + 1`` — re-
        # summary fires at M, 2M, 3M, … so the first resummary is the
        # Mth summary written, and the Nth chained summaries remain
        # 1..M-1, M+1..2M-1, etc.
        upcoming_number = self._summaries_generated + 1
        use_resummary = (
            self._resummarize_every > 0
            and upcoming_number % self._resummarize_every == 0
        )

        if use_resummary:
            # Re-derive from the last M × summary_every raw turns (cap
            # at what's available so short sessions still degrade
            # gracefully onto however many turns exist).
            window = self._resummarize_every * self._summary_every
            tail = recent_turns[-window:]
            prompt_template = RESUMMARY_PROMPT
        else:
            tail = recent_turns[-self._summary_every :]
            prompt_template = SUMMARY_PROMPT

        if not tail:
            return

        turns_block = "\n".join(
            f"{h.metadata.get('role', '?')}: {h.text}" for h in tail
        )
        prompt = prompt_template.format(turns=turns_block)
        summary_text = self._llm.generate(prompt, max_tokens=512).strip()
        if not summary_text:
            return
        summary_meta: dict[str, object] = {
            "session_id": self._session_id,
            "type": "summary",
            "summarized_turn_start": tail[0].metadata.get("turn_index"),
            "summarized_turn_end": tail[-1].metadata.get("turn_index"),
            "resummary": use_resummary,
        }
        self._stamp_user_id(summary_meta, user_id)
        self._memory.store(summary_text, metadata=summary_meta)
        # Count only after a successful write so a no-op generate()
        # (empty string) doesn't advance the cadence.
        self._summaries_generated += 1

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        include_superseded: bool = False,
        user_id: Any = _UNSET,
        where: dict[str, object] | None = None,
    ) -> list[MemoryHit]:
        """Retrieve the top-k relevant entries for ``query``.

        Scoped to this session by default. Already-superseded facts are
        filtered out unless ``include_superseded=True``. Facts,
        summaries, and raw turns compete on cosine score.

        ``user_id`` overrides the constructor-set user_id for this one
        call (sentinel default preserves the constructor value; pass
        ``user_id=None`` for the admin drill-down that sees every
        user's entries).

        ``where`` is an optional extra metadata filter, composed with
        the internal session/user/superseded filters via AND. Caller
        keys override the internal keys (advanced use — most callers
        should leave this None).
        """
        effective_user = self._resolve_user_id(user_id)
        final_where: dict[str, object] = {"session_id": self._session_id}
        if not include_superseded:
            final_where["superseded_by"] = {"$eq": None}
        if effective_user is not None:
            final_where["user_id"] = effective_user
        if where:
            # Caller-supplied keys win — lets advanced callers narrow
            # or relax a filter we set by default. Most callers pass
            # no where= and get the default session/user scoping.
            final_where.update(where)
        return self._memory.retrieve(query, k=k, where=final_where)

    def list_facts(self) -> list[MemoryHit]:
        """All fact entries for this session (excludes superseded)."""
        entries = self._session_entries(type_filter="fact")
        return [e for e in entries if e.metadata.get("superseded_by") is None]

    def get_summary(self) -> MemoryHit | None:
        """Return the most recent summary entry for this session, or None."""
        summaries = self._session_entries(type_filter="summary")
        if not summaries:
            return None
        return summaries[-1]

    def supersede(self, old_node_id: str, new_text: str) -> str:
        """Public helper: supersede an existing entry with ``new_text``.

        Writes the same ``supersedes`` / ``superseded_by`` pointer pair
        as the internal SUPERSEDE reconcile op, and returns the new
        entry's id.

        When this :class:`ConversationalMemory` was constructed with a
        ``user_id``, the target fact's ``metadata.user_id`` must match —
        otherwise this call raises :class:`PermissionError` so one
        tenant can't invalidate another tenant's entries on a shared
        bundle. Callers with no ``user_id`` set (pre-Phase-12 and
        single-tenant deploys) skip this check.
        """
        if old_node_id not in self._memory:
            raise KeyError(f"node_id {old_node_id!r} not found in memory")
        if self._user_id is not None:
            existing = self._memory.get(old_node_id)
            assert existing is not None  # contains-check above proved it
            owner = existing.metadata.get("user_id")
            if owner != self._user_id:
                raise PermissionError(
                    f"user {self._user_id!r} cannot supersede entry "
                    f"{old_node_id!r} owned by {owner!r}"
                )
        new_id = self._add_fact(
            new_text,
            category="other",
            extra_meta={"supersedes": old_node_id},
            user_id=self._user_id,
        )
        self._memory.update_metadata(
            old_node_id,
            {
                "superseded_by": new_id,
                "superseded_at_step": self._memory._step,
            },
        )
        return new_id

    def clear_session(self, *, keep_summaries: bool = True) -> int:
        """Delete all turns + facts for this session.

        Summaries are preserved by default so the long-term memory
        skeleton of the session survives a reset. Pass
        ``keep_summaries=False`` for a full wipe. Returns the number of
        entries removed.

        When this :class:`ConversationalMemory` was constructed with a
        ``user_id``, only entries owned by that user are cleared —
        other tenants sharing the bundle are untouched. Callers with
        no ``user_id`` set match pre-Phase-12 behaviour and clear
        every entry for the session.

        Superseded entries ARE removed here — they're scoped to the
        session being cleared, and retention is a separate concern
        (see docs/plans/... follow-up on GDPR-grade forgetting).

        In async-extraction mode any pending extract+reconcile futures
        are drained **before** the wipe so facts in flight at the time
        of the call land + are then deleted rather than leaking in
        after the clear.
        """
        # Drain pending async extractions first so late-arriving facts
        # don't resurrect a just-cleared session. Sync mode is a no-op.
        self.flush()
        to_delete: list[str] = []
        for nid in list(self._memory._ids):
            hit = self._memory.get(nid)
            if hit is None:
                continue
            if hit.metadata.get("session_id") != self._session_id:
                continue
            if (
                self._user_id is not None
                and hit.metadata.get("user_id") != self._user_id
            ):
                continue
            ent_type = hit.metadata.get("type")
            if ent_type == "summary" and keep_summaries:
                continue
            to_delete.append(nid)
        removed = 0
        for nid in to_delete:
            if self._memory.forget(nid):
                removed += 1
        return removed

    def flush(self, timeout: float | None = None) -> None:
        """Drain pending async extractions and hand off for durability.

        In sync mode (default) this only passes through to
        :meth:`MemoryLayer.flush` — the underlying layer has already
        fsynced on every write, but this call lets batched-mode
        bundles fsync pending state on orderly shutdown.

        In async mode this additionally blocks on every pending
        extract+reconcile future. Any exception raised on an executor
        thread is re-raised here on the first future to surface it —
        callers expecting fire-and-forget semantics should wrap
        ``flush()`` in try/except. Futures that have not completed
        within ``timeout`` (seconds; ``None`` = wait indefinitely) are
        left in ``self._pending_futures`` for a later :meth:`flush`.
        """
        if self._executor is not None and self._pending_futures:
            done, not_done = concurrent.futures.wait(
                self._pending_futures, timeout=timeout,
            )
            # Keep the not-yet-done futures for the next flush; drop the
            # completed ones after surfacing any exceptions they held.
            self._pending_futures = list(not_done)
            for fut in done:
                # ``result()`` re-raises whatever the worker raised, so
                # operator errors (LLM crash, parse blow-up, transient
                # network) surface here rather than being swallowed.
                fut.result()
        # Pass through to the MemoryLayer so an attached bundle fsyncs
        # any batched-mode state on callers' orderly shutdown.
        self._memory.flush()

    def close(self) -> None:
        """Flush pending extractions and shut down the background executor.

        Safe to call multiple times — subsequent calls are no-ops once
        the executor has been torn down. In sync mode :meth:`close`
        still calls :meth:`MemoryLayer.flush` via :meth:`flush` so the
        durability hand-off works the same either way.
        """
        if self._executor is None:
            # Sync mode (or already-closed async): still pass through
            # to the MemoryLayer so callers get the durability hand-off
            # whether or not they ever turned async extraction on.
            self._memory.flush()
            return
        try:
            self.flush()
        finally:
            # shutdown(wait=True) blocks until the single worker
            # finishes. We call it even if flush() re-raised so a
            # misbehaving extractor doesn't leak the executor thread.
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> ConversationalMemory:
        """Enter the context manager; returns ``self`` unchanged."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit: drain pending work and shut down the executor."""
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _session_entries(
        self,
        *,
        type_filter: str | None = None,
        user_id: Any = _UNSET,
    ) -> list[MemoryHit]:
        """Return this session's entries in insertion order, filtered by type.

        ``user_id`` controls the per-user scoping:

        - ``_UNSET`` (default): fall back to the constructor's user_id
          — i.e. scope to the wrapper's own user when one was set, or
          return all users when the wrapper has no user_id.
        - ``None``: explicitly unscoped (admin drill-down).
        - a string: scope to that user_id.

        Callers that pre-Phase-12 called ``_session_entries(type_filter=...)``
        on a wrapper without user_id see byte-identical behaviour; the
        sentinel default falls through to the constructor value, which
        is ``None`` for them.
        """
        effective_user = self._resolve_user_id(user_id)
        out: list[MemoryHit] = []
        for nid in self._memory._ids:
            hit = self._memory.get(nid)
            if hit is None:
                continue
            if hit.metadata.get("session_id") != self._session_id:
                continue
            if (
                effective_user is not None
                and hit.metadata.get("user_id") != effective_user
            ):
                continue
            if type_filter is not None and hit.metadata.get("type") != type_filter:
                continue
            out.append(hit)
        return out
